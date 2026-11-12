"""AgentPlanner: bridges natural-language requests, the RAGRetriever tool, and
PawPal+'s deterministic Schedule/Task engine.

Design boundary (see README "Design Decisions"): the LLM never computes
dates, priority order, recurrence, or conflict windows itself. It only
chooses which deterministic tool to call and with what arguments; all of
that math happens inside pawpal_system.py, untouched by this module.

Grounding boundary: `add_task` refuses to run (via a `PermissionError` that
comes back to the model as a tool error) until `search_guidelines` has been
called at least once in the same turn, so a proposed task is always checked
against the knowledge base first.

Uses the OpenAI API (`openai` package) for the LLM calls -- except in mock
mode (see "Offline mock mode" below), which never touches the network.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from openai import OpenAI

from pawpal_system import Owner, Pet, Schedule, Task
from src.rag import RAGRetriever

DEFAULT_MODEL = os.environ.get("PAWPAL_AGENT_MODEL", "gpt-4o-mini")
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = """You are the PawPal+ scheduling agent. You translate a pet \
owner's natural-language request into calls against PawPal+'s deterministic \
scheduling tools -- you never compute dates, durations, priority order, or \
conflict windows yourself; that math always happens inside the tools.

Before you add, reschedule, or otherwise plan any task, call \
`search_guidelines` at least once to check PawPal+'s internal pet care \
knowledge base (feeding rules, breed notes, general care guidelines) and \
ground your plan in what it returns. If the guidelines conflict with the \
user's request (e.g. an unsafe feeding frequency for the pet's age/breed), \
say so instead of silently complying.

After making changes, call `check_for_conflicts` to confirm the schedule is \
still valid before reporting back to the user. If it reports an overlap, \
resolve it (e.g. by rescheduling the lower-priority task) and check again \
before finishing.
"""

INTENT_SYSTEM_PROMPT = """Extract the pet-care scheduling intent from the \
user's message. Respond with a JSON object of exactly this shape:
{"action": "<one of add_task, list_pets, get_overdue_tasks, \
check_for_conflicts, generate_plan, mark_task_complete, query>", \
"entities": {<only the fields relevant to that action>}}
"""

MOCK_ACTIONS = (
    "add_task",
    "list_pets",
    "get_overdue_tasks",
    "check_for_conflicts",
    "generate_plan",
    "mark_task_complete",
    "query",
)


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_use_mock() -> bool:
    """Mock mode activates if USE_MOCK_LLM is truthy, or automatically if no
    OpenAI API key is configured -- so running the agent locally without
    credentials never fails, it just runs offline."""
    if _env_flag_true("USE_MOCK_LLM"):
        return True
    return not bool(os.environ.get("OPENAI_API_KEY"))


class AgentPlanner:
    """Routes a natural-language request through RAG grounding and deterministic tools.

    Offline mock mode
    ------------------
    Set the environment variable `USE_MOCK_LLM=true`, or simply don't set an
    `OPENAI_API_KEY` (the default auto-detects that), and AgentPlanner never
    calls the OpenAI API. Instead, a small rule-based stand-in picks which
    tool the "model" would have called next -- but every tool it picks still
    runs for real against RAGRetriever and the real Owner/Pet/Task/Schedule
    objects. Mock mode only replaces the LLM's reasoning step, never the
    deterministic tools it drives, so it's safe to use for local testing and
    demos without incurring API costs. Pass `use_mock=True`/`False`
    explicitly to override the auto-detection.
    """

    def __init__(
        self,
        owner: Owner,
        schedule: Schedule,
        retriever: RAGRetriever,
        *,
        model: str = DEFAULT_MODEL,
        client: Optional[OpenAI] = None,
        use_mock: Optional[bool] = None,
    ) -> None:
        self.owner = owner
        self.schedule = schedule
        self.retriever = retriever
        self.model = model
        self._client = client
        self.use_mock = _default_use_mock() if use_mock is None else use_mock
        self.audit_log: list[dict[str, Any]] = []
        self._rag_consulted = False

    @property
    def client(self) -> OpenAI:
        """Lazily constructed so importing/instantiating this class never
        requires API credentials -- only actually calling the real (non-mock)
        LLM path does."""
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    # ---- natural-language intent parsing -------------------------------

    def parse_intent(self, user_input: str) -> dict:
        """One-shot structured extraction of {action, entities} from free text."""
        if self.use_mock:
            action = self._classify_mock_action(user_input)
            entities = self._mock_extract_task_entities(user_input) if action == "add_task" else None
            return {"action": action, "entities": entities or {}}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    # ---- the agentic tool-use loop --------------------------------------

    def plan_workflow(self, user_input: str) -> dict:
        """Run the full agentic loop for one user request.

        Returns a dict with the agent's final text response, the ordered
        tool-call trace (for the README's "System Log" audit trail), and a
        deterministic conflict check from reflect_and_verify().
        """
        self._rag_consulted = False

        if self.use_mock:
            return self._run_mock_workflow(user_input)

        call_log: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        tools = self._tool_schemas()

        message = None
        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Invalid JSON arguments: {exc}",
                        }
                    )
                    call_log.append({"tool": name, "input": tool_call.function.arguments, "error": str(exc)})
                    continue

                try:
                    result = self._execute_tool(name, tool_input)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
                    )
                    call_log.append({"tool": name, "input": tool_input, "output": result})
                except Exception as exc:  # noqa: BLE001 -- surfaced to the model as a tool error
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(exc)})
                    call_log.append({"tool": name, "input": tool_input, "error": str(exc)})

        final_text = message.content if message is not None else ""

        self.audit_log.extend(call_log)
        conflicts_resolved = self.reflect_and_verify({"tool_calls": call_log})

        return {
            "response": final_text or "",
            "tool_calls": call_log,
            "stop_reason": message.finish_reason if message is not None else None,
            "conflicts_resolved": conflicts_resolved,
        }

    def reflect_and_verify(self, execution_results: dict) -> bool:
        """Deterministic post-execution check -- never re-asks the LLM.

        Re-runs Schedule.check_for_conflicts() against the current state and
        records any remaining conflicts onto execution_results. Returns True
        only when no overlapping scheduled tasks remain.
        """
        conflicts = self.schedule.check_for_conflicts()
        execution_results["remaining_conflicts"] = conflicts
        return not conflicts

    # ---- offline mock mode -----------------------------------------------
    #
    # Stands in for the LLM's *reasoning* only: which tool to call next, and
    # what to say about the results. Every tool it picks still runs for real
    # against RAGRetriever / Owner / Pet / Task / Schedule -- nothing here
    # bypasses or reimplements the deterministic core.

    def _classify_mock_action(self, user_input: str) -> str:
        text = user_input.lower()
        if "overdue" in text:
            return "get_overdue_tasks"
        if "conflict" in text:
            return "check_for_conflicts"
        if any(keyword in text for keyword in ("plan", "schedule for today", "today's schedule")):
            return "generate_plan"
        if any(keyword in text for keyword in ("complete", "done", "finished")):
            return "mark_task_complete"
        if any(
            keyword in text
            for keyword in ("add", "walk", "feed", "feeding", "medication", "meds", "groom", "schedule a")
        ):
            return "add_task"
        if "pet" in text or "list" in text:
            return "list_pets"
        return "query"

    def _mock_extract_task_entities(self, user_input: str) -> Optional[dict]:
        """Best-effort, regex/keyword-based stand-in for an LLM's slot filling."""
        text = user_input.lower()
        pet = next((p for p in self.owner.pets if p.name.lower() in text), None)
        if pet is None:
            return None

        duration_match = re.search(r"(\d+)\s*-?\s*min", text)
        duration_minutes = int(duration_match.group(1)) if duration_match else 30

        if "high" in text:
            priority = "high"
        elif "low" in text:
            priority = "low"
        else:
            priority = "medium"

        if "walk" in text:
            category = "walk"
        elif "feed" in text:
            category = "feeding"
        elif "med" in text:
            category = "meds"
        elif "groom" in text:
            category = "grooming"
        else:
            category = "general"

        due = datetime.now().replace(second=0, microsecond=0)
        if "tomorrow" in text:
            due += timedelta(days=1)

        return {
            "pet_id": pet.pet_id,
            "title": f"{category.title()} for {pet.name}",
            "description": f"Mock-generated {category} task from request: {user_input!r}",
            "category": category,
            "duration_minutes": duration_minutes,
            "priority": priority,
            "due_date": due.isoformat(),
        }

    def _mock_find_task_by_title(self, user_input: str) -> Optional[Task]:
        text = user_input.lower()
        for task in self.owner.get_all_tasks():
            if task.title.lower() in text:
                return task
        for task in self.owner.get_all_tasks():
            if any(word and word in text for word in task.title.lower().split()):
                return task
        return None

    def _run_mock_workflow(self, user_input: str) -> dict:
        call_log: list[dict[str, Any]] = []

        def run(name: str, tool_input: dict) -> Any:
            try:
                result = self._execute_tool(name, tool_input)
                call_log.append({"tool": name, "input": tool_input, "output": result})
                return result
            except Exception as exc:  # noqa: BLE001 -- recorded like a real tool error
                call_log.append({"tool": name, "input": tool_input, "error": str(exc)})
                raise

        # Mirrors the system prompt's rule: ground the plan before anything else.
        run("search_guidelines", {"query": user_input, "top_k": 3})

        action = self._classify_mock_action(user_input)

        if action == "get_overdue_tasks":
            overdue = run("get_overdue_tasks", {})
            if overdue:
                summary = "; ".join(f"{t['title']} (due {t['due_date']})" for t in overdue)
                final_text = f"Overdue right now: {summary}."
            else:
                final_text = "Nothing is overdue right now."

        elif action == "add_task":
            entities = self._mock_extract_task_entities(user_input)
            if entities is None:
                final_text = "I couldn't tell which pet that task was for, so I didn't add anything."
            else:
                try:
                    added = run("add_task", entities)
                except Exception as exc:
                    final_text = f"Couldn't add the task: {exc}"
                else:
                    conflicts = run("check_for_conflicts", {})
                    final_text = f"Added '{added['title']}' for the pet."
                    if conflicts:
                        final_text += f" Heads up -- conflict detected: {conflicts[0]}"

        elif action == "check_for_conflicts":
            conflicts = run("check_for_conflicts", {})
            final_text = "No conflicts found." if not conflicts else "; ".join(conflicts)

        elif action == "generate_plan":
            plan = run("generate_plan", {})
            final_text = f"Generated a plan with {len(plan)} task(s)."

        elif action == "mark_task_complete":
            task = self._mock_find_task_by_title(user_input)
            if task is None:
                final_text = "I couldn't find a matching task to mark complete."
            else:
                updated = run("mark_task_complete", {"task_id": task.task_id})
                final_text = f"Marked '{updated['title']}' complete."

        elif action == "list_pets":
            pets = run("list_pets", {})
            names = ", ".join(p["name"] for p in pets) if pets else "none yet"
            final_text = f"Pets on file: {names}."

        else:
            final_text = (
                "Mock agent: I consulted the guidelines but didn't recognize a "
                "specific scheduling action in that request."
            )

        self.audit_log.extend(call_log)
        conflicts_resolved = self.reflect_and_verify({"tool_calls": call_log})

        return {
            "response": final_text,
            "tool_calls": call_log,
            "stop_reason": "mock",
            "conflicts_resolved": conflicts_resolved,
        }

    # ---- tool schemas (OpenAI function-calling format) ---------------------

    @staticmethod
    def _to_openai_tool(anthropic_shaped_schema: dict) -> dict:
        """Wrap a {name, description, input_schema, strict} tool definition
        (e.g. RAGRetriever.tool_schema()) into OpenAI's function envelope,
        without modifying the source of that schema.

        OpenAI's strict mode requires every property to appear in
        "required" (optional ones representing "not provided" via a
        nullable type); schemas that don't already satisfy that fall back
        to non-strict function calling instead of being rewritten.
        """
        input_schema = anthropic_shaped_schema["input_schema"]
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        strict = bool(anthropic_shaped_schema.get("strict")) and set(properties) <= required
        return {
            "type": "function",
            "function": {
                "name": anthropic_shaped_schema["name"],
                "description": anthropic_shaped_schema["description"],
                "parameters": input_schema,
                "strict": strict,
            },
        }

    def _tool_schemas(self) -> list[dict]:
        return [
            self._to_openai_tool(self.retriever.tool_schema()),
            {
                "type": "function",
                "function": {
                    "name": "add_task",
                    "description": (
                        "Add a new care task for one of the owner's pets. Requires "
                        "search_guidelines to have been called first this turn."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pet_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "category": {
                                "type": "string",
                                "description": "e.g. walk, feeding, meds, grooming, enrichment",
                            },
                            "duration_minutes": {"type": "integer", "minimum": 1},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                            "due_date": {
                                "type": "string",
                                "description": "ISO 8601 datetime, e.g. 2026-11-12T09:00:00",
                            },
                            "recurrence": {"type": ["string", "null"], "enum": ["daily", "weekly", None]},
                        },
                        "required": [
                            "pet_id",
                            "title",
                            "description",
                            "category",
                            "duration_minutes",
                            "priority",
                            "due_date",
                            "recurrence",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_pets",
                    "description": "List the owner's pets and their profiles.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_overdue_tasks",
                    "description": "List all tasks that are overdue right now, across every pet.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_for_conflicts",
                    "description": "Check the full household schedule for overlapping scheduled tasks.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_plan",
                    "description": "Build today's (or a given date's) time-boxed plan from pending tasks due that day.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": ["string", "null"],
                                "description": "ISO 8601 date/datetime; defaults to today if omitted.",
                            },
                            "available_minutes": {"type": ["integer", "null"], "minimum": 1},
                        },
                        "required": ["date", "available_minutes"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_complete",
                    "description": (
                        "Mark a task complete by task_id (recurring tasks roll forward to "
                        "their next occurrence instead)."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                        "required": ["task_id"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # ---- tool execution (dispatches only to Owner/Pet/Task/Schedule's own API) ----

    def _find_pet(self, pet_id: str) -> Pet:
        for pet in self.owner.pets:
            if pet.pet_id == pet_id:
                return pet
        raise ValueError(f"No pet with id {pet_id!r} for this owner.")

    def _find_task(self, task_id: str) -> Task:
        for task in self.owner.get_all_tasks():
            if task.task_id == task_id:
                return task
        raise ValueError(f"No task with id {task_id!r}.")

    @staticmethod
    def _task_summary(task: Task) -> dict:
        return {
            "task_id": task.task_id,
            "pet_id": task.pet_id,
            "title": task.title,
            "category": task.category,
            "priority": task.priority,
            "duration_minutes": task.duration_minutes,
            "due_date": task.due_date.isoformat(),
            "scheduled_time": task.scheduled_time.isoformat() if task.scheduled_time else None,
            "recurrence": task.recurrence,
            "is_completed": task.is_completed,
        }

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        if name == "search_guidelines":
            self._rag_consulted = True
            return json.loads(self.retriever.handle_tool_call(tool_input))

        if name == "add_task":
            if not self._rag_consulted:
                raise PermissionError(
                    "Call search_guidelines before adding a task, so the task is "
                    "grounded in PawPal+'s care guidelines."
                )
            pet = self._find_pet(tool_input["pet_id"])
            task = Task(
                task_id=str(uuid.uuid4()),
                pet_id=pet.pet_id,
                title=tool_input["title"],
                description=tool_input.get("description", ""),
                category=tool_input["category"],
                duration_minutes=int(tool_input["duration_minutes"]),
                priority=tool_input["priority"],
                due_date=datetime.fromisoformat(tool_input["due_date"]),
                recurrence=tool_input.get("recurrence"),
            )
            pet.add_task(task)
            return self._task_summary(task)

        if name == "list_pets":
            return [pet.get_profile() for pet in self.owner.get_pets()]

        if name == "get_overdue_tasks":
            return [self._task_summary(t) for t in self.schedule.get_overdue_tasks()]

        if name == "check_for_conflicts":
            return self.schedule.check_for_conflicts()

        if name == "generate_plan":
            date = tool_input.get("date")
            plan = self.schedule.generate_plan(
                date=datetime.fromisoformat(date) if date else None,
                available_minutes=tool_input.get("available_minutes"),
            )
            return [self._task_summary(t) for t in plan]

        if name == "mark_task_complete":
            task = self._find_task(tool_input["task_id"])
            task.mark_complete()
            return self._task_summary(task)

        raise ValueError(f"Unknown tool: {name!r}")
