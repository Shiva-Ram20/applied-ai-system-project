"""Entry point for PawPal+'s agentic workflow.

Routes a natural-language request through AgentPlanner (grounded by
RAGRetriever) and executes it against the same deterministic Owner/Pet/
Task/Schedule engine the rest of PawPal+ uses. Requires an Anthropic API
key (see README) -- the plain deterministic demo in main.py has no such
dependency and is unaffected by this file.

Usage:
    python agent_main.py "Add a 20-minute walk for Biscuit today, high priority"
"""

from __future__ import annotations

import sys

from main import build_demo_owner
from pawpal_system import Schedule
from src.planner import AgentPlanner
from src.rag import RAGRetriever


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What's overdue right now?"

    owner = build_demo_owner()
    owner.schedule = Schedule(schedule_id="s1", owner=owner)

    retriever = RAGRetriever("knowledge_base")
    planner = AgentPlanner(owner, owner.schedule, retriever)

    print(f"User: {prompt}\n")
    result = planner.plan_workflow(prompt)

    if result["tool_calls"]:
        print("Tool calls:")
        for call in result["tool_calls"]:
            detail = call.get("output", call.get("error"))
            print(f"  - {call['tool']}({call['input']}) -> {detail}")
        print()

    print(f"Agent: {result['response']}")

    if not result["conflicts_resolved"]:
        print(f"\n(!) Unresolved conflicts: {result['remaining_conflicts']}")


if __name__ == "__main__":
    main()
