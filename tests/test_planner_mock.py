"""Offline tests for AgentPlanner's mock mode -- no network, no API key.

These exercise the real deterministic core (Owner/Pet/Task/Schedule) and the
real RAGRetriever; only the "which tool would the LLM call next" decision is
mocked.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pawpal_system import Owner, Pet, Schedule, Task
from src.planner import AgentPlanner, _default_use_mock
from src.rag import RAGRetriever

PAST_DATE = datetime(2000, 1, 1, 8, 0)


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make mock-mode auto-detection deterministic regardless of the host env."""
    monkeypatch.delenv("USE_MOCK_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def owner_with_pet() -> Owner:
    owner = Owner(owner_id="o1", name="Alex", email="alex@example.com", phone="555-0100")
    biscuit = Pet(pet_id="p1", name="Biscuit", species="dog", breed="Golden Retriever", age=3, owner_id="o1")
    owner.add_pet(biscuit)
    overdue_task = Task(
        task_id="t1",
        pet_id="p1",
        title="Morning walk",
        description="30-minute walk",
        category="walk",
        duration_minutes=30,
        priority="high",
        due_date=PAST_DATE,
    )
    biscuit.add_task(overdue_task)
    return owner


@pytest.fixture
def schedule(owner_with_pet: Owner) -> Schedule:
    return Schedule(schedule_id="s1", owner=owner_with_pet)


@pytest.fixture
def retriever(tmp_path: Path) -> RAGRetriever:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "feeding.md").write_text("## Puppy Feeding\nFeed puppies three to four times a day.\n")
    return RAGRetriever(kb)


def test_use_mock_auto_detected_without_api_key(owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever) -> None:
    planner = AgentPlanner(owner_with_pet, schedule, retriever)
    assert planner.use_mock is True
    assert _default_use_mock() is True


def test_mock_overdue_prompt_calls_search_then_get_overdue(
    owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever
) -> None:
    planner = AgentPlanner(owner_with_pet, schedule, retriever)

    result = planner.plan_workflow("What's overdue right now?")

    tool_names = [call["tool"] for call in result["tool_calls"]]
    assert tool_names == ["search_guidelines", "get_overdue_tasks"]
    assert "Morning walk" in result["response"]
    assert result["stop_reason"] == "mock"
    assert result["conflicts_resolved"] is True


def test_mock_add_task_prompt_grounds_then_adds_task(
    owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever
) -> None:
    planner = AgentPlanner(owner_with_pet, schedule, retriever)

    result = planner.plan_workflow("Add a 20-minute walk for Biscuit today, high priority")

    tool_names = [call["tool"] for call in result["tool_calls"]]
    assert tool_names == ["search_guidelines", "add_task", "check_for_conflicts"]

    # The task was really added to the real deterministic Pet object.
    biscuit = next(p for p in owner_with_pet.pets if p.pet_id == "p1")
    new_tasks = [t for t in biscuit.tasks if t.task_id != "t1"]
    assert len(new_tasks) == 1
    assert new_tasks[0].duration_minutes == 20
    assert new_tasks[0].priority == "high"
    assert new_tasks[0].category == "walk"


def test_mock_add_task_without_rag_call_still_enforced(
    owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever
) -> None:
    """Even in mock mode, add_task's grounding guard is the real one."""
    planner = AgentPlanner(owner_with_pet, schedule, retriever)
    planner._rag_consulted = False

    with pytest.raises(PermissionError):
        planner._execute_tool(
            "add_task",
            {
                "pet_id": "p1",
                "title": "Walk",
                "description": "",
                "category": "walk",
                "duration_minutes": 20,
                "priority": "high",
                "due_date": (datetime.now() + timedelta(hours=1)).isoformat(),
            },
        )


def test_parse_intent_mock_mode_no_network(
    owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever
) -> None:
    planner = AgentPlanner(owner_with_pet, schedule, retriever)
    intent = planner.parse_intent("What's overdue right now?")
    assert intent["action"] == "get_overdue_tasks"


def test_explicit_use_mock_false_overrides_auto_detect(
    owner_with_pet: Owner, schedule: Schedule, retriever: RAGRetriever
) -> None:
    planner = AgentPlanner(owner_with_pet, schedule, retriever, use_mock=False)
    assert planner.use_mock is False
