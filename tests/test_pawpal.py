from datetime import datetime, timedelta

import pytest

from pawpal_system import Owner, Pet, Schedule, Task

# Fixed reference dates (not datetime.now()-relative) so overdue/pending
# assertions never depend on when the test suite happens to run.
PAST_DATE = datetime(2000, 1, 1, 8, 0)
FUTURE_DATE = datetime(2999, 1, 1, 8, 0)


def make_task(
    task_id: str = "t1",
    pet_id: str = "p1",
    title: str = "Morning walk",
    duration_minutes: int = 30,
    priority: str = "high",
    due_date: datetime = datetime(2026, 1, 1, 8, 0),
    recurrence: str | None = None,
    scheduled_time: datetime | None = None,
    is_completed: bool = False,
) -> Task:
    """Build a Task with sensible defaults; override only what a test cares about."""
    task = Task(
        task_id=task_id,
        pet_id=pet_id,
        title=title,
        description="30-minute walk around the block",
        category="walk",
        duration_minutes=duration_minutes,
        priority=priority,
        due_date=due_date,
        recurrence=recurrence,
    )
    task.scheduled_time = scheduled_time
    task.is_completed = is_completed
    return task


@pytest.fixture
def owner_with_two_pets() -> Owner:
    """An owner with two pets, each pet starting with an empty task list."""
    owner = Owner(owner_id="o1", name="Alex", email="alex@example.com", phone="555-0100")
    biscuit = Pet(pet_id="p1", name="Biscuit", species="dog", breed="Golden Retriever", age=3, owner_id="o1")
    whiskers = Pet(pet_id="p2", name="Whiskers", species="cat", breed="Tabby", age=5, owner_id="o1")
    owner.add_pet(biscuit)
    owner.add_pet(whiskers)
    return owner


@pytest.fixture
def schedule(owner_with_two_pets: Owner) -> Schedule:
    return Schedule(schedule_id="s1", owner=owner_with_two_pets)


def add_tasks(owner: Owner, pet_id: str, *tasks: Task) -> None:
    """Attach each task to the owner's pet with the given pet_id."""
    pet = next(p for p in owner.pets if p.pet_id == pet_id)
    for task in tasks:
        pet.add_task(task)


# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


def test_task_completion():
    task = make_task()
    assert task.is_completed is False

    task.mark_complete()

    assert task.is_completed is True


def test_task_addition():
    pet = Pet(
        pet_id="p1",
        name="Biscuit",
        species="dog",
        breed="Golden Retriever",
        age=3,
        owner_id="o1",
    )
    assert len(pet.tasks) == 0

    pet.add_task(make_task(task_id="t1", pet_id=pet.pet_id))

    assert len(pet.tasks) == 1


# ---------------------------------------------------------------------------
# 1) Sorting correctness: Schedule.sort_by_time()
# ---------------------------------------------------------------------------


def test_sort_by_time_orders_tasks_chronologically(schedule, owner_with_two_pets):
    # Added out of order on purpose; sort_by_time() should return them
    # earliest-first regardless of insertion order.
    noon = make_task(task_id="noon", scheduled_time=datetime(2026, 1, 1, 12, 0))
    morning = make_task(task_id="morning", scheduled_time=datetime(2026, 1, 1, 8, 0))
    evening = make_task(task_id="evening", scheduled_time=datetime(2026, 1, 1, 18, 0))
    add_tasks(owner_with_two_pets, "p1", noon, morning, evening)

    ordered = schedule.sort_by_time()

    assert [t.task_id for t in ordered] == ["morning", "noon", "evening"]


def test_sort_by_time_pushes_unscheduled_tasks_last(schedule, owner_with_two_pets):
    scheduled = make_task(task_id="scheduled", scheduled_time=datetime(2026, 1, 1, 9, 0))
    unscheduled = make_task(task_id="unscheduled", scheduled_time=None)
    add_tasks(owner_with_two_pets, "p1", unscheduled, scheduled)

    ordered = schedule.sort_by_time()

    assert [t.task_id for t in ordered] == ["scheduled", "unscheduled"]


def test_sort_by_time_does_not_mutate_the_underlying_task_list(schedule, owner_with_two_pets):
    first = make_task(task_id="first", scheduled_time=datetime(2026, 1, 1, 15, 0))
    second = make_task(task_id="second", scheduled_time=datetime(2026, 1, 1, 9, 0))
    add_tasks(owner_with_two_pets, "p1", first, second)

    schedule.sort_by_time()

    # schedule.tasks reads straight from the pets, so insertion order survives
    # a sort_by_time() call -- sort_by_time() must return a new list, not
    # reorder in place.
    assert [t.task_id for t in schedule.tasks] == ["first", "second"]


# ---------------------------------------------------------------------------
# 2) Recurrence logic: Task.mark_complete()
# ---------------------------------------------------------------------------


def test_mark_complete_daily_task_advances_due_date_by_one_day():
    due_date = datetime(2026, 3, 10, 8, 0)
    task = make_task(recurrence="daily", due_date=due_date, scheduled_time=datetime(2026, 3, 10, 9, 0))

    task.mark_complete()

    assert task.due_date == due_date + timedelta(days=1)
    assert task.is_completed is False  # recurring tasks stay pending for the next occurrence
    assert task.scheduled_time is None  # cleared so it can be re-planned


def test_mark_complete_weekly_task_advances_due_date_by_seven_days():
    due_date = datetime(2026, 3, 10, 8, 0)
    task = make_task(recurrence="weekly", due_date=due_date)

    task.mark_complete()

    assert task.due_date == due_date + timedelta(days=7)
    assert task.is_completed is False


def test_mark_complete_one_off_task_completes_without_changing_due_date():
    due_date = datetime(2026, 3, 10, 8, 0)
    task = make_task(recurrence=None, due_date=due_date)

    task.mark_complete()

    assert task.is_completed is True
    assert task.due_date == due_date


def test_mark_complete_is_case_insensitive_for_recurrence():
    due_date = datetime(2026, 3, 10, 8, 0)
    task = make_task(recurrence="Daily", due_date=due_date)

    task.mark_complete()

    assert task.due_date == due_date + timedelta(days=1)


def test_mark_complete_unsupported_recurrence_raises_value_error():
    task = make_task(recurrence="monthly", due_date=datetime(2026, 3, 10, 8, 0))

    with pytest.raises(ValueError):
        task.mark_complete()


def test_mark_complete_advances_by_a_single_interval_even_if_overdue():
    # A daily task that's been overdue for a week should only jump one day
    # ahead per mark_complete() call -- it does not "catch up" to today.
    due_date = datetime(2026, 1, 1, 8, 0)
    task = make_task(recurrence="daily", due_date=due_date)

    task.mark_complete()

    assert task.due_date == datetime(2026, 1, 2, 8, 0)


def test_mark_complete_twice_advances_due_date_cumulatively():
    due_date = datetime(2026, 1, 1, 8, 0)
    task = make_task(recurrence="daily", due_date=due_date)

    task.mark_complete()
    task.mark_complete()

    assert task.due_date == due_date + timedelta(days=2)
    assert task.is_completed is False


# ---------------------------------------------------------------------------
# 3) Conflict detection: Schedule.check_for_conflicts()
# ---------------------------------------------------------------------------


def test_check_for_conflicts_flags_overlapping_tasks(schedule, owner_with_two_pets):
    walk = make_task(
        task_id="walk", title="Walk", duration_minutes=30,
        scheduled_time=datetime(2026, 1, 1, 9, 0),
    )
    vet = make_task(
        task_id="vet", title="Vet visit", duration_minutes=30,
        scheduled_time=datetime(2026, 1, 1, 9, 15),  # starts mid-walk
    )
    add_tasks(owner_with_two_pets, "p1", walk, vet)

    warnings = schedule.check_for_conflicts()

    assert len(warnings) == 1
    assert "Walk" in warnings[0]
    assert "Vet visit" in warnings[0]
    assert "Conflict" in warnings[0]


def test_check_for_conflicts_flags_duplicate_scheduled_times(schedule, owner_with_two_pets):
    same_time = datetime(2026, 1, 1, 9, 0)
    feeding = make_task(task_id="feeding", title="Feeding", duration_minutes=15, scheduled_time=same_time)
    meds = make_task(task_id="meds", title="Meds", duration_minutes=15, scheduled_time=same_time)
    add_tasks(owner_with_two_pets, "p1", feeding, meds)

    warnings = schedule.check_for_conflicts()

    assert len(warnings) == 1


def test_check_for_conflicts_back_to_back_tasks_are_not_flagged(schedule, owner_with_two_pets):
    # Second task starts exactly when the first ends -- adjacent, not
    # overlapping, so this must NOT be reported as a conflict.
    first = make_task(task_id="first", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 0))
    second = make_task(task_id="second", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 30))
    add_tasks(owner_with_two_pets, "p1", first, second)

    warnings = schedule.check_for_conflicts()

    assert warnings == []


def test_check_for_conflicts_ignores_unscheduled_tasks(schedule, owner_with_two_pets):
    scheduled = make_task(task_id="scheduled", scheduled_time=datetime(2026, 1, 1, 9, 0))
    unscheduled = make_task(task_id="unscheduled", scheduled_time=None)
    add_tasks(owner_with_two_pets, "p1", scheduled, unscheduled)

    warnings = schedule.check_for_conflicts()

    assert warnings == []


def test_check_for_conflicts_returns_empty_when_no_tasks_scheduled(schedule):
    assert schedule.check_for_conflicts() == []


def test_check_for_conflicts_reports_each_overlapping_pair_once(schedule, owner_with_two_pets):
    # Three tasks all overlapping one another (9:00-9:30, 9:10-9:40, 9:20-9:50)
    # should produce exactly the 3 overlapping pairs, not duplicates.
    a = make_task(task_id="a", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 0))
    b = make_task(task_id="b", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 10))
    c = make_task(task_id="c", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 20))
    add_tasks(owner_with_two_pets, "p1", a, b, c)

    warnings = schedule.check_for_conflicts()

    assert len(warnings) == 3


def test_check_for_conflicts_never_raises_on_a_corrupt_task(schedule, owner_with_two_pets):
    # duration_minutes must be numeric for end_time()'s timedelta math; a
    # corrupt value should surface as a warning, not crash the caller.
    good = make_task(task_id="good", duration_minutes=30, scheduled_time=datetime(2026, 1, 1, 9, 0))
    corrupt = make_task(task_id="corrupt", duration_minutes="oops", scheduled_time=datetime(2026, 1, 1, 9, 10))
    add_tasks(owner_with_two_pets, "p1", good, corrupt)

    warnings = schedule.check_for_conflicts()

    assert len(warnings) == 1
    assert "Could not check for scheduling conflicts" in warnings[0]


# ---------------------------------------------------------------------------
# 4) Filtering logic: Schedule.filter_tasks()
# ---------------------------------------------------------------------------


def test_filter_tasks_by_pet_id(schedule, owner_with_two_pets):
    biscuit_task = make_task(task_id="b1", pet_id="p1")
    whiskers_task = make_task(task_id="w1", pet_id="p2")
    add_tasks(owner_with_two_pets, "p1", biscuit_task)
    add_tasks(owner_with_two_pets, "p2", whiskers_task)

    result = schedule.filter_tasks(pet_id="p1")

    assert [t.task_id for t in result] == ["b1"]


def test_filter_tasks_by_pet_id_with_no_matches_returns_empty(schedule, owner_with_two_pets):
    add_tasks(owner_with_two_pets, "p1", make_task(task_id="b1", pet_id="p1"))

    assert schedule.filter_tasks(pet_id="does-not-exist") == []


def test_filter_tasks_by_status_completed(schedule, owner_with_two_pets):
    done = make_task(task_id="done", is_completed=True)
    pending = make_task(task_id="pending", is_completed=False)
    add_tasks(owner_with_two_pets, "p1", done, pending)

    result = schedule.filter_tasks(status="completed")

    assert [t.task_id for t in result] == ["done"]


def test_filter_tasks_by_status_pending(schedule, owner_with_two_pets):
    done = make_task(task_id="done", is_completed=True)
    pending = make_task(task_id="pending", is_completed=False)
    add_tasks(owner_with_two_pets, "p1", done, pending)

    result = schedule.filter_tasks(status="pending")

    assert [t.task_id for t in result] == ["pending"]


def test_filter_tasks_by_status_overdue(schedule, owner_with_two_pets):
    overdue = make_task(task_id="overdue", due_date=PAST_DATE, is_completed=False)
    not_due_yet = make_task(task_id="future", due_date=FUTURE_DATE, is_completed=False)
    add_tasks(owner_with_two_pets, "p1", overdue, not_due_yet)

    result = schedule.filter_tasks(status="overdue")

    assert [t.task_id for t in result] == ["overdue"]


def test_filter_tasks_overdue_excludes_completed_tasks(schedule, owner_with_two_pets):
    # A task past its due_date that's already completed is NOT overdue --
    # is_overdue() short-circuits on is_completed before checking the date.
    completed_but_late = make_task(task_id="late-but-done", due_date=PAST_DATE, is_completed=True)
    add_tasks(owner_with_two_pets, "p1", completed_but_late)

    assert schedule.filter_tasks(status="overdue") == []


def test_filter_tasks_combines_pet_id_and_status(schedule, owner_with_two_pets):
    biscuit_pending = make_task(task_id="b-pending", pet_id="p1", is_completed=False)
    biscuit_done = make_task(task_id="b-done", pet_id="p1", is_completed=True)
    whiskers_pending = make_task(task_id="w-pending", pet_id="p2", is_completed=False)
    add_tasks(owner_with_two_pets, "p1", biscuit_pending, biscuit_done)
    add_tasks(owner_with_two_pets, "p2", whiskers_pending)

    result = schedule.filter_tasks(pet_id="p1", status="pending")

    assert [t.task_id for t in result] == ["b-pending"]


def test_filter_tasks_with_no_filters_returns_all_tasks(schedule, owner_with_two_pets):
    add_tasks(owner_with_two_pets, "p1", make_task(task_id="b1", pet_id="p1"))
    add_tasks(owner_with_two_pets, "p2", make_task(task_id="w1", pet_id="p2"))

    result = schedule.filter_tasks()

    assert {t.task_id for t in result} == {"b1", "w1"}


def test_filter_tasks_rejects_unsupported_status(schedule, owner_with_two_pets):
    add_tasks(owner_with_two_pets, "p1", make_task(task_id="b1"))

    with pytest.raises(ValueError):
        schedule.filter_tasks(status="not-a-real-status")
