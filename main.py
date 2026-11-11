"""Demo script: build an Owner with multiple pets/tasks and exercise the
Schedule class's sorting and filtering methods from the terminal.
"""

from datetime import datetime, timedelta

from pawpal_system import Owner, Pet, Task, Schedule


def build_demo_owner() -> Owner:
    owner = Owner(
        owner_id="o1",
        name="Jordan",
        email="jordan@example.com",
        phone="555-0100",
    )

    biscuit = Pet(
        pet_id="p1",
        name="Biscuit",
        species="dog",
        breed="Golden Retriever",
        age=3,
        owner_id=owner.owner_id,
    )
    mochi = Pet(
        pet_id="p2",
        name="Mochi",
        species="cat",
        breed="Tabby",
        age=2,
        owner_id=owner.owner_id,
    )
    owner.add_pet(biscuit)
    owner.add_pet(mochi)

    today = datetime.now().replace(second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    # Tasks are added deliberately out of priority/due-date order below, so
    # that printing them afterward with sort_by_priority()/sort_by_time()
    # actually proves those methods reorder the list instead of just
    # reflecting whatever order they were inserted in.
    mochi.add_task(
        Task(
            task_id="t4",
            pet_id=mochi.pet_id,
            title="Evening playtime",
            description="Interactive wand toy session",
            category="enrichment",
            duration_minutes=15,
            priority="low",
            due_date=today.replace(hour=18, minute=30),
        )
    )
    biscuit.add_task(
        Task(
            task_id="t2",
            pet_id=biscuit.pet_id,
            title="Breakfast",
            description="Morning feeding",
            category="feeding",
            duration_minutes=10,
            priority="high",
            due_date=today.replace(hour=8, minute=30),
        )
    )
    mochi.add_task(
        Task(
            task_id="t3",
            pet_id=mochi.pet_id,
            title="Litter box cleaning",
            description="Scoop and refresh litter",
            category="cleaning",
            duration_minutes=10,
            priority="medium",
            due_date=today.replace(hour=12, minute=0),
        )
    )
    biscuit.add_task(
        Task(
            task_id="t1",
            pet_id=biscuit.pet_id,
            title="Morning walk",
            description="30-minute walk around the block",
            category="walk",
            duration_minutes=30,
            priority="high",
            due_date=today.replace(hour=8, minute=0),
        )
    )

    # Extra tasks specifically to exercise filter_tasks()'s status filter.
    overdue_task = Task(
        task_id="t5",
        pet_id=biscuit.pet_id,
        title="Give heartworm medication",
        description="Monthly chewable, missed yesterday",
        category="meds",
        duration_minutes=5,
        priority="high",
        due_date=yesterday.replace(hour=9, minute=0),
    )
    biscuit.add_task(overdue_task)

    completed_task = Task(
        task_id="t6",
        pet_id=mochi.pet_id,
        title="Nail trim",
        description="Trim front claws",
        category="grooming",
        duration_minutes=15,
        priority="low",
        due_date=today.replace(hour=10, minute=0),
    )
    completed_task.mark_complete()
    mochi.add_task(completed_task)

    return owner


def print_tasks(label: str, tasks: list[Task]) -> None:
    print(label)
    print("-" * len(label))
    if not tasks:
        print("  (none)")
    for task in tasks:
        pet_name = task.pet.name if task.pet else task.pet_id
        status = "done" if task.is_completed else "pending"
        time_label = (
            task.scheduled_time.strftime("%I:%M %p") if task.scheduled_time else "unscheduled"
        )
        print(
            f"  [{task.priority:<6}] due {task.due_date.strftime('%Y-%m-%d %I:%M %p')}"
            f"  (sched: {time_label})  {task.title:<26} ({pet_name}) - {status}"
        )
    print()


def print_schedule(owner: Owner) -> None:
    today = datetime.now()
    schedule = owner.schedule

    todays_tasks = sorted(schedule.get_daily_tasks(today), key=lambda t: t.due_date)

    print(f"Today's Schedule -- {today.strftime('%A, %B %d, %Y')}")
    print("=" * 56)

    if not todays_tasks:
        print("No tasks scheduled for today.")
        return

    for task in todays_tasks:
        pet_name = task.pet.name if task.pet else task.pet_id
        time_label = task.due_date.strftime("%I:%M %p")
        status = "done" if task.is_completed else "pending"
        print(
            f"{time_label}  {task.title:<20} ({pet_name}, {task.duration_minutes} min) "
            f"[priority: {task.priority:<6}] - {status}"
        )
    print()


def demo_sorting_and_filtering(owner: Owner) -> None:
    schedule = owner.schedule

    print_tasks("All tasks (insertion order -- deliberately scrambled)", schedule.tasks)
    print_tasks("sort_by_priority() -- highest priority first, ties by due_date", schedule.sort_by_priority())

    print_tasks("filter_tasks(pet_id='p1') -- Biscuit only", schedule.filter_tasks(pet_id="p1"))
    print_tasks("filter_tasks(status='completed')", schedule.filter_tasks(status="completed"))
    print_tasks("filter_tasks(status='pending')", schedule.filter_tasks(status="pending"))
    print_tasks("filter_tasks(status='overdue')", schedule.filter_tasks(status="overdue"))
    print_tasks(
        "filter_tasks(pet_id='p2', status='pending') -- combined filter",
        schedule.filter_tasks(pet_id="p2", status="pending"),
    )

    # sort_by_time() and find_conflicts() both key off scheduled_time, which
    # generate_plan() only ever assigns back-to-back (never overlapping), so
    # assign two overlapping times by hand here to actually exercise conflict
    # detection -- this simulates a user manually pinning two tasks to
    # clashing start times.
    tasks_by_id = {task.task_id: task for task in schedule.tasks}
    base = datetime.now().replace(second=0, microsecond=0)
    tasks_by_id["t4"].scheduled_time = base.replace(hour=18, minute=30)  # Evening playtime, 15 min
    tasks_by_id["t3"].scheduled_time = base.replace(hour=12, minute=0)  # Litter box cleaning, 10 min
    tasks_by_id["t1"].scheduled_time = base.replace(hour=8, minute=0)  # Morning walk, 30 min
    tasks_by_id["t2"].scheduled_time = base.replace(hour=8, minute=15)  # Breakfast, 10 min -- overlaps t1

    print_tasks("sort_by_time() -- chronological by scheduled_time", schedule.sort_by_time())

    label = "check_for_conflicts()"
    print(label)
    print("-" * len(label))
    warnings = schedule.check_for_conflicts()
    if not warnings:
        print("  (none)")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print()


if __name__ == "__main__":
    demo_owner = build_demo_owner()
    demo_owner.schedule = Schedule(schedule_id="s1", owner=demo_owner)

    print_schedule(demo_owner)
    demo_sorting_and_filtering(demo_owner)
