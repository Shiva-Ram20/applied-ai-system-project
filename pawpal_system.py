"""
PawPal+ core domain model.

Classes:
    Owner    - a pet owner; manages multiple Pet objects
    Pet      - a pet belonging to an Owner; manages its own Task objects
    Task     - a single care task for a Pet (walk, feeding, meds, etc.)
    Schedule - the scheduling "brain": pulls every task across *all* of an
               owner's pets (via Owner.get_all_tasks()) and organizes/
               filters/plans them -- never reads from a single Pet alone,
               so multi-pet households are scheduled together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

# Lower rank = higher priority. Unknown/missing priority strings sort last.
PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# How far into the future each recurrence value pushes a task's due_date.
RECURRENCE_INTERVALS: dict[str, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
}

# Status checks usable with Schedule.filter_tasks(status=...).
STATUS_PREDICATES: dict[str, Callable[["Task"], bool]] = {
    "completed": lambda t: t.is_completed,
    "pending": lambda t: not t.is_completed,
    "overdue": lambda t: t.is_overdue(),
}


@dataclass
class Task:
    """A single care task for a pet (e.g., walk, feeding, meds, grooming)."""

    task_id: str
    pet_id: str
    title: str
    description: str
    category: str  # e.g., "walk", "feeding", "meds", "grooming", "enrichment"
    duration_minutes: int
    priority: str  # e.g., "low", "medium", "high"
    due_date: datetime  # deadline for one-off tasks; anchor date for recurring ones
    recurrence: Optional[str] = None  # e.g., "daily", "weekly", None for one-off
    # Start time assigned by Schedule.generate_plan() -- distinct from due_date
    # (a deadline), this is what's actually compared for overlap/conflicts.
    scheduled_time: Optional[datetime] = None
    is_completed: bool = False
    # Back-reference set by Pet.add_task(); excluded from repr/eq to avoid
    # Task <-> Pet recursion (Pet.tasks contains these Task objects).
    pet: Optional["Pet"] = field(default=None, repr=False, compare=False)

    def mark_complete(self) -> None:
        """Mark this task as completed.

        One-off tasks (recurrence is None) simply become completed. Recurring
        tasks ('daily'/'weekly') don't stay completed: due_date advances to
        the next occurrence (today's due_date + the recurrence's timedelta,
        via compute_next_occurrence()/RECURRENCE_INTERVALS), scheduled_time
        is cleared so it gets re-planned for that new date, and the task is
        left pending for that next occurrence.
        """
        if self.recurrence is not None:
            self.due_date = self.compute_next_occurrence()
            self.scheduled_time = None
            self.is_completed = False
        else:
            self.is_completed = True

    def mark_incomplete(self) -> None:
        """Mark this task as not completed."""
        self.is_completed = False

    def is_overdue(self, as_of: Optional[datetime] = None) -> bool:
        """Return True if this task is incomplete and past its due_date."""
        if self.is_completed:
            return False
        reference = as_of or datetime.now()
        return self.due_date < reference

    def reschedule(self, new_date: datetime) -> None:
        """Move this task's due_date and clear any assigned scheduled_time."""
        self.due_date = new_date
        self.scheduled_time = None

    def edit_details(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        priority: Optional[str] = None,
        due_date: Optional[datetime] = None,
        recurrence: Optional[str] = None,
    ) -> None:
        """Update one or more fields in place; omitted args are left unchanged."""
        if title is not None:
            self.title = title
        if description is not None:
            self.description = description
        if category is not None:
            self.category = category
        if duration_minutes is not None:
            self.duration_minutes = duration_minutes
        if priority is not None:
            self.priority = priority
        if due_date is not None:
            self.due_date = due_date
        if recurrence is not None:
            self.recurrence = recurrence

    def priority_rank(self) -> int:
        """Lower number = higher priority; unknown priority strings sort last."""
        return PRIORITY_ORDER.get(self.priority.lower(), len(PRIORITY_ORDER))

    def compute_next_occurrence(self) -> Optional[datetime]:
        """For recurring tasks: the due_date of the next instance after this one.

        Returns:
            None if this task isn't recurring; otherwise due_date advanced by
            the recurrence's interval (RECURRENCE_INTERVALS) -- a single
            interval, regardless of how overdue the current due_date is.

        Raises:
            ValueError: If `recurrence` is set to an unsupported value.
        """
        if self.recurrence is None:
            return None
        interval = RECURRENCE_INTERVALS.get(self.recurrence.lower())
        if interval is None:
            raise ValueError(f"Unsupported recurrence: {self.recurrence!r}")
        return self.due_date + interval

    def end_time(self) -> Optional[datetime]:
        """scheduled_time + duration, or None if not yet scheduled."""
        if self.scheduled_time is None:
            return None
        return self.scheduled_time + timedelta(minutes=self.duration_minutes)

    def overlaps_with(self, other: "Task") -> bool:
        """Check whether this task's scheduled_time/duration overlaps another's."""
        if self.scheduled_time is None or other.scheduled_time is None:
            return False
        return self.scheduled_time < other.end_time() and other.scheduled_time < self.end_time()


@dataclass
class Pet:
    """A pet belonging to an Owner. Owns (and is the source of truth for) its tasks."""

    pet_id: str
    name: str
    species: str
    breed: str
    age: int
    owner_id: str
    weight: Optional[float] = None
    tasks: list[Task] = field(default_factory=list)  # single source of truth for this pet's tasks
    # Back-reference, wired up by Owner.add_pet(). Excluded from repr/eq for
    # the same reason as Task.pet above.
    owner: Optional["Owner"] = field(default=None, repr=False, compare=False)

    def add_task(self, task: Task) -> None:
        """Attach a task to this pet, wiring up its back-reference."""
        task.pet = self
        task.pet_id = self.pet_id
        self.tasks.append(task)

    def remove_task(self, task_id: str) -> None:
        """Detach the task with the given id from this pet, if present."""
        remaining = []
        for task in self.tasks:
            if task.task_id == task_id:
                task.pet = None
            else:
                remaining.append(task)
        self.tasks = remaining

    def list_tasks(self) -> list[Task]:
        """Return a copy of this pet's tasks."""
        return list(self.tasks)

    def get_profile(self) -> dict:
        """Return this pet's profile fields as a dict."""
        return {
            "pet_id": self.pet_id,
            "name": self.name,
            "species": self.species,
            "breed": self.breed,
            "age": self.age,
            "owner_id": self.owner_id,
            "weight": self.weight,
        }

    def update_profile(self, **kwargs) -> None:
        """Update one or more profile fields in place, by keyword."""
        allowed = {"name", "species", "breed", "age", "weight"}
        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Cannot update unknown or protected field: {key!r}")
            setattr(self, key, value)


class Owner:
    """A pet owner who manages one or more pets."""

    def __init__(
        self,
        owner_id: str,
        name: str,
        email: str,
        phone: str,
        pets: Optional[list[Pet]] = None,
    ) -> None:
        """Create an owner, optionally with a starting list of pets."""
        self.owner_id = owner_id
        self.name = name
        self.email = email
        self.phone = phone
        self.pets: list[Pet] = pets if pets is not None else []
        # Set when a Schedule is created for this owner.
        self.schedule: Optional["Schedule"] = None

    def add_pet(self, pet: Pet) -> None:
        """Add a pet to this owner, wiring up its back-reference."""
        pet.owner = self
        pet.owner_id = self.owner_id
        self.pets.append(pet)

    def remove_pet(self, pet_id: str) -> None:
        """Remove the pet with the given id from this owner, if present."""
        remaining = []
        for pet in self.pets:
            if pet.pet_id == pet_id:
                pet.owner = None
            else:
                remaining.append(pet)
        self.pets = remaining

    def get_pets(self) -> list[Pet]:
        """Return a copy of this owner's pets."""
        return list(self.pets)

    def get_all_tasks(self) -> list[Task]:
        """Collect every task across all of this owner's pets."""
        return [task for pet in self.pets for task in pet.tasks]

    def update_contact_info(self, email: Optional[str] = None, phone: Optional[str] = None) -> None:
        """Update this owner's email and/or phone number, if provided."""
        if email is not None:
            self.email = email
        if phone is not None:
            self.phone = phone


class Schedule:
    """The scheduling "brain".

    Retrieves tasks by calling `owner.get_all_tasks()` (never by reading a
    single Pet's task list directly), so a household with several pets is
    organized, filtered, and planned as one combined schedule.
    """

    def __init__(self, schedule_id: str, owner: Owner) -> None:
        """Create a schedule for the given owner."""
        self.schedule_id = schedule_id
        self.owner = owner

    @property
    def tasks(self) -> list[Task]:
        """All tasks across every pet this owner has, read fresh each time."""
        return self.owner.get_all_tasks()

    def get_tasks_for_pet(self, pet_id: str) -> list[Task]:
        """Return only the tasks belonging to the given pet."""
        return [task for task in self.tasks if task.pet_id == pet_id]

    def get_daily_tasks(self, date: datetime) -> list[Task]:
        """Return all tasks due on the given date."""
        target_date = date.date()
        return [task for task in self.tasks if task.due_date.date() == target_date]

    def get_upcoming_tasks(self, days: int = 7) -> list[Task]:
        """Return incomplete tasks due within the next `days` days."""
        now = datetime.now()
        horizon = now + timedelta(days=days)
        return [
            task
            for task in self.tasks
            if not task.is_completed and now <= task.due_date <= horizon
        ]

    def get_overdue_tasks(self, as_of: Optional[datetime] = None) -> list[Task]:
        """Return all tasks that are overdue as of the given time (default: now)."""
        reference = as_of or datetime.now()
        return [task for task in self.tasks if task.is_overdue(reference)]

    def sort_by_priority(self, tasks: Optional[list[Task]] = None) -> list[Task]:
        """Highest priority first; ties broken by earlier due_date.

        Args:
            tasks: Pool to sort; defaults to every task across the owner's pets.

        Returns:
            A new sorted list -- the input pool (or self.tasks) is untouched.
        """
        pool = self.tasks if tasks is None else tasks
        return sorted(pool, key=lambda t: (t.priority_rank(), t.due_date))

    def filter_tasks(
        self, pet_id: Optional[str] = None, status: Optional[str] = None
    ) -> list[Task]:
        """Filter tasks by pet_id and/or status.

        Args:
            pet_id: Keep only tasks belonging to this pet; None matches every pet.
            status: One of 'completed', 'pending', 'overdue' (see
                STATUS_PREDICATES); None skips the status check entirely.

        Returns:
            Tasks across the owner's pets satisfying both filters (AND'd).

        Raises:
            ValueError: If `status` is given and isn't a supported value.
        """
        if status is not None and status not in STATUS_PREDICATES:
            raise ValueError(f"Unsupported status: {status!r}")
        status_ok = STATUS_PREDICATES.get(status, lambda t: True)
        return [
            task
            for task in self.tasks
            if (pet_id is None or task.pet_id == pet_id) and status_ok(task)
        ]

    def sort_by_time(self, tasks: Optional[list[Task]] = None) -> list[Task]:
        """Chronological order by scheduled_time; unscheduled tasks sort last.

        Args:
            tasks: Pool to sort; defaults to every task across the owner's pets.

        Returns:
            A new sorted list -- the input pool (or self.tasks) is untouched.
        """
        pool = self.tasks if tasks is None else tasks
        return sorted(pool, key=lambda t: (t.scheduled_time is None, t.scheduled_time))

    def find_conflicts(self) -> list[tuple[Task, Task]]:
        """Return pairs of scheduled tasks whose time windows overlap.

        Only tasks with a scheduled_time assigned are considered; tasks are
        compared chronologically so each overlapping pair is reported once,
        earlier-starting task first. See check_for_conflicts() for a version
        that returns printable warnings instead of raw Task pairs.
        """
        scheduled = sorted(
            (t for t in self.tasks if t.scheduled_time is not None),
            key=lambda t: t.scheduled_time,
        )
        conflicts: list[tuple[Task, Task]] = []
        for i, task in enumerate(scheduled):
            for other in scheduled[i + 1 :]:
                if task.overlaps_with(other):
                    conflicts.append((task, other))
        return conflicts

    def check_for_conflicts(self) -> list[str]:
        """Lightweight conflict check: same idea as find_conflicts(), but
        returns ready-to-print warning strings instead of raw Task pairs,
        and never lets a bad task derail the check.

        Any unexpected error while comparing a pair of tasks (e.g. a corrupt
        scheduled_time/duration on one of them) is caught and turned into its
        own warning line rather than propagating and crashing the caller.
        """
        warnings: list[str] = []
        try:
            conflicts = self.find_conflicts()
        except Exception as exc:  # defensive: never crash the caller over a scheduling check
            return [f"Could not check for scheduling conflicts: {exc}"]

        for first, second in conflicts:
            try:
                first_window = f"{first.scheduled_time:%I:%M %p}-{first.end_time():%I:%M %p}"
                second_window = f"{second.scheduled_time:%I:%M %p}-{second.end_time():%I:%M %p}"
                warnings.append(
                    f"Conflict: '{first.title}' ({first_window}) overlaps with "
                    f"'{second.title}' ({second_window})"
                )
            except Exception as exc:  # defensive: formatting shouldn't crash the check either
                warnings.append(
                    f"Conflict detected between '{first.title}' and '{second.title}' "
                    f"(could not format times: {exc})"
                )
        return warnings

    def generate_plan(
        self,
        date: Optional[datetime] = None,
        available_minutes: Optional[int] = None,
    ) -> list[Task]:
        """Build an ordered, time-boxed plan for `date` (default: today).

        Incomplete tasks due that day are sorted by priority (ties broken
        by due_date), then greedily packed back-to-back -- skipping any
        task that would exceed `available_minutes` -- until all candidates
        are placed or time runs out. Each selected task gets a
        `scheduled_time` assigned.
        """
        target_date = (date or datetime.now()).date()
        candidates = [
            task
            for task in self.tasks
            if not task.is_completed and task.due_date.date() == target_date
        ]
        ordered = self.sort_by_priority(candidates)

        plan: list[Task] = []
        minutes_used = 0
        current_time = datetime.combine(target_date, datetime.min.time())

        for task in ordered:
            if available_minutes is not None and minutes_used + task.duration_minutes > available_minutes:
                continue
            task.scheduled_time = current_time
            current_time += timedelta(minutes=task.duration_minutes)
            minutes_used += task.duration_minutes
            plan.append(task)

        return plan
