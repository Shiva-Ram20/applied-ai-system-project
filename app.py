import uuid
from datetime import datetime, time

import streamlit as st

from pawpal_system import Owner, Pet, Schedule, Task

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to PawPal+, a pet care planning assistant.

Add your pet(s), give them some care tasks, then generate today's schedule.
"""
)

with st.expander("Scenario", expanded=False):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.
"""
    )

st.divider()

# --- Persistent app state -----------------------------------------------
# The Owner (and everything hanging off it -- its Pets, and each Pet's
# Tasks) is created once and stored in st.session_state. Every later rerun
# (button click, widget change, etc.) reuses that same object instead of
# rebuilding it, so pets/tasks added earlier don't disappear.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(
        owner_id=str(uuid.uuid4()),
        name="Jordan",
        email="",
        phone="",
    )

owner: Owner = st.session_state.owner

if "schedule" not in st.session_state:
    st.session_state.schedule = Schedule(schedule_id=str(uuid.uuid4()), owner=owner)

schedule: Schedule = st.session_state.schedule

# --- Owner info -----------------------------------------------------------
st.subheader("Owner Info")
owner.name = st.text_input("Owner name", value=owner.name)
owner.email = st.text_input("Email", value=owner.email)

st.divider()

# --- Add a pet --------------------------------------------------------------
st.subheader("Pets")

col1, col2, col3 = st.columns(3)
with col1:
    new_pet_name = st.text_input("Pet name", value="Mochi")
with col2:
    new_species = st.selectbox("Species", ["dog", "cat", "other"])
with col3:
    new_age = st.number_input("Age", min_value=0, max_value=30, value=1)

if st.button("Add pet"):
    if new_pet_name:
        pet = Pet(
            pet_id=str(uuid.uuid4()),
            name=new_pet_name,
            species=new_species,
            breed="Unknown",
            age=int(new_age),
            owner_id=owner.owner_id,
        )
        owner.add_pet(pet)
        st.success(f"Added {pet.name}.")
    else:
        st.warning("Enter a pet name first.")

if not owner.pets:
    st.info("No pets yet. Add one above.")
    st.stop()

pet_names = {pet.pet_id: pet.name for pet in owner.pets}
selected_pet_id = st.selectbox(
    "Select a pet to manage tasks for",
    options=list(pet_names.keys()),
    format_func=lambda pid: pet_names[pid],
)
selected_pet = next(pet for pet in owner.pets if pet.pet_id == selected_pet_id)

st.divider()

# --- Add tasks for the selected pet -----------------------------------------
st.subheader(f"Tasks for {selected_pet.name}")

col1, col2, col3 = st.columns(3)
with col1:
    task_title = st.text_input("Task title", value="Morning walk")
with col2:
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
with col3:
    priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)

col4, col5 = st.columns(2)
with col4:
    due_date_input = st.date_input("Due date", value=datetime.now().date())
with col5:
    due_time_input = st.time_input("Due time", value=time(9, 0))

if st.button("Add task"):
    task = Task(
        task_id=str(uuid.uuid4()),
        pet_id=selected_pet.pet_id,
        title=task_title,
        description="",
        category="general",
        duration_minutes=int(duration),
        priority=priority,
        due_date=datetime.combine(due_date_input, due_time_input),
    )
    selected_pet.add_task(task)
    st.success(f"Added '{task.title}' for {selected_pet.name}.")

if selected_pet.tasks:
    st.write(f"Current tasks for {selected_pet.name}:")
    st.table(
        [
            {
                "Title": t.title,
                "Due": t.due_date.strftime("%Y-%m-%d %I:%M %p"),
                "Duration (min)": t.duration_minutes,
                "Priority": t.priority,
                "Completed": t.is_completed,
            }
            for t in selected_pet.tasks
        ]
    )
else:
    st.info("No tasks yet for this pet. Add one above.")

st.divider()

# --- Generate schedule -------------------------------------------------------
st.subheader("Today's Schedule")
st.caption("Pulls tasks across every pet this owner has, not just the one selected above.")

if st.button("Generate schedule"):
    schedule.generate_plan(datetime.now())
    st.session_state.plan_generated = True

# --- Conflict warnings -----------------------------------------------------
# Shown prominently at the top so overlapping scheduled_time windows are the
# first thing the owner notices, before any sorting/filtering happens.
conflict_warnings = schedule.check_for_conflicts()
if conflict_warnings:
    for warning_text in conflict_warnings:
        st.warning(warning_text)

pet_lookup = {pet.pet_id: pet.name for pet in owner.pets}

# --- Sorting & filtering controls -------------------------------------------
sort_col, pet_filter_col, status_filter_col = st.columns(3)
with sort_col:
    sort_choice = st.selectbox("Sort by", ["Priority", "Scheduled time"])
with pet_filter_col:
    pet_filter_ids = st.multiselect(
        "Filter by pet",
        options=list(pet_lookup.keys()),
        format_func=lambda pid: pet_lookup[pid],
        default=list(pet_lookup.keys()),
    )
with status_filter_col:
    status_choice = st.selectbox("Filter by status", ["All", "Pending", "Completed", "Overdue"])

status_lookup = {"All": None, "Pending": "pending", "Completed": "completed", "Overdue": "overdue"}
status_value = status_lookup[status_choice]

filtered_tasks: list = []
seen_task_ids: set[str] = set()
for pid in pet_filter_ids:
    for task in schedule.filter_tasks(pet_id=pid, status=status_value):
        if task.task_id not in seen_task_ids:
            filtered_tasks.append(task)
            seen_task_ids.add(task.task_id)

if sort_choice == "Priority":
    display_tasks = schedule.sort_by_priority(filtered_tasks)
else:
    display_tasks = schedule.sort_by_time(filtered_tasks)

# --- Summary metrics ---------------------------------------------------------
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total tasks", len(schedule.tasks))
metric_col2.metric("Shown", len(display_tasks))
metric_col3.metric("Completed", len(schedule.filter_tasks(status="completed")))

if display_tasks:
    st.table(
        [
            {
                "Scheduled time": (
                    t.scheduled_time.strftime("%I:%M %p") if t.scheduled_time else "Not scheduled"
                ),
                "Task": t.title,
                "Pet": pet_lookup.get(t.pet_id, t.pet_id),
                "Duration (min)": t.duration_minutes,
                "Priority": t.priority,
                "Status": "done" if t.is_completed else "pending",
            }
            for t in display_tasks
        ]
    )
    st.success(f"Showing {len(display_tasks)} task(s).")
else:
    st.info("No tasks match the selected filters.")
