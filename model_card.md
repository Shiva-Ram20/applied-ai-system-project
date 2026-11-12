# Model Card: PawPal+ Agentic Task Manager

## System Limitations and Biases
* **Ambiguous Natural Language Parsing:** While the underlying deterministic engine (`pawpal_system.py`) executes schedules with 100% precision, the LLM planner can misinterpret vague or relative timeframes (e.g., "later this afternoon" or "next weekend") if explicit clock times are omitted.
* **Lack of Veterinary Medical Judgment:** The system treats all tasks purely as calendar blocks and priority weights. It has no clinical awareness of medication interactions, species-specific dosage limits, or emergency symptoms.
* **Assumption of Standard Daily Routines:** Default scheduling heuristics assume standard daytime hours and single-timezone operation, which may not fit users with irregular shift schedules or traveling pet owners.

## Potential Misuse and Prevention
* **Potential Misuse:** A user could over-rely on the AI to auto-schedule critical, time-sensitive medical treatments (such as insulin injections or post-surgery medications) without verifying the generated calendar entries, or attempt prompt injections to bypass input validation guardrails.
* **Prevention Strategies:** 
  * **Input & Tool Guardrails:** Strict schema validation ensures the LLM can only pass properly formatted parameters to backend functions.
  * **Human-in-the-Loop Confirmation:** High-priority medical tasks trigger explicit confirmation prompts and log full execution traces so the owner must review and approve critical schedule changes.
  * **Deterministic Conflict Checks:** The reflection loop automatically blocks overlapping tasks rather than silently overwriting existing reminders.

## Surprises During Reliability Testing
* **Environment vs. Logic Stability:** While migrating the legacy PawPal+ codebase into the new modular directory, all 28 deterministic unit tests passed immediately (`0.10s`), proving how resilient isolated Python functions are. However, setting up the fresh virtual environment revealed that missing test dependencies (`pytest`) can break verification workflows even when the core code is completely sound.
* **Formatting Brittleness:** During early prompt and documentation testing, small formatting inconsistencies (such as non-standard date strings or unescaped Markdown blocks) easily disrupted downstream parsing until strict structural rules were enforced.

## AI Collaboration Reflection
* **How I Collaborated with AI:** I used AI as an architectural thought partner and scaffolding assistant to design the Input-Process-Output diagram, structure the `README.md` documentation, and plan the separation between probabilistic LLM routing and deterministic Python tools.
* **Suggestion Accepted:** I accepted the AI's recommendation to keep all date-offset math (`timedelta`), sorting, and conflict detection strictly inside deterministic Python modules rather than letting the LLM calculate dates directly. This preserved my 28 passing `pytest` tests and eliminated arithmetic hallucinations.
* **Suggestion Rejected / Revised:** When migrating my legacy PawPal+ files into the new repository, the AI initially presented an option to delete my local folder, re-clone the old repository, and force-push the Git history. I rejected that approach because it risked overwriting my newly created `diagrams/`, `assets/`, and `README.md` files; instead, I selectively copied only the core Python modules and test suite into the clean repository structure.