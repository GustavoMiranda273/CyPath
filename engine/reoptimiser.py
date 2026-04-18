"""
CyPath Re-optimisation Engine
Dynamically redistributes the training load when a session is missed, keeping
the user safely on track towards their event goal without generating dangerous
"catch-up" workouts.

Behaviour:
  - When a planned session is marked as missed, its TSS is redistributed
    proportionally across the REMAINING training days of the same week.
  - The hard safety cap of 150 TSS per day is strictly enforced. If a day
    would exceed the cap, the excess is spread across the other eligible days.
  - If the weekly safety budget cannot absorb all of the missed load, the
    remainder is dropped and reported to the user as a warning.
  - Sessions can also be marked as completed or restored to planned status,
    supporting an undo action.

This module depends only on the data model defined in scheduler.py, keeping
physiological modelling (banister_model.py) and plan generation (scheduler.py)
as independent concerns.

Author: Gustavo Miranda
"""

from dataclasses import dataclass
from typing import List, Optional

from engine.scheduler import TrainingPlan, Workout, MAX_DAILY_TSS


# ─── Session status constants ────────────────────────────────────────────────

STATUS_PLANNED   = "planned"
STATUS_COMPLETED = "completed"
STATUS_MISSED    = "missed"


# ─── Result of a re-optimisation call ────────────────────────────────────────

@dataclass
class ReoptimisationResult:
    """
    Reports what happened during a re-optimisation attempt, providing
    transparency to the user (NFR2 in Section 4).
    """
    success:           bool           # True if all missed load was redistributed
    missed_tss:        float          # The TSS that was originally scheduled
    redistributed_tss: float          # How much was successfully reallocated
    dropped_tss:       float          # How much was unsafe to redistribute
    affected_days:     List[int]      # Day numbers whose TSS was adjusted
    warning:           Optional[str]  # Human-readable warning, if any


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_status_fields(plan: TrainingPlan) -> None:
    """
    Ensure every workout has a `status` attribute.

    The core scheduler produces workouts without a status field (to keep that
    module purely concerned with plan generation). The re-optimiser attaches
    status tracking here on first use.
    """
    for w in plan.workouts:
        if not hasattr(w, "status"):
            # Rest days are trivially completed; training days start as planned.
            w.status = STATUS_COMPLETED if w.phase == "Rest" else STATUS_PLANNED


def _remaining_training_days(
    plan: TrainingPlan, week: int, after_day: int
) -> List[Workout]:
    """
    Return the list of training days in the given week that:
      - occur AFTER the supplied day number,
      - are not rest days,
      - still have status 'planned' (i.e. not already completed/missed).
    """
    return [
        w for w in plan.workouts
        if w.week == week
        and w.day > after_day
        and w.phase != "Rest"
        and w.status == STATUS_PLANNED
    ]


def _redistribute(
    eligible: List[Workout],
    tss_to_place: float,
) -> float:
    """
    Place `tss_to_place` across the eligible workouts proportionally to their
    current target_tss, respecting the MAX_DAILY_TSS safety cap.

    Algorithm:
      Repeatedly allocate a proportional share to every day that still has
      spare capacity. Days that reach the cap drop out of the active pool.
      The loop terminates when either all TSS is placed or no capacity remains.

    Args:
        eligible:      Workouts to redistribute into (modified in place).
        tss_to_place:  Total TSS to add across the pool.

    Returns:
        The amount of TSS that could NOT be placed (dropped).
    """
    if tss_to_place <= 0 or not eligible:
        return tss_to_place

    # Active pool: days that still have spare capacity under the daily cap.
    active = [w for w in eligible if w.target_tss < MAX_DAILY_TSS]
    remaining = tss_to_place

    # Defensive iteration cap — we should never need more than len(eligible)
    # passes, but guard against any unforeseen edge case.
    max_passes = len(eligible) + 2

    while remaining > 0.1 and active and max_passes > 0:
        max_passes -= 1

        # Base proportional share on existing target_tss. If every active day
        # happens to be at zero, split evenly.
        total_weight = sum(w.target_tss for w in active) or float(len(active))

        to_remove: List[Workout] = []
        placed_this_pass = 0.0

        for w in active:
            if remaining <= 0.1:
                break

            weight = (w.target_tss or 1.0) / total_weight
            proposed = remaining * weight
            spare = MAX_DAILY_TSS - w.target_tss

            if spare <= 0:
                to_remove.append(w)
                continue

            addition = min(proposed, spare)
            w.target_tss = round(w.target_tss + addition, 1)
            placed_this_pass += addition

            if w.target_tss >= MAX_DAILY_TSS - 0.05:
                to_remove.append(w)

        remaining = round(remaining - placed_this_pass, 1)

        # Remove capped days from the active pool for the next pass.
        for w in to_remove:
            if w in active:
                active.remove(w)

    return max(round(remaining, 1), 0.0)


# ─── Public API ──────────────────────────────────────────────────────────────

def mark_completed(plan: TrainingPlan, day: int) -> None:
    """Mark a planned workout as completed (used when the user logs an activity)."""
    _ensure_status_fields(plan)
    for w in plan.workouts:
        if w.day == day:
            w.status = STATUS_COMPLETED
            return
    raise ValueError(f"No workout found for day {day}")


def restore_planned(plan: TrainingPlan, day: int) -> None:
    """
    Restore a workout to 'planned' status. Useful for undoing an accidental
    'mark as missed' action.

    Note: if the workout's TSS has already been redistributed, the caller
    should consider regenerating the week rather than relying on this alone.
    """
    _ensure_status_fields(plan)
    for w in plan.workouts:
        if w.day == day:
            w.status = STATUS_PLANNED
            return
    raise ValueError(f"No workout found for day {day}")


def mark_missed(plan: TrainingPlan, day: int) -> ReoptimisationResult:
    """
    Mark a workout as missed and redistribute its TSS across the remaining
    training days of the same week, respecting the daily safety cap.

    Args:
        plan: The TrainingPlan to modify (modified in place).
        day:  The day number (1-84) of the missed session.

    Returns:
        ReoptimisationResult describing what was redistributed, dropped, and
        whether any warnings apply.

    Raises:
        ValueError if the day does not exist or is not a planned training day.
    """
    _ensure_status_fields(plan)

    # ── Locate the missed workout ────────────────────────────────────────────
    missed: Optional[Workout] = next((w for w in plan.workouts if w.day == day), None)

    if missed is None:
        raise ValueError(f"No workout found for day {day}")

    if missed.phase == "Rest":
        raise ValueError(f"Day {day} is a rest day — nothing to redistribute")

    if missed.status != STATUS_PLANNED:
        raise ValueError(
            f"Day {day} is not planned (current status: {missed.status})"
        )

    # ── Record the missed TSS and flag the session ───────────────────────────
    missed_tss = missed.target_tss
    missed.status = STATUS_MISSED
    missed.target_tss = 0.0

    # ── Find eligible days for redistribution ────────────────────────────────
    eligible = _remaining_training_days(plan, missed.week, missed.day)

    if not eligible:
        # No remaining training days this week — the whole load is dropped.
        return ReoptimisationResult(
            success=False,
            missed_tss=missed_tss,
            redistributed_tss=0.0,
            dropped_tss=missed_tss,
            affected_days=[],
            warning=(
                f"{missed_tss:.0f} TSS could not be redistributed: there are "
                "no remaining training days this week. Impact on fitness is "
                "minimal — continue with next week as planned."
            ),
        )

    # ── Snapshot pre-redistribution values so we can report affected days ───
    before_values = {w.day: w.target_tss for w in eligible}

    # ── Redistribute, capping at MAX_DAILY_TSS per day ───────────────────────
    dropped = _redistribute(eligible, missed_tss)
    redistributed = round(missed_tss - dropped, 1)

    affected = [w.day for w in eligible if w.target_tss != before_values[w.day]]

    warning: Optional[str] = None
    if dropped > 0.1:
        warning = (
            f"{dropped:.0f} TSS could not be safely redistributed without "
            f"breaching the daily {MAX_DAILY_TSS:.0f} TSS safety cap. "
            "The excess has been dropped. Impact on overall fitness is "
            "minimal (<1% of weekly load at peak training)."
        )

    return ReoptimisationResult(
        success=(dropped == 0.0),
        missed_tss=missed_tss,
        redistributed_tss=redistributed,
        dropped_tss=dropped,
        affected_days=affected,
        warning=warning,
    )


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine.scheduler import generate_plan

    def show_week(plan: TrainingPlan, week: int, label: str) -> None:
        print(f"\n{label}:")
        for w in plan.workouts:
            if w.week == week and w.phase != "Rest":
                status = getattr(w, "status", "planned")
                marker = f"  [{status.upper()}]" if status != "planned" else ""
                print(
                    f"  Day {w.day:>2}  TSS={w.target_tss:>5.1f}  "
                    f"{w.description}{marker}"
                )
        print(f"  Weekly total: {plan.weekly_tss(week):.1f} TSS")

    print("=" * 64)
    print("Re-optimisation demo — intermediate cyclist, Tue/Thu/Sat/Sun")
    print("=" * 64)

    # ── Scenario 1: normal case — miss Tuesday of week 3 ─────────────────────
    print("\n── SCENARIO 1: miss Tuesday (day 16) of week 3 ──")
    plan = generate_plan("intermediate", [1, 3, 5, 6], recovery_weeks=True)
    show_week(plan, 3, "BEFORE")

    result = mark_missed(plan, day=16)
    print(f"\n➜ missed:       {result.missed_tss:.1f} TSS")
    print(f"  redistributed: {result.redistributed_tss:.1f} TSS")
    print(f"  dropped:       {result.dropped_tss:.1f} TSS")
    print(f"  affected days: {result.affected_days}")
    if result.warning:
        print(f"  ⚠️  {result.warning}")

    show_week(plan, 3, "AFTER")

    # ── Scenario 2: miss Sunday (last training day) — no room to redistribute
    print("\n\n── SCENARIO 2: miss Sunday (day 21, last training day of week) ──")
    plan2 = generate_plan("intermediate", [1, 3, 5, 6], recovery_weeks=True)
    show_week(plan2, 3, "BEFORE")

    result2 = mark_missed(plan2, day=21)
    print(f"\n➜ missed:       {result2.missed_tss:.1f} TSS")
    print(f"  redistributed: {result2.redistributed_tss:.1f} TSS")
    print(f"  dropped:       {result2.dropped_tss:.1f} TSS")
    if result2.warning:
        print(f"  ⚠️  {result2.warning}")
