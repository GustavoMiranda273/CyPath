"""
CyPath Banister Model
Implements the Banister Fitness-Fatigue (Impulse-Response) model.

The model tracks two competing physiological processes driven by daily
Training Stress Score (TSS):

  Fitness  (CTL) — a slow-decaying measure of chronic training adaptation
                    (τ = 42 days, the 'positive' function)
  Fatigue  (ATL) — a fast-decaying measure of acute training stress
                    (τ = 7 days, the 'negative' function)
  Readiness (TSB) = Fitness - Fatigue

compute_curve() extends the core class to project a full 84-day curve
from a TrainingPlan, combining historical (completed/missed) days with
future projections from planned sessions.

Author: Gustavo Miranda
References: Banister (1991); Hellard et al. (2006)
"""

import math
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.scheduler import TrainingPlan


# ─── Day-level snapshot returned by compute_curve() ─────────────────────────

@dataclass
class DaySnapshot:
    """Physiological state at the end of a single day."""
    day:       int    # 1-84
    week:      int    # 1-12
    ctl:       float  # Chronic Training Load (Fitness)
    atl:       float  # Acute Training Load (Fatigue)
    tsb:       float  # Training Stress Balance (Readiness = CTL - ATL)
    tss_used:  float  # TSS that was applied on this day
    projected: bool   # False = based on actual logged data, True = future projection


class BanisterModel:
    """
    Implementation of the Banister Fitness-Fatigue (Training Impulse) model.
    Tracks chronic training load (fitness) and acute training load (fatigue)
    using exponential decay.
    """

    def __init__(self, initial_fitness: float = 0.0, initial_fatigue: float = 0.0):
        """
        Initialize the Banister model with baseline metrics.

        Args:
            initial_fitness (float): Starting chronic training load.
            initial_fatigue (float): Starting acute training load.
        """
        # Standard physiological time constants (in days)
        self.tau_fitness = 42.0
        self.tau_fatigue = 7.0
        
        self.fitness = initial_fitness
        self.fatigue = initial_fatigue

    def add_daily_load(self, training_load: float):
        """
        Update fitness and fatigue scores based on daily training load.

        Args:
            training_load (float): The quantifiable stress score of the workout. 
                                   Input 0.0 for a rest day.
        """
        # Calculate exponential decay factors based on respective time constants
        fitness_decay_factor = math.exp(-1 / self.tau_fitness)
        fatigue_decay_factor = math.exp(-1 / self.tau_fatigue)
        
        # Apply decay to existing scores and integrate new training load
        self.fitness = (self.fitness * fitness_decay_factor) + (training_load * (1 - fitness_decay_factor))
        self.fatigue = (self.fatigue * fatigue_decay_factor) + (training_load * (1 - fatigue_decay_factor))

    def get_readiness(self) -> float:
        """
        Calculate the current Training Stress Balance (Readiness).

        Returns:
            float: The difference between fitness and fatigue.
        """
        return self.fitness - self.fatigue


# ─── Plan-level fitness curve computation ────────────────────────────────────

def compute_curve(
    plan: "TrainingPlan",
    today_day: int = 1,
) -> List[DaySnapshot]:
    """
    Walk all 84 days of a TrainingPlan and compute the CTL/ATL/TSB value at
    the end of each day, returning a full fitness curve.

    For days up to and including today_day:
      - Completed sessions:  use the workout's target_tss as the actual load
      - Missed sessions:     use 0.0 TSS (the session did not happen)
      - Rest days:           use 0.0 TSS (no training)

    For days after today_day (future projections):
      - Planned sessions:    use target_tss as the projected load
      - Rest days:           use 0.0 TSS

    The initial CTL and ATL are seeded from the plan's start_ctl value using
    the same approach as the scheduler — this ensures the curve starts at the
    user's declared fitness level rather than zero.

    Args:
        plan:      The TrainingPlan to evaluate.
        today_day: The current day (1-84). Days <= today_day are treated as
                   historical; days > today_day are projected. Defaults to 1
                   (i.e. all days projected) when the plan has just been created.

    Returns:
        A list of 84 DaySnapshot objects, one per day.
    """
    from engine.reoptimiser import STATUS_COMPLETED, STATUS_MISSED

    # Seed initial fitness/fatigue from the user's starting CTL. We use a
    # 7× multiplier consistent with the scheduler's weekly TSS calculation,
    # then derive an equivalent steady-state ATL. For a user at equilibrium,
    # ATL ≈ CTL × (τ_fitness / τ_fatigue).
    model = BanisterModel(
        initial_fitness=plan.start_ctl,
        initial_fatigue=plan.start_ctl * (7.0 / 42.0),
    )

    # Build a quick lookup: day_number -> Workout
    workout_map = {w.day: w for w in plan.workouts}

    snapshots: List[DaySnapshot] = []

    for day_num in range(1, 85):
        workout = workout_map.get(day_num)
        is_projected = day_num > today_day

        if workout is None:
            # Shouldn't happen for a valid 84-day plan, but be safe.
            tss = 0.0
        elif workout.phase == "Rest":
            tss = 0.0
        elif is_projected:
            # Future day — use the scheduled target TSS as the projection.
            tss = workout.target_tss
        else:
            # Historical day — use actual outcome.
            status = getattr(workout, "status", "planned")
            if status == STATUS_COMPLETED:
                tss = workout.target_tss
            elif status == STATUS_MISSED:
                tss = 0.0
            else:
                # 'planned' but in the past (edge case if today_day is mid-plan).
                tss = workout.target_tss

        model.add_daily_load(tss)

        snapshots.append(DaySnapshot(
            day=day_num,
            week=workout.week if workout else ((day_num - 1) // 7 + 1),
            ctl=round(model.fitness, 2),
            atl=round(model.fatigue, 2),
            tsb=round(model.get_readiness(), 2),
            tss_used=round(tss, 1),
            projected=is_projected,
        ))

    return snapshots


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Original class demo (unchanged) ──────────────────────────────────────
    athlete = BanisterModel(initial_fitness=20.0, initial_fatigue=10.0)
    print(f"Pre-workout State  -> Fitness: {athlete.fitness:.1f}, "
          f"Fatigue: {athlete.fatigue:.1f}, "
          f"Readiness: {athlete.get_readiness():.1f}")

    athlete.add_daily_load(150.0)
    print(f"Post-workout State -> Fitness: {athlete.fitness:.1f}, "
          f"Fatigue: {athlete.fatigue:.1f}, "
          f"Readiness: {athlete.get_readiness():.1f}")

    # ── Curve demo ────────────────────────────────────────────────────────────
    print("\n── Fitness curve for intermediate cyclist, Tue/Thu/Sat/Sun ──")
    from engine.scheduler import generate_plan
    from engine.reoptimiser import mark_missed, mark_completed

    plan = generate_plan("intermediate", [1, 3, 5, 6], recovery_weeks=True)

    # Simulate: user has completed days 1-14 (weeks 1-2), missed day 16.
    for d in range(1, 15):
        w = next((x for x in plan.workouts if x.day == d), None)
        if w and w.phase != "Rest":
            mark_completed(plan, d)
    mark_missed(plan, day=16)

    curve = compute_curve(plan, today_day=21)

    print(f"\n{'Day':>4} {'Week':>5} {'CTL':>7} {'ATL':>7} {'TSB':>7}  {'Type'}")
    print("-" * 45)
    for snap in curve:
        tag = "projected" if snap.projected else "actual   "
        # Print every 7th day plus the first for a readable summary.
        if snap.day == 1 or snap.day % 7 == 0:
            print(f"{snap.day:>4} {snap.week:>5} "
                  f"{snap.ctl:>7.1f} {snap.atl:>7.1f} {snap.tsb:>7.1f}  {tag}")

    final = curve[-1]
    print(f"\nDay 84 (race day): CTL={final.ctl:.1f}, "
          f"ATL={final.atl:.1f}, TSB={final.tsb:.1f}")