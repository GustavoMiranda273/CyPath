"""
CyPath Scheduler
Generates a 12-week periodised training plan structured into Base, Build, and
Taper phases, following established endurance training principles.

The scheduler produces a day-by-day prescribed Training Stress Score (TSS)
for each day of the 12-week plan, respecting:
  - The user's self-declared fitness level (beginner/intermediate/experienced)
  - The user's available training days per week
  - Hard safety caps on daily and weekly TSS

Author: Gustavo Miranda
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─── Configuration constants ─────────────────────────────────────────────────

# Starting Chronic Training Load (CTL) for each experience level.
# These values reflect typical fitness baselines from the sports-science
# literature and are used as the target weekly TSS at the END of the Base phase.
FITNESS_PROFILES: Dict[str, float] = {
    "beginner":     30.0,   # Untrained or lightly active
    "intermediate": 50.0,   # Rides regularly, some event experience
    "experienced":  70.0,   # Consistent training, familiar with endurance events
}

# Phase structure (12-week plan).
# Each tuple is (phase_name, start_week, end_week) — inclusive.
PHASES = [
    ("Base",  1, 6),
    ("Build", 7, 10),
    ("Taper", 11, 12),
]

# Safety caps — hard-coded physiological limits (see Section 4, NFR1).
MAX_DAILY_TSS:  float = 150.0
MAX_WEEKLY_TSS: float = 700.0

# Relative weekly TSS multipliers for each phase, expressed as a fraction of
# the user's target weekly TSS (derived from their starting CTL).
# Base ramps up, Build peaks, Taper drops sharply.
PHASE_LOAD_MULTIPLIERS: Dict[str, Dict[int, float]] = {
    "Base":  {1: 0.70, 2: 0.80, 3: 0.90, 4: 0.65, 5: 1.00, 6: 1.10},
    "Build": {7: 1.20, 8: 1.30, 9: 0.95, 10: 1.40},
    "Taper": {11: 0.75, 12: 0.50},
}
# Note: weeks 4 and 9 are recovery weeks (~30% drop from the previous week).

# Session weights — how weekly TSS is split across training days.
# The algorithm assigns weights in descending order of TSS:
#   heaviest session = 0.35 of weekly TSS (the long ride)
#   second heaviest  = 0.25 (interval / tempo session)
#   remaining days   = split the rest proportionally
SESSION_WEIGHTS: List[float] = [0.35, 0.25, 0.18, 0.12, 0.10]


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Workout:
    """Represents a single day in the plan."""
    day:         int    # 1-84 (day number across the whole 12-week plan)
    week:        int    # 1-12
    phase:       str    # "Base", "Build", "Taper", or "Rest"
    target_tss:  float  # Prescribed TSS for the day (0 = rest day)
    description: str    # Short label (e.g. "Long endurance ride")
    detail:      str = ""  # One-line instruction telling the user what to do


@dataclass
class TrainingPlan:
    """A full 12-week periodised plan."""
    profile:   str
    start_ctl: float
    workouts:  List[Workout] = field(default_factory=list)

    def weekly_tss(self, week_number: int) -> float:
        """Return the total prescribed TSS for a given week."""
        return sum(w.target_tss for w in self.workouts if w.week == week_number)

    def to_dict(self) -> Dict:
        """Convert to a dictionary for JSON serialisation (used by the API)."""
        return {
            "profile":   self.profile,
            "start_ctl": self.start_ctl,
            "workouts":  [w.__dict__ for w in self.workouts],
        }


# ─── Core scheduling logic ───────────────────────────────────────────────────

def _phase_for_week(week: int) -> str:
    """Return the phase name for a given week number (1-12)."""
    for name, start, end in PHASES:
        if start <= week <= end:
            return name
    raise ValueError(f"Week {week} is outside the 12-week plan")


# ─── Phase-aware session catalogue ──────────────────────────────────────────
# Each entry is (description, detail) — the detail tells the user exactly
# what kind of workout to do and how to approach it.
# Sessions are ordered heaviest-to-lightest (index 0 = longest/hardest).

SESSION_CATALOGUE: Dict[str, List[tuple]] = {
    "Base": [
        ("Long endurance ride",
         "Road cycling — long, steady effort at 65-75% max HR. "
         "Focus on smooth pedalling and building your aerobic base."),
        ("Interval session",
         "Road cycling — 6×4 min hard efforts at 85-90% max HR, "
         "3 min easy between each. Builds aerobic power."),
        ("Tempo ride",
         "Road cycling — sustained effort at 75-85% max HR. "
         "Comfortably hard — you can talk, but only in short sentences."),
        ("Recovery spin",
         "Easy cycling or indoor spin — very light effort, 50-60% max HR. "
         "Keeps the legs moving without adding fatigue."),
        ("Easy endurance ride",
         "Road cycling — relaxed conversational pace at 60-70% max HR. "
         "Good day to focus on technique and cadence."),
    ],
    "Build": [
        ("Long road ride",
         "Road cycling — long ride pushing the climbs at 70-80% max HR. "
         "Practice eating and drinking on the bike to simulate race day."),
        ("High-intensity intervals",
         "Road cycling — 5×5 min near-max efforts at 90-95% max HR. "
         "Full recovery between reps. Builds race-specific power."),
        ("Threshold ride",
         "Road cycling — sustained hard effort at 80-90% max HR. "
         "Hold threshold pace for 2×20 min blocks with 5 min rest between."),
        ("Active recovery",
         "Easy cycling, yoga, or light stretching — 50-60% max HR. "
         "Active recovery reduces soreness and keeps you moving safely."),
        ("Aerobic base ride",
         "Road cycling — steady aerobic pace at 65-75% max HR. "
         "Focus on building your engine for the long event ahead."),
    ],
    "Taper": [
        ("Short endurance ride",
         "Road cycling — shorter version of your long ride at 65-70% max HR. "
         "Keep legs fresh — this is not the time to push hard."),
        ("Light intervals",
         "Road cycling — 4×3 min moderate efforts at 80-85% max HR. "
         "Just enough intensity to stay sharp without adding fatigue."),
        ("Easy tempo ride",
         "Road cycling — light effort at 70-75% max HR. "
         "A confidence booster — your body is ready, trust the taper."),
        ("Gentle recovery spin",
         "Very easy cycling or yoga — 50% max HR. "
         "Flush the legs and stay relaxed ahead of your event."),
        ("Easy road ride",
         "Road cycling — relaxed conversational pace at 60-65% max HR. "
         "Enjoy the ride and arrive at race week feeling fresh."),
    ],
}


def _session_labels(num_sessions: int, phase: str = "Base") -> List[tuple]:
    """
    Return (description, detail) tuples for a week's training sessions,
    ordered heaviest-to-lightest. Phase-aware so workout instructions
    reflect the current training block (NFR2 in Section 4).

    Args:
        num_sessions: Number of training days this week.
        phase:        Current training phase ("Base", "Build", or "Taper").

    Returns:
        A list of (description, detail) tuples, length num_sessions.
    """
    catalogue = SESSION_CATALOGUE.get(phase, SESSION_CATALOGUE["Base"])
    return catalogue[:num_sessions]


def _distribute_weekly_tss(
    weekly_tss: float,
    training_days: List[int],
    phase: str = "Base",
) -> Dict[int, tuple]:
    """
    Split a week's total TSS across the user's available training days using
    the hard/easy pattern defined by SESSION_WEIGHTS.

    Args:
        weekly_tss:    Total TSS to distribute this week.
        training_days: List of weekday numbers (0=Mon ... 6=Sun) on which the
                       user can train.
        phase:         Current training phase — used to select appropriate
                       workout descriptions from SESSION_CATALOGUE.

    Returns:
        A dict mapping weekday -> (target_tss, description, detail).
    """
    num_sessions = len(training_days)
    if num_sessions == 0:
        return {}

    # Use only as many weights as there are sessions, then re-normalise.
    weights = SESSION_WEIGHTS[:num_sessions]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    session_info = _session_labels(num_sessions, phase)

    # Assign the heaviest session to the last available day (typically a
    # weekend long ride), then work backwards.
    ordered_days = sorted(training_days, reverse=True)

    distribution: Dict[int, tuple] = {}
    for day_index, (weight, (description, detail)) in enumerate(zip(weights, session_info)):
        tss = round(weekly_tss * weight, 1)
        # Enforce the daily safety cap (NFR1).
        tss = min(tss, MAX_DAILY_TSS)
        distribution[ordered_days[day_index]] = (tss, description, detail)

    return distribution


def generate_plan(
    profile: str,
    training_days: List[int],
    recovery_weeks: bool = True,
) -> TrainingPlan:
    """
    Generate a full 12-week training plan.

    Args:
        profile:        One of "beginner", "intermediate", or "experienced".
        training_days:  Weekdays the user can train on (0=Mon ... 6=Sun).
        recovery_weeks: If True, weeks 4 and 9 are de-loaded. If False, those
                        weeks follow a smoother progression instead.

    Returns:
        A TrainingPlan object containing 84 Workout entries (one per day).
    """

    # ── Validate inputs ──────────────────────────────────────────────────────
    if profile not in FITNESS_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Must be one of: {list(FITNESS_PROFILES.keys())}"
        )
    if not training_days or not all(0 <= d <= 6 for d in training_days):
        raise ValueError("training_days must be a non-empty list of 0-6 values")

    # ── Derive the target weekly TSS from the starting CTL ──────────────────
    # A common rule of thumb is that sustainable weekly TSS ≈ 7 × CTL.
    start_ctl = FITNESS_PROFILES[profile]
    base_weekly_tss = start_ctl * 7

    plan = TrainingPlan(profile=profile, start_ctl=start_ctl)

    # ── Build the plan week by week ──────────────────────────────────────────
    day_counter = 1
    for week in range(1, 13):
        phase = _phase_for_week(week)

        # Get the weekly multiplier for this phase/week.
        multiplier = PHASE_LOAD_MULTIPLIERS[phase][week]

        # If recovery weeks are disabled, replace them with a smoother value.
        if not recovery_weeks and week in (4, 9):
            # Interpolate between the neighbouring weeks.
            prev_mult = PHASE_LOAD_MULTIPLIERS[_phase_for_week(week - 1)][week - 1]
            next_mult = PHASE_LOAD_MULTIPLIERS[_phase_for_week(week + 1)][week + 1]
            multiplier = (prev_mult + next_mult) / 2

        weekly_tss = base_weekly_tss * multiplier
        # Enforce the weekly safety cap (NFR1).
        weekly_tss = min(weekly_tss, MAX_WEEKLY_TSS)

        distribution = _distribute_weekly_tss(weekly_tss, training_days, phase)

        # ── Build the 7 days of this week ───────────────────────────────────
        for weekday in range(7):
            if weekday in distribution:
                tss, description, detail = distribution[weekday]
                plan.workouts.append(Workout(
                    day=day_counter, week=week, phase=phase,
                    target_tss=tss, description=description, detail=detail,
                ))
            else:
                plan.workouts.append(Workout(
                    day=day_counter, week=week, phase="Rest",
                    target_tss=0.0, description="Rest day",
                    detail="No training today — rest and recovery are essential for adaptation.",
                ))
            day_counter += 1

    return plan


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: an intermediate cyclist who can train Tue/Thu/Sat/Sun.
    example = generate_plan(
        profile="intermediate",
        training_days=[1, 3, 5, 6],  # Tue, Thu, Sat, Sun
        recovery_weeks=True,
    )

    print(f"Plan for a {example.profile} cyclist (starting CTL {example.start_ctl})")
    print("=" * 64)
    for week in range(1, 13):
        total = example.weekly_tss(week)
        phase = _phase_for_week(week)
        print(f"Week {week:>2} ({phase:<5})  |  Weekly TSS: {total:>6.1f}")

    print("\nSample of first week's daily prescriptions:")
    for w in example.workouts[:7]:
        print(f"  Day {w.day:>2}  W{w.week} {w.phase:<5}  "
              f"TSS={w.target_tss:>5.1f}  {w.description}")