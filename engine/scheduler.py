

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
    """A full N-week periodised plan for a user-defined distance goal."""
    profile:     str
    start_ctl:   float
    goal_km:     float = 100.0   # Target event distance in kilometres
    total_weeks: int   = 12      # Total plan duration in weeks
    workouts:    List[Workout] = field(default_factory=list)

    def weekly_tss(self, week_number: int) -> float:
        """Return the total prescribed TSS for a given week."""
        return sum(w.target_tss for w in self.workouts if w.week == week_number)

    def total_days(self) -> int:
        """Total number of days in the plan."""
        return self.total_weeks * 7

    def to_dict(self) -> Dict:
        """Convert to a dictionary for JSON serialisation (used by the API)."""
        return {
            "profile":     self.profile,
            "start_ctl":   self.start_ctl,
            "goal_km":     self.goal_km,
            "total_weeks": self.total_weeks,
            "workouts":    [w.__dict__ for w in self.workouts],
        }


# ─── Core scheduling logic ───────────────────────────────────────────────────

def _build_phase_structure(total_weeks: int) -> List[tuple]:
 
    base_end  = max(1, round(total_weeks * 0.50))
    build_end = max(base_end + 1, round(total_weeks * 0.83))
    taper_end = total_weeks
    return [
        ("Base",  1,            base_end),
        ("Build", base_end + 1, build_end),
        ("Taper", build_end + 1, taper_end),
    ]


def _phase_for_week(week: int, phases: List[tuple] = None) -> str:
    """Return the phase name for a given week number."""
    if phases is None:
        phases = PHASES   # Fall back to the static 12-week default.
    for name, start, end in phases:
        if start <= week <= end:
            return name
    raise ValueError(f"Week {week} is outside the plan")


def _build_multipliers(phases: List[tuple], recovery_weeks: bool) -> Dict[int, float]:
   
    multipliers: Dict[int, float] = {}

    for phase_name, start, end in phases:
        n = end - start + 1  # Number of weeks in this phase.
        for i, week in enumerate(range(start, end + 1)):
            position = i / max(n - 1, 1)  # 0.0 … 1.0 within the phase.

            if phase_name == "Base":
                # Ramp from 0.70 to 1.10 with optional recovery dips.
                base_mult = 0.70 + position * 0.40
                # Every 4th week in Base is a recovery week.
                if recovery_weeks and (i + 1) % 4 == 0 and i < n - 1:
                    base_mult *= 0.65
            elif phase_name == "Build":
                # Ramp from 1.20 to 1.40, mid-phase dip if long enough.
                base_mult = 1.20 + position * 0.20
                if recovery_weeks and n >= 3 and i == n // 2:
                    base_mult *= 0.72
            else:  # Taper
                # Step down from 0.75 to 0.50.
                base_mult = 0.75 - position * 0.25

            multipliers[week] = round(base_mult, 3)

    return multipliers


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


# ─── Goal presets ────────────────────────────────────────────────────────────
# Maps a distance goal to a recommended number of training weeks and a TSS
# scale factor relative to the 100 km baseline.
GOAL_PRESETS: Dict[float, tuple] = {
    50.0:  (6,  0.60),   # 50 km in 6 weeks  — 60% of peak TSS
    75.0:  (9,  0.80),   # 75 km in 9 weeks  — 80% of peak TSS
    100.0: (12, 1.00),   # 100 km in 12 weeks — full TSS (baseline)
}


def generate_plan(
    profile: str,
    training_days: List[int],
    recovery_weeks: bool = True,
    goal_km: float = 100.0,
    total_weeks: int = 12,
) -> TrainingPlan:
    """
    Generate a periodised training plan for a user-defined distance goal.

    Args:
        profile:        One of "beginner", "intermediate", or "experienced".
        training_days:  Weekdays the user can train on (0=Mon ... 6=Sun).
        recovery_weeks: If True, de-load weeks are inserted periodically.
        goal_km:        Target event distance in kilometres.
        total_weeks:    Total plan duration in weeks.

    Returns:
        A TrainingPlan containing (total_weeks × 7) Workout entries.
    """

    # ── Validate inputs ──────────────────────────────────────────────────────
    if profile not in FITNESS_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Must be one of: {list(FITNESS_PROFILES.keys())}"
        )
    if not training_days or not all(0 <= d <= 6 for d in training_days):
        raise ValueError("training_days must be a non-empty list of 0-6 values")
    if total_weeks < 4:
        raise ValueError("total_weeks must be at least 4")

    # ── Derive target weekly TSS ─────────────────────────────────────────────
    # Sustainable weekly TSS ≈ 7 × CTL (standard coaching rule of thumb).
    # Scale by the goal's TSS factor so shorter events don't over-stress.
    start_ctl = FITNESS_PROFILES[profile]
    # Find the closest preset to get the TSS scale factor.
    closest_preset = min(GOAL_PRESETS.keys(), key=lambda k: abs(k - goal_km))
    _, tss_scale = GOAL_PRESETS[closest_preset]
    base_weekly_tss = start_ctl * 7 * tss_scale

    # ── Build dynamic phase structure and multipliers ────────────────────────
    phases      = _build_phase_structure(total_weeks)
    multipliers = _build_multipliers(phases, recovery_weeks)

    plan = TrainingPlan(
        profile=profile,
        start_ctl=start_ctl,
        goal_km=goal_km,
        total_weeks=total_weeks,
    )

    # ── Build the plan week by week ──────────────────────────────────────────
    day_counter = 1
    for week in range(1, total_weeks + 1):
        phase      = _phase_for_week(week, phases)
        multiplier = multipliers[week]
        weekly_tss = min(base_weekly_tss * multiplier, MAX_WEEKLY_TSS)

        distribution = _distribute_weekly_tss(weekly_tss, training_days, phase)

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