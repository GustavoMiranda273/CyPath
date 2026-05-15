"""
CyPath Algorithmic Validation Test Suite
=========================================
Tests the three engine modules:
  - BanisterModel  (banister_model.py)
  - generate_plan  (scheduler.py)
  - mark_missed /  (reoptimiser.py)
    mark_completed /
    restore_planned

Run from /home/claude/cypath_tests/:
    pytest test_cypath.py -v
"""

import math
import pytest

from engine.scheduler import (
    generate_plan,
    TrainingPlan,
    Workout,
    FITNESS_PROFILES,
    MAX_DAILY_TSS,
    MAX_WEEKLY_TSS,
)
from engine.banister_model import BanisterModel
from engine.reoptimiser import (
    mark_missed,
    mark_completed,
    restore_planned,
    STATUS_PLANNED,
    STATUS_COMPLETED,
    STATUS_MISSED,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _plan(profile="intermediate", days=None, recovery=True,
          goal_km=100.0, total_weeks=12):
    """Generate a plan with sensible defaults for test use."""
    return generate_plan(
        profile=profile,
        training_days=days or [1, 3, 5, 6],
        recovery_weeks=recovery,
        goal_km=goal_km,
        total_weeks=total_weeks,
    )


def _first_training_day_of_week(plan: TrainingPlan, week: int) -> Workout:
    """Return the first non-rest workout in a given week."""
    return next(
        w for w in plan.workouts
        if w.week == week and w.phase != "Rest"
    )


def _last_training_day_of_week(plan: TrainingPlan, week: int) -> Workout:
    """Return the last non-rest workout in a given week."""
    return max(
        (w for w in plan.workouts if w.week == week and w.phase != "Rest"),
        key=lambda w: w.day,
    )


def _training_days_in_week(plan: TrainingPlan, week: int):
    return [w for w in plan.workouts if w.week == week and w.phase != "Rest"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BANISTER IMPULSE-RESPONSE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestBanisterModel:
    """
    Validates the BanisterModel class against hand-calculated reference values.

    The model uses the standard exponential decay formula:
        CTL_new = CTL_old × e^(−1/τ_CTL) + TSS × (1 − e^(−1/τ_CTL))  τ = 42
        ATL_new = ATL_old × e^(−1/τ_ATL) + TSS × (1 − e^(−1/τ_ATL))  τ =  7
        TSB     = CTL − ATL
    """

    # Pre-computed constants for assertions
    K_F = math.exp(-1 / 42.0)   # ≈ 0.97647
    K_A = math.exp(-1 / 7.0)    # ≈ 0.86688

    def test_initial_state_zero(self):
        """A fresh model starts at CTL=0 and ATL=0."""
        m = BanisterModel()
        assert m.fitness == 0.0
        assert m.fatigue == 0.0

    def test_initial_state_custom(self):
        """Custom starting values are stored correctly."""
        m = BanisterModel(initial_fitness=50.0, initial_fatigue=10.0)
        assert m.fitness == 50.0
        assert m.fatigue == 10.0

    def test_readiness_is_ctl_minus_atl(self):
        """TSB (readiness) always equals CTL minus ATL."""
        m = BanisterModel(initial_fitness=40.0, initial_fatigue=25.0)
        assert m.get_readiness() == pytest.approx(15.0)

    def test_readiness_can_be_negative(self):
        """TSB is negative when fatigue exceeds fitness — heavy load state."""
        m = BanisterModel(initial_fitness=10.0, initial_fatigue=30.0)
        assert m.get_readiness() == pytest.approx(-20.0)

    def test_zero_load_causes_decay(self):
        """With zero TSS, CTL and ATL both decay toward zero."""
        m = BanisterModel(initial_fitness=50.0, initial_fatigue=20.0)
        m.add_daily_load(0.0)
        assert m.fitness == pytest.approx(50.0 * self.K_F, rel=1e-6)
        assert m.fatigue == pytest.approx(20.0 * self.K_A, rel=1e-6)

    def test_single_load_ctl_value(self):
        """
        After one day of 100 TSS from zero, CTL matches the formula exactly.
        Expected: CTL = 100 × (1 − e^(−1/42)) ≈ 2.353
        """
        m = BanisterModel()
        m.add_daily_load(100.0)
        expected_ctl = 100.0 * (1 - self.K_F)
        assert m.fitness == pytest.approx(expected_ctl, rel=1e-6)

    def test_single_load_atl_value(self):
        """
        After one day of 100 TSS from zero, ATL matches the formula exactly.
        Expected: ATL = 100 × (1 − e^(−1/7)) ≈ 13.312
        """
        m = BanisterModel()
        m.add_daily_load(100.0)
        expected_atl = 100.0 * (1 - self.K_A)
        assert m.fatigue == pytest.approx(expected_atl, rel=1e-6)

    def test_atl_rises_faster_than_ctl(self):
        """
        ATL (τ=7) responds faster to a new load than CTL (τ=42).
        After one day of heavy training, ATL should be larger than CTL.
        """
        m = BanisterModel()
        m.add_daily_load(100.0)
        assert m.fatigue > m.fitness

    def test_readiness_negative_after_first_hard_day(self):
        """After a single hard day from a rested state, TSB is negative."""
        m = BanisterModel()
        m.add_daily_load(100.0)
        assert m.get_readiness() < 0.0

    def test_atl_converges_to_steady_state_faster_than_ctl(self):
        """
        With consistent daily load, both CTL and ATL converge to the same
        steady-state value (the daily load itself), but ATL (τ=7) gets there
        faster than CTL (τ=42). After 90 days, ATL is closer to 80 than CTL.
        """
        m = BanisterModel()
        for _ in range(90):
            m.add_daily_load(80.0)
        steady_state = 80.0
        atl_gap = abs(m.fatigue - steady_state)
        ctl_gap = abs(m.fitness - steady_state)
        assert atl_gap < ctl_gap, (
            f"ATL gap={atl_gap:.4f} should be smaller than CTL gap={ctl_gap:.4f}"
        )

    def test_two_days_cumulative(self):
        """Two consecutive 100 TSS days produce the correct cumulative CTL."""
        m = BanisterModel()
        m.add_daily_load(100.0)
        ctl_day1 = 100.0 * (1 - self.K_F)
        m.add_daily_load(100.0)
        expected_ctl2 = ctl_day1 * self.K_F + 100.0 * (1 - self.K_F)
        assert m.fitness == pytest.approx(expected_ctl2, rel=1e-6)

    def test_decay_after_rest_reduces_both(self):
        """A rest day (0 TSS) after training reduces both CTL and ATL."""
        m = BanisterModel()
        m.add_daily_load(100.0)
        m.add_daily_load(100.0)
        ctl_before = m.fitness
        atl_before = m.fatigue
        m.add_daily_load(0.0)
        assert m.fitness < ctl_before
        assert m.fatigue < atl_before

    def test_ctl_decays_slower_than_atl(self):
        """
        CTL (τ=42) decays more slowly than ATL (τ=7).
        After a period of rest, ATL should drop by a greater fraction.
        """
        m = BanisterModel(initial_fitness=50.0, initial_fatigue=50.0)
        ctl_before, atl_before = m.fitness, m.fatigue
        for _ in range(7):
            m.add_daily_load(0.0)
        ctl_drop_frac = (ctl_before - m.fitness) / ctl_before
        atl_drop_frac = (atl_before - m.fatigue) / atl_before
        assert atl_drop_frac > ctl_drop_frac

    def test_time_constants_are_correct(self):
        """Model stores the standard physiological time constants."""
        m = BanisterModel()
        assert m.tau_fitness == 42.0
        assert m.tau_fatigue == 7.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PLAN GENERATOR (scheduler.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduler:
    """Validates that generate_plan produces a correctly structured plan."""

    def test_plan_has_correct_total_days(self):
        """A 12-week plan has exactly 84 workouts (12 × 7)."""
        plan = _plan()
        assert len(plan.workouts) == 84

    def test_plan_has_correct_week_count(self):
        """Every week from 1 to 12 is present in the plan."""
        plan = _plan()
        weeks = {w.week for w in plan.workouts}
        assert weeks == set(range(1, 13))

    def test_total_days_method(self):
        """TrainingPlan.total_days() returns total_weeks × 7."""
        plan = _plan(total_weeks=6)
        assert plan.total_days() == 42

    def test_correct_start_ctl_for_intermediate(self):
        """Intermediate profile starts with CTL of 50."""
        plan = _plan(profile="intermediate")
        assert plan.start_ctl == FITNESS_PROFILES["intermediate"]
        assert plan.start_ctl == 50.0

    def test_correct_start_ctl_for_beginner(self):
        """Beginner profile starts with CTL of 30."""
        plan = _plan(profile="beginner")
        assert plan.start_ctl == 30.0

    def test_correct_start_ctl_for_experienced(self):
        """Experienced profile starts with CTL of 70."""
        plan = _plan(profile="experienced")
        assert plan.start_ctl == 70.0

    def test_training_sessions_only_on_specified_days(self):
        """Workouts with TSS > 0 only appear on the selected training days."""
        days = [1, 3, 5, 6]  # Tue, Thu, Sat, Sun
        plan = _plan(days=days)
        for w in plan.workouts:
            weekday = (w.day - 1) % 7
            if w.phase != "Rest":
                assert weekday in days, (
                    f"Training session on day {w.day} (weekday {weekday}) "
                    f"but {weekday} not in selected days {days}"
                )

    def test_rest_days_have_zero_tss(self):
        """All rest-phase workouts have a target TSS of 0."""
        plan = _plan()
        for w in plan.workouts:
            if w.phase == "Rest":
                assert w.target_tss == 0.0

    def test_no_session_exceeds_daily_cap(self):
        """No individual session exceeds MAX_DAILY_TSS (150)."""
        plan = _plan(profile="experienced")
        for w in plan.workouts:
            assert w.target_tss <= MAX_DAILY_TSS, (
                f"Day {w.day} TSS={w.target_tss} exceeds daily cap {MAX_DAILY_TSS}"
            )

    def test_no_week_exceeds_weekly_cap(self):
        """No week's total TSS exceeds MAX_WEEKLY_TSS (700)."""
        plan = _plan(profile="experienced")
        for week in range(1, 13):
            total = plan.weekly_tss(week)
            assert total <= MAX_WEEKLY_TSS + 0.1, (
                f"Week {week} total TSS={total:.1f} exceeds weekly cap {MAX_WEEKLY_TSS}"
            )

    def test_phase_structure_base_early(self):
        """Weeks 1–6 are assigned to the Base phase in a 12-week plan."""
        plan = _plan()
        for w in plan.workouts:
            if w.week <= 6 and w.phase != "Rest":
                assert w.phase == "Base"

    def test_phase_structure_build_mid(self):
        """Weeks 7–10 are assigned to the Build phase in a 12-week plan."""
        plan = _plan()
        for w in plan.workouts:
            if 7 <= w.week <= 10 and w.phase != "Rest":
                assert w.phase == "Build"

    def test_phase_structure_taper_late(self):
        """Weeks 11–12 are assigned to the Taper phase in a 12-week plan."""
        plan = _plan()
        for w in plan.workouts:
            if w.week >= 11 and w.phase != "Rest":
                assert w.phase == "Taper"

    def test_build_load_exceeds_base_load(self):
        """Average weekly TSS in the Build phase exceeds that of the Base phase."""
        plan = _plan()
        base_avg  = sum(plan.weekly_tss(w) for w in range(1, 7))  / 6
        build_avg = sum(plan.weekly_tss(w) for w in range(7, 11)) / 4
        assert build_avg > base_avg

    def test_taper_load_below_build_peak(self):
        """Peak Taper weekly TSS is below peak Build weekly TSS."""
        plan = _plan()
        build_peak = max(plan.weekly_tss(w) for w in range(7, 11))
        taper_peak = max(plan.weekly_tss(w) for w in range(11, 13))
        assert taper_peak < build_peak

    def test_progressive_overload_base_phase(self):
        """Week 5 has higher TSS than Week 1 — Base phase ramps up."""
        plan = _plan(recovery=False)
        assert plan.weekly_tss(5) > plan.weekly_tss(1)

    def test_recovery_week_lower_than_adjacent(self):
        """With recovery_weeks=True, week 4 is lower than weeks 3 and 5."""
        plan = _plan(recovery=True)
        assert plan.weekly_tss(4) < plan.weekly_tss(3)
        assert plan.weekly_tss(4) < plan.weekly_tss(5)

    def test_six_week_plan_length(self):
        """A 6-week (50 km) plan has exactly 42 workouts."""
        plan = _plan(total_weeks=6)
        assert len(plan.workouts) == 42

    def test_nine_week_plan_length(self):
        """A 9-week (75 km) plan has exactly 63 workouts."""
        plan = _plan(total_weeks=9)
        assert len(plan.workouts) == 63

    def test_invalid_profile_raises(self):
        """An unknown fitness profile raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            generate_plan("elite", [1, 3, 5])

    def test_empty_training_days_raises(self):
        """An empty training days list raises a ValueError."""
        with pytest.raises(ValueError):
            generate_plan("intermediate", [])

    def test_too_short_plan_raises(self):
        """A plan shorter than 4 weeks raises a ValueError."""
        with pytest.raises(ValueError, match="at least 4"):
            generate_plan("intermediate", [1, 3], total_weeks=3)

    def test_single_training_day_works(self):
        """A plan with only one training day per week is valid and generates correctly."""
        plan = _plan(days=[6])  # Sunday only
        training = [w for w in plan.workouts if w.phase != "Rest"]
        assert len(training) == 12  # one per week

    def test_all_workouts_have_descriptions(self):
        """Every training session has a non-empty description."""
        plan = _plan()
        for w in plan.workouts:
            if w.phase != "Rest":
                assert w.description, f"Day {w.day} has no description"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RE-OPTIMISATION ENGINE (reoptimiser.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestReoptimiser:
    """
    Scenario-based tests for the re-optimisation engine.
    Each test creates a controlled plan state and asserts the correct outcome.
    """

    # ── Status management ─────────────────────────────────────────────────────

    def test_mark_completed_sets_status(self):
        """mark_completed sets the workout status to STATUS_COMPLETED."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 1)
        mark_completed(plan, w.day)
        assert w.status == STATUS_COMPLETED

    def test_restore_planned_resets_status(self):
        """restore_planned resets a completed workout back to STATUS_PLANNED."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 1)
        mark_completed(plan, w.day)
        restore_planned(plan, w.day)
        assert w.status == STATUS_PLANNED

    def test_restore_from_missed_resets_status(self):
        """restore_planned also resets a missed workout to STATUS_PLANNED."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 1)
        mark_missed(plan, w.day)
        restore_planned(plan, w.day)
        assert w.status == STATUS_PLANNED

    # ── Redistribution: normal case ───────────────────────────────────────────

    def test_mark_missed_sets_status(self):
        """A missed session is flagged with STATUS_MISSED."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 2)
        mark_missed(plan, w.day)
        assert w.status == STATUS_MISSED

    def test_mark_missed_zeroes_tss(self):
        """A missed session has its target TSS set to 0."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 2)
        mark_missed(plan, w.day)
        assert w.target_tss == 0.0

    def test_full_redistribution_no_drop(self):
        """
        Missing the first session of a week with multiple sessions remaining
        should redistribute all missed TSS with nothing dropped.
        """
        plan = _plan()
        w = _first_training_day_of_week(plan, 2)
        result = mark_missed(plan, w.day)
        assert result.dropped_tss == 0.0
        assert result.redistributed_tss == pytest.approx(result.missed_tss, rel=1e-3)

    def test_redistribution_affects_later_days(self):
        """Redistribution increases TSS on at least one subsequent training day."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 2)
        result = mark_missed(plan, w.day)
        assert len(result.affected_days) > 0

    def test_missed_tss_equals_redistributed_plus_dropped(self):
        """
        Conservation law: missed_tss == redistributed_tss + dropped_tss
        within floating-point tolerance.
        """
        plan = _plan()
        w = _first_training_day_of_week(plan, 3)
        result = mark_missed(plan, w.day)
        total = round(result.redistributed_tss + result.dropped_tss, 1)
        assert total == pytest.approx(result.missed_tss, abs=0.2)

    def test_no_session_exceeds_cap_after_redistribution(self):
        """After redistribution, no session in the affected week exceeds MAX_DAILY_TSS."""
        plan = _plan(profile="experienced")
        w = _first_training_day_of_week(plan, 8)  # Peak Build week
        mark_missed(plan, w.day)
        week_num = w.week
        for workout in plan.workouts:
            if workout.week == week_num:
                assert workout.target_tss <= MAX_DAILY_TSS + 0.1, (
                    f"Day {workout.day} TSS={workout.target_tss} exceeds cap after redistribution"
                )

    # ── Redistribution: last day of week ─────────────────────────────────────

    def test_last_day_of_week_nothing_to_redistribute(self):
        """
        Missing the last training session of a week leaves no eligible days.
        All missed TSS is dropped and the result is unsuccessful.
        """
        plan = _plan()
        w = _last_training_day_of_week(plan, 3)
        result = mark_missed(plan, w.day)
        assert result.redistributed_tss == 0.0
        assert result.dropped_tss == pytest.approx(result.missed_tss, rel=1e-3)
        assert result.success is False
        assert result.affected_days == []

    def test_last_day_warning_is_issued(self):
        """A warning message is produced when TSS cannot be redistributed."""
        plan = _plan()
        w = _last_training_day_of_week(plan, 3)
        result = mark_missed(plan, w.day)
        assert result.warning is not None
        assert len(result.warning) > 0

    # ── Redistribution: daily cap enforcement ────────────────────────────────

    def test_cap_respected_when_remaining_days_are_near_limit(self):
        """
        When remaining days are already close to MAX_DAILY_TSS, redistribution
        must not push them over the cap, even if that means dropping TSS.
        """
        plan = _plan(profile="intermediate")
        week = 5  # High-load Base week
        training = _training_days_in_week(plan, week)

        # Artificially inflate all but the last training day to just below cap
        for i, w in enumerate(training[:-1]):
            w.target_tss = MAX_DAILY_TSS - 1.0  # 149 TSS — almost at cap

        # Miss the last training day (which is actually the first in the week
        # because mark_missed looks at days AFTER the missed day)
        first = training[0]
        first.target_tss = 60.0  # give it a real value to miss
        result = mark_missed(plan, first.day)

        # After redistribution, nothing should exceed the cap
        for w in plan.workouts:
            if w.week == week:
                assert w.target_tss <= MAX_DAILY_TSS + 0.1

    # ── Error cases ───────────────────────────────────────────────────────────

    def test_cannot_miss_rest_day(self):
        """Attempting to miss a rest day raises a ValueError."""
        plan = _plan()
        rest = next(w for w in plan.workouts if w.phase == "Rest")
        with pytest.raises(ValueError, match="rest day"):
            mark_missed(plan, rest.day)

    def test_cannot_miss_already_missed_session(self):
        """Attempting to miss a session that is already missed raises a ValueError."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 1)
        mark_missed(plan, w.day)
        with pytest.raises(ValueError, match="not planned"):
            mark_missed(plan, w.day)

    def test_cannot_miss_completed_session(self):
        """Attempting to miss a session that is already completed raises a ValueError."""
        plan = _plan()
        w = _first_training_day_of_week(plan, 1)
        mark_completed(plan, w.day)
        with pytest.raises(ValueError, match="not planned"):
            mark_missed(plan, w.day)

    def test_cannot_miss_nonexistent_day(self):
        """A day number outside the plan raises a ValueError."""
        plan = _plan()
        with pytest.raises(ValueError, match="No workout found"):
            mark_missed(plan, 999)

    def test_cannot_complete_nonexistent_day(self):
        """mark_completed on a non-existent day raises a ValueError."""
        plan = _plan()
        with pytest.raises(ValueError, match="No workout found"):
            mark_completed(plan, 999)

    def test_cannot_restore_nonexistent_day(self):
        """restore_planned on a non-existent day raises a ValueError."""
        plan = _plan()
        with pytest.raises(ValueError, match="No workout found"):
            restore_planned(plan, 999)

    # ── Affected days are only within the same week ───────────────────────────

    def test_redistribution_stays_within_same_week(self):
        """
        Redistribution must not affect workouts outside the missed session's week.
        TSS values in all other weeks must remain unchanged.
        """
        plan = _plan()
        week = 4
        w = _first_training_day_of_week(plan, week)

        # Snapshot TSS for all days outside the missed week
        other_week_snapshot = {
            wo.day: wo.target_tss
            for wo in plan.workouts
            if wo.week != week
        }

        mark_missed(plan, w.day)

        for wo in plan.workouts:
            if wo.week != week:
                assert wo.target_tss == other_week_snapshot[wo.day], (
                    f"Day {wo.day} (week {wo.week}) TSS changed unexpectedly"
                )


    # ── Restore: original TSS and redistribution reversal ────────────────────

    def test_restore_after_miss_recovers_original_tss(self):
        """
        Restoring a missed session puts the original target TSS back exactly.
        The day should not remain at 0 after restore.
        """
        plan = _plan()
        w = _first_training_day_of_week(plan, 2)
        original_tss = w.target_tss
        mark_missed(plan, w.day)
        assert w.target_tss == 0.0          # zeroed by mark_missed
        restore_planned(plan, w.day)
        assert w.target_tss == pytest.approx(original_tss, rel=1e-6)

    def test_restore_reverses_redistribution_on_planned_days(self):
        """
        When a session is restored, any TSS that was redistributed to
        still-planned days must be subtracted back from those days.
        After restore the week's total TSS should equal the original total.
        """
        plan = _plan()
        week = 3
        original_total = plan.weekly_tss(week)

        w = _first_training_day_of_week(plan, week)
        mark_missed(plan, w.day)
        restore_planned(plan, w.day)

        restored_total = plan.weekly_tss(week)
        assert restored_total == pytest.approx(original_total, abs=0.5)

    def test_restore_leaves_completed_redistributed_days_alone(self):
        """
        If a day that received redistributed TSS has since been marked
        completed, restore must not alter its TSS — it reflects actual
        ride data and should be preserved.
        """
        plan = _plan()
        week = 2

        # Miss the first training day — redistribution goes to later days
        missed_w = _first_training_day_of_week(plan, week)
        result = mark_missed(plan, missed_w.day)

        # Mark one of the affected (redistributed) days as completed
        if result.affected_days:
            completed_day_num = result.affected_days[0]
            completed_w = next(w for w in plan.workouts if w.day == completed_day_num)
            tss_after_redistribution = completed_w.target_tss
            mark_completed(plan, completed_day_num)

            # Now restore the missed session
            restore_planned(plan, missed_w.day)

            # The completed day's TSS must be unchanged
            assert completed_w.target_tss == pytest.approx(
                tss_after_redistribution, rel=1e-6
            ), "Completed day TSS was altered by restore — should be left alone"
    def test_redistribution_only_affects_days_after_missed(self):
        """
        Only training days that come AFTER the missed day in the same week
        should have their TSS changed.
        """
        plan = _plan()
        week = 2
        w = _first_training_day_of_week(plan, week)

        # Snapshot TSS for all days before or on the missed day
        earlier_snapshot = {
            wo.day: wo.target_tss
            for wo in plan.workouts
            if wo.week == week and wo.day <= w.day
        }

        mark_missed(plan, w.day)

        for wo in plan.workouts:
            if wo.week == week and wo.day < w.day:
                assert wo.target_tss == earlier_snapshot[wo.day], (
                    f"Day {wo.day} (before missed day) TSS changed unexpectedly"
                )