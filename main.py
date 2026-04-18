"""
CyPath API Gateway
Flask application that exposes the CyPath algorithmic engine to the frontend.

Routes:
    GET  /                   -> Onboarding form (profile + training days)
    POST /generate-plan      -> Build a 12-week plan and redirect to dashboard
    GET  /dashboard          -> Main dashboard view of the current plan
    POST /complete-session   -> Mark a day's workout as completed
    POST /miss-session       -> Mark a day as missed; triggers re-optimisation
    POST /restore-session    -> Undo a missed/completed flag (revert to planned)
    GET  /api/plan           -> Return the current plan as JSON

State management:
    For this university prototype, plans are stored in a simple in-memory
    dictionary keyed by Flask session ID. This is appropriate for the single-
    user evaluation scope defined in the ethics approval (ID 70224).
    Production deployments would substitute a proper database; this is noted
    as future work in Section 7.

Author: Gustavo Miranda
"""

import uuid
from typing import Optional

from flask import (
    Flask, render_template, request, redirect, url_for, session, jsonify, flash
)

from engine.scheduler import generate_plan, TrainingPlan
from engine.banister_model import compute_curve
from engine.reoptimiser import (
    mark_missed, mark_completed, restore_planned, ReoptimisationResult,
    STATUS_COMPLETED, STATUS_MISSED, STATUS_PLANNED,
)


app = Flask(__name__)
# Secret key enables Flask's signed session cookies. For production this
# would be loaded from an environment variable.
app.secret_key = "cypath-dev-secret-change-in-production"


# ─── In-memory store: { session_id: TrainingPlan } ──────────────────────────
# See module docstring — this is intentional for a prototype. See Section 7
# for planned future migration to persistent storage.
PLANS: dict = {}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _session_id() -> str:
    """Return the current user's session id, creating one if necessary."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def _current_plan() -> Optional[TrainingPlan]:
    """Return the plan for the current session, or None if no plan exists yet."""
    return PLANS.get(_session_id())


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Onboarding view. If the user already has a plan, send them to the
    dashboard; otherwise show the plan-creation form.
    """
    if _current_plan() is not None:
        return redirect(url_for("dashboard"))
    return render_template("onboarding.html")


@app.route("/generate-plan", methods=["POST"])
def create_plan():
    """
    Handle the onboarding form submission, generate a 12-week plan, and
    redirect to the dashboard.
    """
    profile = request.form.get("profile", "intermediate")

    # Training days arrive as a list of strings (e.g. ['1', '3', '5', '6']).
    # Convert to ints for the scheduler.
    training_days = [int(d) for d in request.form.getlist("training_days")]

    if not training_days:
        flash("Please select at least one training day.")
        return redirect(url_for("index"))

    recovery_weeks = request.form.get("recovery_weeks") == "on"

    try:
        plan = generate_plan(
            profile=profile,
            training_days=training_days,
            recovery_weeks=recovery_weeks,
        )
    except ValueError as e:
        flash(f"Could not generate plan: {e}")
        return redirect(url_for("index"))

    sid = _session_id()
    PLANS[sid] = plan
    # Track which day the user is currently on (starts at day 1).
    session["today_day"] = 1
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Main dashboard view. Shows the fitness curve and the week-by-week plan,
    allowing the user to mark sessions as completed or missed.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    today_day = session.get("today_day", 1)

    # ── Compute the Banister fitness curve ───────────────────────────────────
    curve = compute_curve(plan, today_day=today_day)

    # Prepare data for Chart.js — three series across 84 days.
    chart_labels  = [f"W{s.week}D{s.day}" for s in curve]
    chart_ctl     = [s.ctl for s in curve]
    chart_atl     = [s.atl for s in curve]
    chart_tsb     = [s.tsb for s in curve]
    # Index where projected data begins (used to draw a vertical marker).
    split_index   = today_day - 1

    # ── Group workouts by week for the plan table ────────────────────────────
    weeks = []
    for week_number in range(1, 13):
        weeks.append({
            "number":    week_number,
            "phase":     next(
                (w.phase for w in plan.workouts
                 if w.week == week_number and w.phase != "Rest"),
                "Rest",
            ),
            "total_tss": round(plan.weekly_tss(week_number), 1),
            "workouts":  [w for w in plan.workouts if w.week == week_number],
        })

    # Current day summary for the header card.
    today_workout = next((w for w in plan.workouts if w.day == today_day), None)
    today_snap    = curve[today_day - 1] if curve else None

    return render_template(
        "dashboard.html",
        plan=plan,
        weeks=weeks,
        today_day=today_day,
        today_workout=today_workout,
        today_snap=today_snap,
        chart_labels=chart_labels,
        chart_ctl=chart_ctl,
        chart_atl=chart_atl,
        chart_tsb=chart_tsb,
        split_index=split_index,
        STATUS_COMPLETED=STATUS_COMPLETED,
        STATUS_MISSED=STATUS_MISSED,
        STATUS_PLANNED=STATUS_PLANNED,
    )


@app.route("/complete-session", methods=["POST"])
def complete_session():
    """Mark a day as completed and advance today_day if it was today."""
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    day = int(request.form["day"])
    try:
        mark_completed(plan, day)
        flash(f"Day {day} marked as completed. 💪")
        if session.get("today_day", 1) == day:
            session["today_day"] = min(day + 1, 84)
    except ValueError as e:
        flash(str(e))

    return redirect(url_for("dashboard"))


@app.route("/miss-session", methods=["POST"])
def miss_session():
    """
    Mark a day as missed, run the re-optimisation engine, and advance today_day.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    day = int(request.form["day"])
    try:
        result: ReoptimisationResult = mark_missed(plan, day)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("dashboard"))

    summary = (
        f"Day {day} marked as missed. "
        f"Redistributed {result.redistributed_tss:.0f} TSS "
        f"across {len(result.affected_days)} other training day(s)."
    )
    if result.warning:
        summary += f" ⚠️ {result.warning}"
    flash(summary)

    if session.get("today_day", 1) == day:
        session["today_day"] = min(day + 1, 84)

    return redirect(url_for("dashboard"))


@app.route("/advance-day", methods=["POST"])
def advance_day():
    """
    Manually advance today_day by one. Useful during the usability evaluation
    to simulate time passing without logging every workout.
    """
    current = session.get("today_day", 1)
    session["today_day"] = min(current + 1, 84)
    return redirect(url_for("dashboard"))


@app.route("/restore-session", methods=["POST"])
def restore_session():
    """Undo a missed/completed flag, reverting the day to 'planned'."""
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    day = int(request.form["day"])
    try:
        restore_planned(plan, day)
        flash(f"Day {day} restored to planned.")
    except ValueError as e:
        flash(str(e))

    return redirect(url_for("dashboard"))


@app.route("/reset", methods=["POST"])
def reset():
    """Discard the current plan so the user can create a new one."""
    PLANS.pop(_session_id(), None)
    return redirect(url_for("index"))


# ─── JSON API (used by future frontend / for testing) ────────────────────────

@app.route("/api/plan", methods=["GET"])
def api_plan():
    """Return the current plan as JSON. Returns 404 if no plan exists."""
    plan = _current_plan()
    if plan is None:
        return jsonify({"error": "No plan for this session"}), 404
    return jsonify(plan.to_dict())


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)