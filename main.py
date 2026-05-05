import uuid
from typing import Optional

from flask import (
    Flask, render_template, request, redirect, url_for, session, jsonify, flash
)

from engine.scheduler import generate_plan, TrainingPlan
from engine.gpx_parser import parse_gpx, GPXSummary
from engine.banister_model import compute_curve
from engine.reoptimiser import (
    mark_missed, mark_completed, restore_planned, ReoptimisationResult,
    STATUS_COMPLETED, STATUS_MISSED, STATUS_PLANNED,
)


app = Flask(__name__)

app.secret_key = "cypath-dev-secret-change-in-production"



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
    training_days = [int(d) for d in request.form.getlist("training_days")]

    if not training_days:
        flash("Please select at least one training day.")
        return redirect(url_for("index"))

    recovery_weeks = request.form.get("recovery_weeks") == "on"

    goal_preset = request.form.get("goal_preset", "100")
    PRESET_MAP = {"50": (50.0, 6), "75": (75.0, 9), "100": (100.0, 12)}
    if goal_preset in PRESET_MAP:
        goal_km, total_weeks = PRESET_MAP[goal_preset]
    else:
        try:
            goal_km     = float(request.form.get("custom_km",    "100"))
            total_weeks = int(request.form.get("custom_weeks", "12"))
            goal_km     = max(10.0, min(goal_km, 1000.0))
            total_weeks = max(4,    min(total_weeks, 52))
        except ValueError:
            goal_km, total_weeks = 100.0, 12

    try:
        plan = generate_plan(
            profile=profile,
            training_days=training_days,
            recovery_weeks=recovery_weeks,
            goal_km=goal_km,
            total_weeks=total_weeks,
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
    Main dashboard view. Shows today's workout, the readiness curve, and
    progress towards the goal. The full training plan lives on its own page.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    today_day = session.get("today_day", 1)

    # ── Compute the Banister fitness curve (used for chart + today's snapshot)
    curve = compute_curve(plan, today_day=today_day)

    chart_tsb   = [s.tsb for s in curve]
    split_index = today_day - 1  # vertical "today" marker on the chart

    # Current day summary for the header card.
    today_workout = next((w for w in plan.workouts if w.day == today_day), None)
    today_snap    = curve[today_day - 1] if curve else None

    pending_gpx = session.get("pending_gpx")
    return render_template(
        "dashboard.html",
        plan=plan,
        today_day=today_day,
        today_workout=today_workout,
        today_snap=today_snap,
        chart_tsb=chart_tsb,
        split_index=split_index,
        STATUS_COMPLETED=STATUS_COMPLETED,
        STATUS_MISSED=STATUS_MISSED,
        STATUS_PLANNED=STATUS_PLANNED,
        pending_gpx=pending_gpx,
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
            session["today_day"] = min(day + 1, plan.total_days())
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
        session["today_day"] = min(day + 1, plan.total_days())

    return redirect(url_for("dashboard"))


@app.route("/advance-day", methods=["POST"])
def advance_day():
    """
    Manually advance today_day by one. Useful during the usability evaluation
    to simulate time passing without logging every workout.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))
    current = session.get("today_day", 1)
    session["today_day"] = min(current + 1, plan.total_days())
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


# ─── GPX import ──────────────────────────────────────────────────────────────

@app.route("/upload-gpx", methods=["POST"])
def upload_gpx():
    """
    Accept a GPX file, parse it, and store the summary in the session so the
    confirmation card can be shown on the dashboard.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    gpx_file = request.files.get("gpx_file")
    if not gpx_file or gpx_file.filename == "":
        flash("No file selected. Please choose a .gpx file.")
        return redirect(url_for("dashboard"))

    if not gpx_file.filename.lower().endswith(".gpx"):
        flash("Only .gpx files are supported.")
        return redirect(url_for("dashboard"))

    try:
        summary = parse_gpx(gpx_file.read())
    except ValueError as e:
        flash(f"Could not read GPX file: {e}")
        return redirect(url_for("dashboard"))

    # Store parsed summary in session for the confirmation step.
    session["pending_gpx"] = {
        "day":           session.get("today_day", 1),
        "distance_km":   summary.distance_km,
        "duration_str":  summary.duration_str,
        "elevation_m":   summary.elevation_gain_m,
        "avg_hr":        summary.avg_hr_bpm,
        "estimated_tss": summary.estimated_tss,
        "method":        summary.method,
        "activity_name": summary.activity_name,
    }
    return redirect(url_for("dashboard"))


@app.route("/confirm-gpx", methods=["POST"])
def confirm_gpx():
    """
    Log the GPX session with the (possibly user-adjusted) TSS value.
    Updates the workout's target_tss to the actual value before marking
    it completed, so the Banister model reflects real training data.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    pending = session.pop("pending_gpx", None)
    if pending is None:
        flash("No pending GPX session to confirm.")
        return redirect(url_for("dashboard"))

    try:
        actual_tss = float(request.form.get("actual_tss", pending["estimated_tss"]))
        actual_tss = max(0.0, round(actual_tss, 1))
    except ValueError:
        actual_tss = pending["estimated_tss"]

    day = pending["day"]

    # Update the workout's target_tss to the actual GPX value so the
    # Banister model tracks real load rather than the scheduled estimate.
    for w in plan.workouts:
        if w.day == day:
            w.target_tss = actual_tss
            break

    try:
        mark_completed(plan, day)
        if session.get("today_day", 1) == day:
            session["today_day"] = min(day + 1, plan.total_days())
        flash(
            f"GPX session logged — {pending['distance_km']} km · "
            f"{pending['duration_str']} · {actual_tss} TSS recorded."
        )
    except ValueError as e:
        flash(str(e))

    return redirect(url_for("dashboard"))


@app.route("/dismiss-gpx", methods=["POST"])
def dismiss_gpx():
    """Dismiss the GPX confirmation card without logging."""
    session.pop("pending_gpx", None)
    return redirect(url_for("dashboard"))


# ─── Plan page ───────────────────────────────────────────────────────────────

@app.route("/plan", methods=["GET"])
def plan_page():
    """Dedicated full training plan page — linked from the Train nav button."""
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    today_day    = session.get("today_day", 1)
    current_week = (today_day - 1) // 7 + 1

    weeks = []
    for week_number in range(1, plan.total_weeks + 1):
        workouts = [w for w in plan.workouts if w.week == week_number]
        training_days = sum(1 for w in workouts if w.phase != "Rest")
        done = sum(1 for w in workouts
                   if w.phase != "Rest" and getattr(w, "status", "planned") == STATUS_COMPLETED)
        pct = round((done / training_days * 100)) if training_days else 0
        weeks.append({
            "number":       week_number,
            "phase":        next((w.phase for w in workouts if w.phase != "Rest"), "Rest"),
            "total_tss":    round(plan.weekly_tss(week_number), 1),
            "workouts":     workouts,
            "training_days":training_days,
            "pct_done":     pct,
        })

    return render_template(
        "plan.html",
        plan=plan,
        weeks=weeks,
        today_day=today_day,
        current_week=current_week,
        STATUS_COMPLETED=STATUS_COMPLETED,
        STATUS_MISSED=STATUS_MISSED,
        STATUS_PLANNED=STATUS_PLANNED,
    )


# ─── Settings page ────────────────────────────────────────────────────────────

@app.route("/settings", methods=["GET"])
def settings_page():
    """Settings page — dark mode toggle and restart option."""
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    today_day = session.get("today_day", 1)
    pct_done  = round((today_day / plan.total_days()) * 100)

    return render_template(
        "settings.html",
        plan=plan,
        today_day=today_day,
        pct_done=pct_done,
    )


# ─── Form page ───────────────────────────────────────────────────────────────

@app.route("/form", methods=["GET"])
def form_page():
    """
    Form & Fitness page — visualises the Banister CTL/ATL/TSB curves over the
    full plan, with today's snapshot and a TSB-based readiness interpretation.
    """
    plan = _current_plan()
    if plan is None:
        return redirect(url_for("index"))

    today_day = session.get("today_day", 1)
    curve     = compute_curve(plan, today_day=today_day)

    chart_ctl   = [s.ctl for s in curve]
    chart_atl   = [s.atl for s in curve]
    chart_tsb   = [s.tsb for s in curve]
    split_index = today_day - 1
    today_snap  = curve[today_day - 1] if curve else None

    return render_template(
        "form.html",
        plan=plan,
        today_day=today_day,
        today_snap=today_snap,
        chart_ctl=chart_ctl,
        chart_atl=chart_atl,
        chart_tsb=chart_tsb,
        split_index=split_index,
    )


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
    app.run(debug=False, host="0.0.0.0", port=5000)