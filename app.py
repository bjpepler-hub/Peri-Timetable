# app.py — Peripatetic Timetable Generator
# Features:
# - Manual entry or CSV upload
# - Choose 1 or 2 teaching days (Mon–Fri), with day-specific start/end
# - Odd/Even year-group break/lunch windows
# - Locks per student: fixed Day and/or fixed Start time (HH:MM)
# - Multi-week rotation (>= min shift on same day; switching day allowed)
# - Generate K alternatives (default 3): forbid prior Week-1 choices + penalise repeats

from datetime import datetime, timedelta, time
import io
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

# ---------------- Helpers ----------------
FMT = "%H:%M"

def to_dt(s: str) -> datetime:
    return datetime.strptime(s, FMT)

def overlaps(a0, a1, b0, b1) -> bool:
    return not (a1 <= b0 or a0 >= b1)

def t_to_str(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def build_breaks_dict(odd_break, odd_lunch, even_break, even_lunch):
    """Return dict {year_group: [(start,end), (start,end)]} for odd/even year groups."""
    odd_years = [7, 9, 11, 13]
    even_years = [8, 10, 12]
    b = {}
    for yg in odd_years:
        b[yg] = [(t_to_str(odd_break[0]), t_to_str(odd_break[1])),
                 (t_to_str(odd_lunch[0]), t_to_str(odd_lunch[1]))]
    for yg in even_years:
        b[yg] = [(t_to_str(even_break[0]), t_to_str(even_break[1])),
                 (t_to_str(even_lunch[0]), t_to_str(even_lunch[1]))]
    return b

def next_letter(n: int) -> str:
    """A, B, C, ... Z, AA, AB, ... for auto-naming."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# ---------------- N-day solver with locks + alternatives ----------------
def solve_multiweek_ndays_with_locks(
    students_df: pd.DataFrame,
    days: dict,                   # {"Tue": ("09:00","15:30"), ...}
    slot_len_min: int,
    breaks: dict,
    weeks: int,
    min_shift_min: int,
    forbid_week1: set | None = None,         # {(s_idx, day, slot_idx), ...} to forbid in Week 1
    penalty_map: dict | None = None          # {(s_idx, day, slot_idx): weight}
):
    """
    Multi-day scheduler supporting:
      - 1 or 2 days (keys of `days`)
      - per-student locks (Day, Start)
      - rotation across weeks (>= min_shift on same day)
      - optional forbids for Week 1 to force alternatives
      - optional penalties to push diversity across all weeks

    Returns: (out_df, chosen_log)
      out_df: DataFrame [Week, Day, Student, YearGroup, Start, End, Duration(min)]
      chosen_log: list of per-week chosen assignments: list[ list[(s_idx, day, slot_idx)] ]
    """
    def to_dt_(s: str) -> datetime: return datetime.strptime(s, FMT)

    # Build slots per day
    slot_time = {}      # (day, slot_idx) -> datetime
    slots_per_day = {}  # day -> number of slots
    for dname, (start_str, end_str) in days.items():
        start_dt, end_dt = to_dt_(start_str), to_dt_(end_str)
        t = start_dt
        idx = 0
        while t + timedelta(minutes=slot_len_min) <= end_dt:
            slot_time[(dname, idx)] = t
            idx += 1
            t += timedelta(minutes=slot_len_min)
        slots_per_day[dname] = idx

    day_names = list(days.keys())
    if len(day_names) == 0:
        raise RuntimeError("No teaching days selected.")
    min_shift_slots = max(1, int(min_shift_min // slot_len_min)) if min_shift_min > 0 else 0

    # Clean headers / types; extend with Lock columns if missing
    df = students_df.rename(columns={"Year Group": "YearGroup"}).copy()
    required = {"Student", "YearGroup", "Duration"}
    if not required.issubset(df.columns):
        raise ValueError("Student table must have columns: Student, YearGroup, Duration")
    # optional lock columns
    if "LockDay" not in df.columns:
        df["LockDay"] = ""    # "", "Any", or Mon/Tue/Wed/Thu/Fri
    if "LockStart" not in df.columns:
        df["LockStart"] = ""  # "", or "HH:MM"

    df["YearGroup"] = pd.to_numeric(df["YearGroup"])
    df["Duration"]  = pd.to_numeric(df["Duration"])

    # Helper: check if lesson (d_slots) starting at (day, slot) is valid (no break overlap)
    def start_is_valid(day, slot_idx, d_slots, yg):
        for k in range(d_slots):
            t0 = slot_time.get((day, slot_idx + k))
            if t0 is None:
                return False
            t1 = t0 + timedelta(minutes=slot_len_min)
            for s, e in breaks.get(int(yg), []):
                b0, b1 = to_dt_(s), to_dt_(e)
                if overlaps(t0, t1, b0, b1):
                    return False
        return True

    # Precompute base valid starts (day, slot_idx) per student, apply locks
    students = []
    for s_idx, r in enumerate(df.itertuples(index=False)):
        name, yg, dur = str(r.Student), int(r.YearGroup), int(r.Duration)
        lock_day = str(getattr(r, "LockDay", "") or "").strip()
        lock_start_str = str(getattr(r, "LockStart", "") or "").strip()
        d_slots = dur // slot_len_min
        if d_slots * slot_len_min != dur:
            raise RuntimeError(f"{name}: duration {dur} not multiple of slot length {slot_len_min}")

        # determine candidate days based on LockDay
        if lock_day and lock_day.lower() not in ("", "any"):
            if lock_day not in day_names:
                raise RuntimeError(f"{name}: LockDay '{lock_day}' not in selected teaching days {day_names}")
            candidate_days = [lock_day]
        else:
            candidate_days = day_names

        # precompute valid starts
        valid = []
        for dname in candidate_days:
            # if LockStart provided, map to exact slot index
            if lock_start_str:
                try:
                    lock_dt = to_dt_(lock_start_str)
                except Exception:
                    raise RuntimeError(f"{name}: LockStart '{lock_start_str}' must be HH:MM, e.g., 11:45")
                # find exact slot whose start equals lock_dt
                slot_idx = None
                for t in range(slots_per_day[dname]):
                    if slot_time[(dname, t)] == lock_dt:
                        slot_idx = t
                        break
                if slot_idx is None:
                    raise RuntimeError(f"{name}: LockStart {lock_start_str} is not aligned to a {slot_len_min}-min slot on {dname}.")
                if start_is_valid(dname, slot_idx, d_slots, yg):
                    valid.append((dname, slot_idx))
                # If lock_start is set, we don't consider other starts for that day.
                continue

            # no lock start: consider all starts on this day
            for t_idx in range(slots_per_day[dname]):
                if start_is_valid(dname, t_idx, d_slots, yg):
                    valid.append((dname, t_idx))

        if not valid:
            raise RuntimeError(f"No valid start times for {name} (YG {yg}, {dur} min) after applying locks/breaks.")

        students.append({
            "name": name, "yg": yg, "dur": dur, "d_slots": d_slots, "valid": valid
        })

    penalty_map = penalty_map or {}
    forbid_week1 = forbid_week1 or set()

    # Solve one week with optional rotation/forbids/penalties
    def solve_week(week_idx, prev_assign=None, enforce_shift=True):
        model = cp_model.CpModel()

        # restrict starts near previous (same day) for rotation
        valid_map = {}
        for s_idx, s in enumerate(students):
            v = s["valid"]
            if enforce_shift and prev_assign is not None and prev_assign[s_idx] is not None and min_shift_slots > 0:
                prev_day, prev_t = prev_assign[s_idx]
                v = [(d, t) for (d, t) in v if (d != prev_day) or (abs(t - prev_t) >= min_shift_slots)] or s["valid"]

            # apply forbids on Week 1 (index 0)
            if week_idx == 0 and forbid_week1:
                v = [(d, t) for (d, t) in v if (s_idx, d, t) not in forbid_week1] or v

            valid_map[s_idx] = v

        # decision vars x[s, d, t]
        x = {}
        for s_idx, v in valid_map.items():
            for (d, t) in v:
                x[(s_idx, d, t)] = model.NewBoolVar(f"x_s{s_idx}_{d}_{t}")

        # each student exactly once
        for s_idx, v in valid_map.items():
            model.Add(sum(x[(s_idx, d, t)] for (d, t) in v) == 1)

        # capacity per (day, slot)
        for dname in day_names:
            for k in range(slots_per_day[dname]):
                active = []
                for s_idx, s in enumerate(students):
                    for (d, t) in valid_map[s_idx]:
                        if d == dname and t <= k < t + s["d_slots"]:
                            active.append(x[(s_idx, d, t)])
                if active:
                    model.Add(sum(active) <= 1)

        # objective: prefer earlier, lightly prefer first day, and penalise repeats across alternatives
        objective_terms = []
        for s_idx, v in valid_map.items():
            for (d, t) in v:
                base_cost = (0 if d == day_names[0] else 10000) + t
                repeat_pen = penalty_map.get((s_idx, d, t), 0)
                objective_terms.append((base_cost + repeat_pen) * x[(s_idx, d, t)])
        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        res = solver.Solve(model)
        if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, None

        chosen = [None] * len(students)
        for s_idx, v in valid_map.items():
            for (d, t) in v:
                if solver.Value(x[(s_idx, d, t)]) == 1:
                    chosen[s_idx] = (d, t)
                    break
        return chosen, res

    # iterate weeks
    prev = None
    rows = []
    chosen_log = []
    for w in range(weeks):
        chosen, res = solve_week(week_idx=w, prev_assign=prev, enforce_shift=(prev is not None))
        if chosen is None and prev is not None:
            # relax rotation if infeasible
            chosen, res = solve_week(week_idx=w, prev_assign=prev, enforce_shift=False)
        if chosen is None:
            raise RuntimeError(f"Week {w+1}: infeasible. Try wider windows or adjust durations/breaks/locks.")

        # record
        chosen_log.append(chosen)
        for s_idx, s in enumerate(students):
            d, t = chosen[s_idx]
            start = slot_time[(d, t)]
            end   = start + timedelta(minutes=s["dur"])
            rows.append({
                "Week": w + 1,
                "Day": d,
                "Student": s["name"],
                "YearGroup": s["yg"],
                "Start": start.strftime(FMT),
                "End": end.strftime(FMT),
                "Duration(min)": s["dur"],
            })
        prev = chosen

    out = pd.DataFrame(rows).sort_values(["Week", "Day", "Start", "Student"]).reset_index(drop=True)
    return out, chosen_log

def generate_alternatives(
    students_df: pd.DataFrame,
    days: dict,
    slot_len_min: int,
    breaks: dict,
    weeks: int,
    min_shift_min: int,
    num_alts: int = 3,
):
    """
    Produce multiple alternative timetables.
    Strategy:
      - Alt #1: normal solve
      - Alt #k>1: forbid reusing Week-1 placements from any previous alt
                  + add penalty for any (s,day,slot) seen before (all weeks)
    """
    results = []
    all_week1_forbids = set()
    penalty_map = {}

    for alt in range(1, num_alts + 1):
        out, chosen_log = solve_multiweek_ndays_with_locks(
            students_df=students_df,
            days=days,
            slot_len_min=slot_len_min,
            breaks=breaks,
            weeks=weeks,
            min_shift_min=min_shift_min,
            forbid_week1=all_week1_forbids if alt > 1 else None,
            penalty_map=penalty_map if alt > 1 else None,
        )
        out = out.copy()
        out.insert(0, "Alternative", alt)
        results.append(out)

        # Update forbids (Week 1 placements)
        week1 = chosen_log[0]  # list[(day,slot)] per student
        for s_idx, (day, slot_idx) in enumerate(week1):
            all_week1_forbids.add((s_idx, day, slot_idx))

        # Update penalties: penalise any reuse of (s, day, slot) across any week
        for w_assign in chosen_log:
            for s_idx, (day, slot_idx) in enumerate(w_assign):
                penalty_map[(s_idx, day, slot_idx)] = penalty_map.get((s_idx, day, slot_idx), 0) + 500

    return results  # list of DataFrames (one per alternative)

# ---------------- UI ----------------
st.set_page_config(page_title="Peri Timetable Generator", page_icon="🎼", layout="centered")
st.title("🎼 Peripatetic Timetable Generator")

# Input method: manual or CSV
method = st.radio("How do you want to add students?", ["Enter manually", "Upload CSV"], horizontal=True)
students_df = None

if method == "Enter manually":
    if "students" not in st.session_state:
        st.session_state["students"] = []

    st.subheader("Add a student")
    with st.form("add_student"):
        name = st.text_input("Student name (optional — leave blank to auto-letter)")
        yg = st.selectbox("Year group", options=list(range(3, 14)), index=4)  # default 7
        dur = st.selectbox("Lesson length (minutes)", options=[15, 30, 45, 60], index=0)
        # Locks (optional)
        lock_day = st.selectbox("Lock Day (optional)", options=["", "Any", "Mon", "Tue", "Wed", "Thu", "Fri"], index=0)
        lock_start = st.text_input("Lock Start (optional, HH:MM)", value="")
        add_clicked = st.form_submit_button("Add student")

    if add_clicked:
        display_name = name.strip() or next_letter(len(st.session_state["students"]))
        st.session_state["students"].append({
            "Student": display_name,
            "YearGroup": int(yg),
            "Duration": int(dur),
            "LockDay": lock_day,
            "LockStart": lock_start.strip(),
        })
        st.success(f"Added: {display_name} (YG {yg}, {dur} min"
                   f"{' • Lock '+lock_day if lock_day else ''}"
                   f"{' @ '+lock_start if lock_start else ''})")
        st.rerun()

    st.subheader("Current students")
    if st.session_state["students"]:
        students_df = pd.DataFrame(st.session_state["students"])
        st.dataframe(students_df, use_container_width=True)
    else:
        st.info("No students yet — add some above.")

else:
    st.markdown("Upload your `students.csv` with columns **Student, YearGroup, Duration**. "
                "Optional columns: **LockDay**, **LockStart** (HH:MM).")
    csv_file = st.file_uploader("Upload students.csv", type=["csv"])
    if csv_file is not None:
        students_df = pd.read_csv(csv_file, encoding="utf-8-sig")

# Day & rules — choose one or two days, and which weekdays
st.subheader("Day & rules (choose one or two teaching days)")
day_count = st.radio("How many teaching days?", [1, 2], index=0, horizontal=True)

weekday_options = ["Mon", "Tue", "Wed", "Thu", "Fri"]
default_days = ["Tue"] if day_count == 1 else ["Tue", "Thu"]
selected_days = st.multiselect(
    "Select the day(s)",
    options=weekday_options,
    default=default_days
)

# Enforce exact selection count
if len(selected_days) != day_count:
    st.warning(f"Please select exactly {day_count} day{'s' if day_count==2 else ''}.")
    can_proceed_days = False
else:
    can_proceed_days = True

# Per-selected-day start/end times
day_windows = {}
if can_proceed_days:
    cols = st.columns(len(selected_days))
    for i, dname in enumerate(selected_days):
        with cols[i]:
            start_t = st.time_input(f"{dname} start", value=time(9, 0), key=f"{dname}_start")
            end_t   = st.time_input(f"{dname} end",   value=time(15, 30), key=f"{dname}_end")
            day_windows[dname] = (t_to_str(start_t), t_to_str(end_t))

slot_len = st.number_input("Slot length (minutes)", min_value=5, max_value=60, step=5, value=15)
weeks    = st.number_input("Number of weeks", min_value=1, max_value=40, step=1, value=6)
min_shift = st.number_input("Min shift each week (minutes)", min_value=0, max_value=240, step=15, value=60)
num_alts = st.number_input("Number of alternatives", min_value=1, max_value=10, step=1, value=3)

st.subheader("Breaks/Lunch (Odd vs Even year groups)")
c1, c2 = st.columns(2)
with c1:
    odd_break = (st.time_input("Odd years: Break start", time(11, 5)),
                 st.time_input("Odd years: Break end",   time(11, 20)))
    odd_lunch = (st.time_input("Odd years: Lunch start", time(13, 25)),
                 st.time_input("Odd years: Lunch end",   time(14, 0)))
with c2:
    even_break = (st.time_input("Even years: Break start", time(10, 0)),
                  st.time_input("Even years: Break end",   time(10, 15)))
    even_lunch = (st.time_input("Even years: Lunch start", time(12, 20)),
                  st.time_input("Even years: Lunch end",   time(12, 55)))

# Generate timetable(s)
can_generate = (
    students_df is not None and
    not (isinstance(students_df, pd.DataFrame) and students_df.empty) and
    can_proceed_days and
    len(day_windows) == day_count
)

if st.button("Generate timetable(s)", type="primary", disabled=not can_generate):
    try:
        # Validate CSV headers if using Upload
        if method == "Upload CSV":
            required = {"Student", "YearGroup", "Duration"}
            if not required.issubset(students_df.columns):
                st.error("CSV must have columns: Student, YearGroup, Duration")
                st.stop()
            # Optional columns fine

        breaks = build_breaks_dict(odd_break, odd_lunch, even_break, even_lunch)

        # Generate alternatives
        alt_tables = generate_alternatives(
            students_df=students_df,
            days=day_windows,
            slot_len_min=int(slot_len),
            breaks=breaks,
            weeks=int(weeks),
            min_shift_min=int(min_shift),
            num_alts=int(num_alts),
        )

        # Show in tabs + downloads
        tabs = st.tabs([f"Alternative {i+1}" for i in range(len(alt_tables))])
        for i, (tab, df_out) in enumerate(zip(tabs, alt_tables), start=1):
            with tab:
                st.dataframe(df_out, use_container_width=True)
                buff = io.StringIO()
                df_out.to_csv(buff, index=False)
                st.download_button(f"Download Alternative {i} (CSV)",
                                   buff.getvalue().encode("utf-8"),
                                   file_name=f"timetable_alt_{i}.csv",
                                   mime="text/csv")

        st.success("Generated alternatives successfully.")
    except Exception as e:
        st.error(str(e))



