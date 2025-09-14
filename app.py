# app.py — Peripatetic Timetable Generator (Wide layout + Paste workflow)
# - Full-width layout for large tables
# - Paste/import/edit student data safely (never wipes on rerun)
# - 1 or 2 selectable teaching days (Mon–Fri) with per-day start/end
# - Odd/Even year-group breaks
# - Optional locks per student: LockDay (Mon–Fri) and LockStart (HH:MM)
# - Multi-week rotation + K alternatives with diversity (forbid Week-1 repeats + penalize reuse)

from datetime import datetime, timedelta, time
import io
import csv
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

# ------------ Page / Layout ------------
st.set_page_config(page_title="Peri Timetable Generator", page_icon="🎼", layout="wide")
st.title("🎼 Peripatetic Timetable Generator")

# ------------ Constants / Helpers ------------
FMT = "%H:%M"
WEEKDAY_OPTIONS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DEFAULT_COLUMNS = ["Student", "YearGroup", "Duration", "LockDay", "LockStart"]

def to_dt(s: str) -> datetime:
    return datetime.strptime(s, FMT)

def t_to_str(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def overlaps(a0, a1, b0, b1) -> bool:
    return not (a1 <= b0 or a0 >= b1)

def build_breaks_dict(odd_break, odd_lunch, even_break, even_lunch):
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

def normalize_lock_day(raw, selected_days):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if s == "":
        return ""
    s_lower = s.lower()
    if s_lower in ("any", "none", "nan"):
        return ""
    alias = {
        "mon": "Mon", "monday": "Mon",
        "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
        "wed": "Wed", "weds": "Wed", "wednesday": "Wed",
        "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
        "fri": "Fri", "friday": "Fri",
    }
    val = alias.get(s_lower, s[:3].title())
    if val not in selected_days:
        raise RuntimeError(f"LockDay '{s}' not in selected teaching days {selected_days}")
    return val

def normalize_lock_start(raw, slot_len_min):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if s == "" or s.lower() in ("none", "nan", "any"):
        return ""
    # Try to coerce to HH:MM / HH:MM:SS
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            dt_try = datetime.strptime(s, fmt)
            return dt_try.strftime("%H:%M")
        except Exception:
            pass
    # Final attempt: split on colon
    try:
        hh, mm = s.split(":")[:2]
        dt_try = datetime.strptime(f"{int(hh):02d}:{int(mm):02d}", "%H:%M")
        return dt_try.strftime("%H:%M")
    except Exception:
        raise RuntimeError(f"LockStart '{s}' must be HH:MM (aligned to {slot_len_min}-min slots)")

# ------------ Robust paste parsing ------------
def parse_paste(text: str) -> pd.DataFrame:
    """
    Accepts data pasted from Excel/Sheets (tab/CSV), with or without headers.
    Maps columns to: Student, YearGroup, Duration, LockDay, LockStart.
    """
    if not text.strip():
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    # Try CSV/TSV sniffer; fallback to tab-separated
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        delim = dialect.delimiter
    except Exception:
        delim = "\t" if "\t" in text else ","

    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim) if any(cell.strip() for cell in r)]
    if not rows:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    # Header detection (case-insensitive)
    header = [h.strip().lower() for h in rows[0]]
    has_header = any(h in ("student", "yeargroup", "year group", "duration", "lockday", "lockstart") for h in header)
    data_rows = rows[1:] if has_header else rows

    # Column index mapping
    col_idx = {c: None for c in DEFAULT_COLUMNS}
    if has_header:
        for i, h in enumerate(header):
            if h in ("student",):
                col_idx["Student"] = i
            elif h in ("yeargroup", "year group"):
                col_idx["YearGroup"] = i
            elif h in ("duration",):
                col_idx["Duration"] = i
            elif h in ("lockday",):
                col_idx["LockDay"] = i
            elif h in ("lockstart",):
                col_idx["LockStart"] = i
    else:
        # Assume first three columns are Student, YearGroup, Duration; optional LockDay, LockStart follow
        for i, key in enumerate(DEFAULT_COLUMNS):
            if i < len(rows[0]):
                col_idx[key] = i

    # Build dataframe
    out = []
    for r in data_rows:
        def get(idx): return r[idx].strip() if (idx is not None and idx < len(r)) else ""
        out.append({
            "Student": get(col_idx["Student"]),
            "YearGroup": get(col_idx["YearGroup"]),
            "Duration": get(col_idx["Duration"]),
            "LockDay": get(col_idx["LockDay"]),
            "LockStart": get(col_idx["LockStart"]),
        })
    df = pd.DataFrame(out, columns=DEFAULT_COLUMNS)
    # Drop completely empty student rows
    df = df[df["Student"].astype(str).str.strip() != ""]
    return df.reset_index(drop=True)

# ------------ Solver (N-day, locks, alternatives) ------------
def solve_multiweek_ndays_with_locks(
    students_df: pd.DataFrame,
    days: dict,                   # {"Tue": ("09:00","15:30"), ...}
    slot_len_min: int,
    breaks: dict,
    weeks: int,
    min_shift_min: int,
    forbid_week1: set | None = None,
    penalty_map: dict | None = None
):
    def to_dt_(s: str) -> datetime: return datetime.strptime(s, FMT)

    # Build slots per day
    slot_time = {}
    slots_per_day = {}
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

    # Ensure columns
    df = students_df.rename(columns={"Year Group": "YearGroup"}).copy()
    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # Coerce numerics
    df["YearGroup"] = pd.to_numeric(df["YearGroup"], errors="coerce")
    df["Duration"]  = pd.to_numeric(df["Duration"],  errors="coerce")

    # Helper check
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

    # Build students, apply locks
    students = []
    for s_idx, r in enumerate(df.itertuples(index=False)):
        name = str(r.Student).strip() or f"Student{s_idx+1}"
        yg   = r.YearGroup
        dur  = r.Duration
        if pd.isna(yg) or pd.isna(dur):
            raise RuntimeError(f"{name}: YearGroup and Duration must be numbers")

        yg, dur = int(yg), int(dur)
        d_slots = dur // slot_len_min
        if d_slots * slot_len_min != dur:
            raise RuntimeError(f"{name}: duration {dur} not multiple of slot length {slot_len_min}")

        lock_day = normalize_lock_day(getattr(r, "LockDay", ""), day_names)
        lock_start_str = normalize_lock_start(getattr(r, "LockStart", ""), slot_len_min)

        candidate_days = [lock_day] if lock_day else day_names

        valid = []
        for dname in candidate_days:
            if lock_start_str:
                # exact slot index for locked time
                slot_idx = None
                for t in range(slots_per_day[dname]):
                    if slot_time[(dname, t)].strftime("%H:%M") == lock_start_str:
                        slot_idx = t
                        break
                if slot_idx is None:
                    raise RuntimeError(f"{name}: LockStart {lock_start_str} not aligned to a {slot_len_min}-min slot on {dname}.")
                if start_is_valid(dname, slot_idx, d_slots, yg):
                    valid.append((dname, slot_idx))
            else:
                for t_idx in range(slots_per_day[dname]):
                    if start_is_valid(dname, t_idx, d_slots, yg):
                        valid.append((dname, t_idx))

        if not valid:
            raise RuntimeError(f"{name}: no valid start times after applying breaks/locks")

        students.append({"name": name, "yg": yg, "dur": dur, "d_slots": d_slots, "valid": valid})

    penalty_map = penalty_map or {}
    forbid_week1 = forbid_week1 or set()

    def solve_week(week_idx, prev_assign=None, enforce_shift=True):
        model = cp_model.CpModel()

        valid_map = {}
        for s_idx, s in enumerate(students):
            v = s["valid"]
            if enforce_shift and prev_assign is not None and prev_assign[s_idx] is not None and min_shift_slots > 0:
                prev_day, prev_t = prev_assign[s_idx]
                v = [(d, t) for (d, t) in v if (d != prev_day) or (abs(t - prev_t) >= min_shift_slots)] or s["valid"]
            if week_idx == 0 and forbid_week1:
                v = [(d, t) for (d, t) in v if (s_idx, d, t) not in forbid_week1] or v
            valid_map[s_idx] = v

        x = {}
        for s_idx, v in valid_map.items():
            for (d, t) in v:
                x[(s_idx, d, t)] = model.NewBoolVar(f"x_s{s_idx}_{d}_{t}")

        # Each student exactly once
        for s_idx, v in valid_map.items():
            model.Add(sum(x[(s_idx, d, t)] for (d, t) in v) == 1)

        # Capacity per (day, slot)
        day_names = list(days.keys())
        for dname in day_names:
            for k in range(slots_per_day[dname]):
                active = []
                for s_idx, s in enumerate(students):
                    for (d, t) in valid_map[s_idx]:
                        if d == dname and t <= k < t + s["d_slots"]:
                            active.append(x[(s_idx, d, t)])
                if active:
                    model.Add(sum(active) <= 1)

        # Objective: prefer earlier; lightly prefer first day; penalize repeats across alternatives
        day_names = list(days.keys())
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

    prev = None
    rows = []
    chosen_log = []
    for w in range(weeks):
        chosen, res = solve_week(week_idx=w, prev_assign=prev, enforce_shift=(prev is not None))
        if chosen is None and prev is not None:
            chosen, res = solve_week(week_idx=w, prev_assign=prev, enforce_shift=False)
        if chosen is None:
            raise RuntimeError(f"Week {w+1}: infeasible. Try wider windows or adjust durations/breaks/locks.")

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

def generate_alternatives(students_df, days, slot_len_min, breaks, weeks, min_shift_min, num_alts=3):
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

        week1 = chosen_log[0]
        for s_idx, (day, slot_idx) in enumerate(week1):
            all_week1_forbids.add((s_idx, day, slot_idx))

        for w_assign in chosen_log:
            for s_idx, (day, slot_idx) in enumerate(w_assign):
                penalty_map[(s_idx, day, slot_idx)] = penalty_map.get((s_idx, day, slot_idx), 0) + 500

    return results

# ------------ State: persistent data grid ------------
if "students_df" not in st.session_state:
    st.session_state["students_df"] = pd.DataFrame(columns=DEFAULT_COLUMNS)

# ------------ Layout: Left (data) | Right (controls) ------------
left, right = st.columns([7, 5], gap="large")

with left:
    st.subheader("Student list")
    st.caption("Paste from Excel/Sheets or type directly. Required: Student, YearGroup, Duration. Optional: LockDay (Mon–Fri) and LockStart (HH:MM).")

    paste_txt = st.text_area(
        "Quick paste (CSV/TSV or copied rows). Leave blank if not using.",
        height=120,
        placeholder="Example without headers:\nA\t7\t15\tTue\t10:00\nB\t9\t30\n\nOr with headers:\nStudent,YearGroup,Duration,LockDay,LockStart\nA,7,15,Tue,10:00"
    )

    c_imp1, c_imp2, c_imp3 = st.columns([1,1,2])
    with c_imp1:
        if st.button("Append pasted rows"):
            try:
                parsed = parse_paste(paste_txt)
                st.session_state["students_df"] = pd.concat(
                    [st.session_state["students_df"], parsed],
                    ignore_index=True
                )
                st.success(f"Appended {len(parsed)} row(s).")
            except Exception as e:
                st.error(f"Paste error: {e}")
    with c_imp2:
        up = st.file_uploader("Import CSV", type=["csv"], label_visibility="collapsed")
        if up is not None and st.button("Load CSV below"):
            try:
                loaded = pd.read_csv(up, encoding="utf-8-sig")
                for c in DEFAULT_COLUMNS:
                    if c not in loaded.columns:
                        loaded[c] = ""
                st.session_state["students_df"] = loaded[DEFAULT_COLUMNS]
                st.success("CSV loaded into the grid.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    # Editable grid (wide, persistent)
    edited_df = st.data_editor(
        st.session_state["students_df"],
        key="students_grid",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Student": st.column_config.TextColumn(required=True),
            "YearGroup": st.column_config.NumberColumn(min_value=1, max_value=13, step=1),
            "Duration": st.column_config.NumberColumn(min_value=5, max_value=240, step=5),
            "LockDay": st.column_config.SelectboxColumn(options=["", "Mon", "Tue", "Wed", "Thu", "Fri", "Any", "None"]),
            "LockStart": st.column_config.TextColumn(help="HH:MM or blank"),
        },
    )
    # Persist edits safely every rerun (no wiping)
    st.session_state["students_df"] = edited_df

    c_exp1, c_exp2, c_exp3 = st.columns([1,1,2])
    with c_exp1:
        if st.button("Download current data"):
            buff = io.StringIO()
            st.session_state["students_df"].to_csv(buff, index=False)
            st.download_button("Save CSV", buff.getvalue().encode("utf-8"),
                               file_name="students.csv", mime="text/csv")
    with c_exp2:
        if st.button("Reset table"):
            st.session_state["students_df"] = pd.DataFrame(columns=DEFAULT_COLUMNS)
            st.rerun()

with right:
    st.subheader("Day & rules")
    day_count = st.radio("Teaching days", [1, 2], index=0, horizontal=True)
    default_days = ["Tue"] if day_count == 1 else ["Tue", "Thu"]
    selected_days = st.multiselect("Select day(s)", options=WEEKDAY_OPTIONS, default=default_days)

    can_proceed_days = len(selected_days) == day_count
    if not can_proceed_days:
        st.warning(f"Select exactly {day_count} day{'s' if day_count==2 else ''}.")

    day_windows = {}
    if can_proceed_days:
        day_cols = st.columns(len(selected_days))
        for i, dname in enumerate(selected_days):
            with day_cols[i]:
                s = st.time_input(f"{dname} start", value=time(9, 0), key=f"{dname}_start")
                e = st.time_input(f"{dname} end",   value=time(15, 30), key=f"{dname}_end")
                day_windows[dname] = (t_to_str(s), t_to_str(e))

    slot_len = st.number_input("Slot length (min)", min_value=5, max_value=60, step=5, value=15)
    weeks    = st.number_input("Number of weeks", min_value=1, max_value=40, step=1, value=6)
    min_shift = st.number_input("Min shift per week (min)", min_value=0, max_value=240, step=15, value=60)
    num_alts = st.number_input("Number of alternatives", min_value=1, max_value=10, step=1, value=3)

    st.subheader("Breaks/Lunch")
    c1, c2 = st.columns(2)
    with c1:
        odd_break = (st.time_input("Odd Y: Break start", time(11, 5)),
                     st.time_input("Odd Y: Break end",   time(11, 20)))
        odd_lunch = (st.time_input("Odd Y: Lunch start", time(13, 25)),
                     st.time_input("Odd Y: Lunch end",   time(14, 0)))
    with c2:
        even_break = (st.time_input("Even Y: Break start", time(10, 0)),
                      st.time_input("Even Y: Break end",   time(10, 15)))
        even_lunch = (st.time_input("Even Y: Lunch start", time(12, 20)),
                      st.time_input("Even Y: Lunch end",   time(12, 55)))

    def _clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Trim text
        for col in ["Student", "LockDay", "LockStart"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        # Remove empty students
        df = df[df["Student"] != ""]
        if df.empty:
            raise RuntimeError("Add some students first.")
        # Numerics
        df["YearGroup"] = pd.to_numeric(df["YearGroup"], errors="coerce")
        df["Duration"]  = pd.to_numeric(df["Duration"],  errors="coerce")
        if df["YearGroup"].isna().any() or df["Duration"].isna().any():
            bad = df[df["YearGroup"].isna() | df["Duration"].isna()]
            raise RuntimeError(f"Non-numeric YearGroup/Duration detected: {bad[['Student','YearGroup','Duration']].to_dict('records')}")
        return df[DEFAULT_COLUMNS]

    can_generate = (not st.session_state["students_df"].empty) and can_proceed_days and (len(day_windows) == day_count)

    if st.button("Generate timetable(s)", type="primary", disabled=not can_generate):
        try:
            students_df = _clean_and_validate(st.session_state["students_df"])
            breaks = build_breaks_dict(odd_break, odd_lunch, even_break, even_lunch)

            alt_tables = generate_alternatives(
                students_df=students_df,
                days=day_windows,
                slot_len_min=int(slot_len),
                breaks=breaks,
                weeks=int(weeks),
                min_shift_min=int(min_shift),
                num_alts=int(num_alts),
            )

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



