"""
Apple Fitness Dashboard (Streamlit) — Upload export.zip

Run:
  pip install streamlit pandas numpy matplotlib
  streamlit run Stream_App.py

Input:
  Apple Health export.zip (contains a folder like apple_health_report/export.xml)

Notes:
- Uses streaming XML parsing (ET.iterparse) to handle huge export.xml safely.
- Only parses "Workout" and "ActivitySummary".
- Filters to selected YEAR during parsing to reduce memory usage.
"""

import os
import io
import re
import zipfile
import shutil
import hashlib
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(page_title="Fitness Dashboard", layout="centered")


# =========================
# Maps / display helpers
# =========================
def clean_type_name(t):
    if not t:
        return "Unknown"
    return t.replace("HKWorkoutActivityType", "")


PRETTY_NAME_MAP = {
    "FunctionalStrengthTraining": "Functional Strength Training",
    "TraditionalStrengthTraining": "Traditional Strength Training",
    "HighIntensityIntervalTraining": "HIIT",
}

EMOJI_MAP = {
    "FunctionalStrengthTraining": "🏋️",
    "TraditionalStrengthTraining": "🏋️",
    "HighIntensityIntervalTraining": "🔥",
    "Running": "🏃",
    "Walking": "🚶",
    "Cycling": "🚴",
    "Swimming": "🏊",
    "Tennis": "🎾",
    "Badminton": "🏸",
    "Pickleball": "🏓",
    "Basketball": "🏀",
    "Soccer": "⚽",
    "Elliptical": "🌀",
    "Rowing": "🚣",
    "StairClimbing": "⬆️",
    "Hiking": "🥾",
    "Yoga": "🧘",
    "Pilates": "🧘",
    "CoreTraining": "🧠",
    "CardioDance": "💃",
    "Cooldown": "❄️",
    "PreparationAndRecovery": "🛠️",
    "Bowling": "🎳",
    "SkatingSports": "⛸️",
}


def pretty_name(t):
    return PRETTY_NAME_MAP.get(t, t)


# =========================
# CSS (emoji list + hardcore card alignment)
# =========================
st.markdown(
    """
<style>
/* Centered page container feel */
.block-container { padding-top: 1.4rem; }

/* Emoji list: aligned icons and text */
ul.emoji-list {
  list-style: none;
  padding: 0;
  margin: 10px auto 0 auto;
  max-width: 760px;
}
ul.emoji-list li {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin: 8px 0;
}
ul.emoji-list .icon {
  width: 34px;
  text-align: center;
  font-size: 18px;
  line-height: 1;
}
ul.emoji-list .txt {
  font-size: 16px;
  line-height: 1.35;
  white-space: nowrap;
}

/* Hardcore card */
.hardcore-card {
  max-width: 760px;
  margin: 0 auto;
  text-align: center;
}
.hardcore-meta {
  font-size: 16px;
  line-height: 1.6;
  margin: 8px 0 14px 0;
}
.kcal-note {
  color: #666;
  font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Zip handling (extract + find export.xml)
# =========================
def fingerprint_file(uploaded_file) -> str:
    """
    Create a lightweight fingerprint without reading entire file into RAM.
    Uses: filename + size + sha1(first 2MB).
    """
    name = getattr(uploaded_file, "name", "uploaded.zip")
    try:
        size = uploaded_file.size
    except Exception:
        # fallback: try buffer length
        try:
            size = len(uploaded_file.getbuffer())
        except Exception:
            size = 0

    # read first 2MB then reset pointer
    pos = uploaded_file.tell()
    head = uploaded_file.read(2 * 1024 * 1024)
    uploaded_file.seek(pos)

    h = hashlib.sha1()
    h.update(name.encode("utf-8", errors="ignore"))
    h.update(str(size).encode("utf-8"))
    h.update(head)
    return h.hexdigest()[:16]


def extract_zip_to_cache(uploaded_file) -> Path:
    """
    Extract zip to a persistent cache directory under user's home.
    Returns path to extraction directory.
    """
    cache_root = Path.home() / ".apple_fitness_streamlit_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    fp = fingerprint_file(uploaded_file)
    out_dir = cache_root / f"health_{fp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "export.zip"

    # write zip to disk (streaming copy)
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        with open(zip_path, "wb") as f:
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, f)

    # extract if not extracted yet (presence of any xml is a good hint)
    marker = out_dir / ".extracted_ok"
    if not marker.exists():
        # clear any partial extraction (except zip itself)
        for p in out_dir.iterdir():
            if p.name in ("export.zip",):
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except Exception:
                    pass

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

        marker.write_text("ok", encoding="utf-8")

    return out_dir


def find_export_xml(extract_dir: Path) -> Path:
    """
    Recursively find export.xml anywhere under extract_dir.
    """
    matches = list(extract_dir.rglob("export.xml"))
    if not matches:
        raise FileNotFoundError("export.xml not found inside the uploaded zip.")
    # If multiple, pick the shortest path (usually correct)
    matches.sort(key=lambda p: len(str(p)))
    return matches[0]


# =========================
# Streaming XML parsing (iterparse)
# =========================
@st.cache_data(show_spinner=False)
def parse_export_xml_iter(xml_path_str: str, year: int):
    """
    Stream-parse export.xml to build df_wk (workouts) and df_ring (rings)
    filtered for the chosen year only.
    """
    xml_path = Path(xml_path_str)
    start_year = date(year, 1, 1)
    end_year = date(year, 12, 31)

    workouts = []
    rings = []

    # NOTE: Use 'end' events so elem.attrib is complete
    context = ET.iterparse(str(xml_path), events=("end",))

    for event, elem in context:
        tag = elem.tag

        # ---- Workout ----
        if tag == "Workout":
            a = elem.attrib
            start_str = a.get("startDate")
            if start_str:
                start_dt = pd.to_datetime(start_str, errors="coerce")
                if not pd.isna(start_dt):
                    d = start_dt.date()
                    if start_year <= d <= end_year:
                        t = a.get("workoutActivityType")
                        workouts.append(
                            {
                                "type": t,
                                "type_clean": clean_type_name(t),
                                "duration": float(a.get("duration") or 0.0),
                                "energy": float(a.get("totalEnergyBurned") or 0.0),
                                "distance": float(a.get("totalDistance") or 0.0),
                                "start": start_dt,
                                "date": d,
                            }
                        )

            # critical: free memory
            elem.clear()

        # ---- ActivitySummary ----
        elif tag == "ActivitySummary":
            a = elem.attrib
            date_str = a.get("dateComponents")
            if date_str:
                dts = pd.to_datetime(date_str, errors="coerce")
                if not pd.isna(dts):
                    d = dts.date()
                    if start_year <= d <= end_year:
                        rings.append(
                            {
                                "date": d,
                                "move": float(a.get("activeEnergyBurned") or 0.0),
                                "move_goal": float(a.get("activeEnergyBurnedGoal") or 0.0),
                                "exercise": float(a.get("appleExerciseTime") or 0.0),
                                "exercise_goal": float(a.get("appleExerciseTimeGoal") or 0.0),
                                "stand": float(a.get("appleStandHours") or 0.0),
                                "stand_goal": float(a.get("appleStandHoursGoal") or 0.0),
                            }
                        )

            elem.clear()

        else:
            # For everything else, still clear to reduce memory
            elem.clear()

    df_wk = pd.DataFrame(workouts)
    df_ring = pd.DataFrame(rings)

    if not df_ring.empty:
        df_ring["move_closed"] = df_ring["move"] >= df_ring["move_goal"]
        df_ring["exercise_closed"] = df_ring["exercise"] >= df_ring["exercise_goal"]
        df_ring["stand_closed"] = df_ring["stand"] >= df_ring["stand_goal"]
        df_ring["rings_closed_count"] = (
            df_ring["move_closed"].astype(int)
            + df_ring["exercise_closed"].astype(int)
            + df_ring["stand_closed"].astype(int)
        )
        df_ring["rings_closed_ratio"] = df_ring["rings_closed_count"] / 3.0
        df_ring["all_closed"] = df_ring["rings_closed_count"] == 3

    return df_wk, df_ring


# =========================
# Stats / analytics
# =========================
def compute_activity_stats(df_wk: pd.DataFrame) -> pd.DataFrame:
    if df_wk.empty:
        return pd.DataFrame(columns=["count", "total_duration", "total_energy", "avg_duration"])

    stats = df_wk.groupby("type_clean").agg(
        count=("type_clean", "count"),
        total_duration=("duration", "sum"),
        total_energy=("energy", "sum"),
    )
    stats["avg_duration"] = stats["total_duration"] / stats["count"]
    stats = stats.sort_values("count", ascending=False)
    return stats


def compute_hardcore_day(df_wk: pd.DataFrame, df_ring: pd.DataFrame):
    """
    Hardcore Day = day with max total workout duration.
    kcal = sum(workout energy) fallback to rings move kcal if 0/missing.
    """
    if df_wk.empty:
        return None, None, None, 0.0, "N/A"

    daily = (
        df_wk.groupby("date")
        .agg(
            total_duration=("duration", "sum"),
            workout_kcal=("energy", "sum"),
            workout_count=("duration", "count"),
        )
        .sort_values("total_duration", ascending=False)
    )

    hardcore_date = daily.index[0]
    row = daily.iloc[0]

    kcal = float(row["workout_kcal"])
    source = "Workout kcal"
    if kcal <= 0:
        source = "Move kcal (rings)"
        kcal = 0.0
        if not df_ring.empty:
            rr = df_ring[df_ring["date"] == hardcore_date]
            if not rr.empty:
                kcal = float(rr["move"].iloc[0])

    per_type = (
        df_wk[df_wk["date"] == hardcore_date]
        .groupby("type_clean")["duration"]
        .sum()
        .sort_values(ascending=False)
    )

    return hardcore_date, row, per_type, kcal, source


def longest_streak_from_dates(dates) -> dict:
    """
    dates: iterable of datetime.date / datetime64 / str
    return: {length, start, end}
    """
    s = pd.to_datetime(pd.Series(list(dates))).dt.normalize().dropna().drop_duplicates().sort_values()
    if s.empty:
        return {"length": 0, "start": None, "end": None}

    day = s.dt.floor("D")
    # consecutive days → diff==1 day
    grp = (day.diff().dt.days.ne(1)).cumsum()
    streak_len = day.groupby(grp).size()
    best_grp = streak_len.idxmax()
    best_len = int(streak_len.loc[best_grp])

    best_days = day[grp == best_grp]
    return {
        "length": best_len,
        "start": best_days.iloc[0].date(),
        "end": best_days.iloc[-1].date(),
    }

# =========================
# Plots
# =========================
def plot_concentric_rings(year: int, df_wk: pd.DataFrame, df_ring: pd.DataFrame) -> plt.Figure:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    total_days_year = len(pd.date_range(start, end))

    workout_days = int(df_wk["date"].nunique()) if not df_wk.empty else 0
    closed_days = int(df_ring["all_closed"].sum()) if not df_ring.empty else 0

    workout_days = min(workout_days, total_days_year)
    closed_days = min(closed_days, workout_days)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    ax.axis("equal")

    outer_gray = "#D8D8D8"
    apple_green = "#34C759"
    apple_orange = "#FF9500"
    inner_bg = "#F7F7F7"

    ax.pie(
        [total_days_year],
        radius=1.05,
        colors=[outer_gray],
        startangle=90,
        wedgeprops=dict(width=0.25, edgecolor="white"),
    )
    ax.pie(
        [workout_days, total_days_year - workout_days],
        radius=0.75,
        colors=[apple_green, inner_bg],
        startangle=90,
        wedgeprops=dict(width=0.25, edgecolor="white"),
    )
    ax.pie(
        [closed_days, total_days_year - closed_days],
        radius=0.45,
        colors=[apple_orange, inner_bg],
        startangle=90,
        wedgeprops=dict(width=0.25, edgecolor="white"),
    )

    ax.text(
        0.5, 0.865,
        f"{total_days_year} days total",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=12, color="black", fontweight="normal",
    )

    label_style = dict(fontsize=12, color="black", fontweight="normal")
    angle = np.deg2rad(90)
    shift = 0.45

    r_w = 1.05
    ax.text(
        r_w * np.cos(angle),
        r_w * np.sin(angle) - shift,
        f"{workout_days} workout days\n({workout_days/total_days_year:.1%})",
        ha="center", va="center",
        **label_style,
    )

    r_c = 0.75
    ax.text(
        r_c * np.cos(angle),
        r_c * np.sin(angle) - shift,
        f"{closed_days} days all rings closed\n({closed_days/total_days_year:.1%})",
        ha="center", va="center",
        **label_style,
    )

    fig.tight_layout()
    return fig


def plot_category_pie(df_wk: pd.DataFrame, year: int):
    if df_wk.empty:
        return None

    aerobic_types = {
        "Running", "Walking", "Cycling", "Swimming", "Elliptical", "Rowing",
        "StairClimbing", "Hiking", "CardioDance"
    }
    anaerobic_types = {
        "FunctionalStrengthTraining", "TraditionalStrengthTraining",
        "CoreTraining", "CrossTraining", "HighIntensityIntervalTraining"
    }
    sports_types = {
        "Tennis", "Badminton", "Pickleball", "Basketball", "Soccer", "TableTennis", "Bowling"
    }

    def map_category(t: str) -> str:
        if t in aerobic_types:
            return "Aerobic"
        if t in anaerobic_types:
            return "Anaerobic"
        if t in sports_types:
            return "Sports"
        return "Other"

    df = df_wk.copy()
    df["category"] = df["type_clean"].apply(map_category)

    cat = df.groupby("category").agg(
        total_duration=("duration", "sum"),
        workout_count=("category", "count"),
        total_energy=("energy", "sum"),
    ).sort_values("total_duration", ascending=False)

    color_map = {
        "Aerobic": "#34C759",
        "Anaerobic": "#FF9500",
        "Sports": "#007AFF",
        "Other": "#D1D1D6",
    }
    labels = cat.index.tolist()
    sizes = cat["total_duration"].values
    colors = [color_map.get(x, "#D1D1D6") for x in labels]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")

    ax.legend(
        wedges,
        labels,
        title="Workout Category",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_ring_calendar_heatmap(year: int, df_ring: pd.DataFrame):
    if df_ring.empty:
        return None

    start_year = date(year, 1, 1)
    end_year = date(year, 12, 31)

    all_dates = pd.date_range(start_year, end_year)
    cal = pd.DataFrame({"date": all_dates})
    cal["date_only"] = cal["date"].dt.date

    df_small = df_ring[["date", "rings_closed_ratio"]].copy()
    cal = cal.merge(df_small, left_on="date_only", right_on="date", how="left")
    cal["rings_closed_ratio"] = cal["rings_closed_ratio"].fillna(0.0)

    first_monday = start_year
    while first_monday.weekday() != 0:
        first_monday = first_monday.replace(day=first_monday.day + 1)

    last_sunday = end_year
    while last_sunday.weekday() != 6:
        last_sunday = last_sunday.replace(day=last_sunday.day - 1)

    all_full = pd.date_range(first_monday, last_sunday, freq="D")
    n_weeks = int(len(all_full) / 7)

    mat = np.full((7, n_weeks), np.nan)
    ratio_map = cal.set_index("date_only")["rings_closed_ratio"].to_dict()

    for idx, d in enumerate(all_full):
        week_idx = idx // 7
        weekday = d.weekday()
        mat[weekday, week_idx] = ratio_map.get(d.date(), np.nan)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    month_pos, month_lbl = [], []
    cur = None
    for w in range(n_weeks):
        ws = first_monday + pd.Timedelta(days=w * 7)
        if ws.month != cur:
            cur = ws.month
            month_pos.append(w)
            month_lbl.append(ws.strftime("%b"))

    ax.set_xticks(month_pos)
    ax.set_xticklabels(month_lbl)

    fig.colorbar(im, ax=ax, label="Ring completion ratio (0–1)")
    fig.tight_layout()
    return fig


def plot_yearly_daily_workout_heatmap(year: int, df_wk: pd.DataFrame):
    if df_wk.empty:
        return None

    start_year = date(year, 1, 1)
    end_year = date(year, 12, 31)

    daily = df_wk.groupby("date")["duration"].sum()

    all_days = pd.date_range(start_year, end_year)
    n_weeks = (len(all_days) + all_days[0].weekday() + 6) // 7

    heatmap = np.full((7, n_weeks), np.nan)
    shift = all_days[0].weekday()

    for d in all_days:
        offset = (d - all_days[0]).days + shift
        week = offset // 7
        weekday = d.weekday()
        heatmap[weekday, week] = float(daily.get(d.date(), 0.0))

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    month_pos, month_lbl = [], []
    for m in range(1, 13):
        ms = date(year, m, 1)
        offset = (pd.Timestamp(ms) - all_days[0]).days + shift
        w = max(0, offset // 7)
        month_pos.append(w)
        month_lbl.append(pd.Timestamp(ms).strftime("%b"))

    ax.set_xticks(month_pos)
    ax.set_xticklabels(month_lbl)

    fig.colorbar(im, ax=ax, label="Workout duration (min)")
    fig.tight_layout()
    return fig


def plot_monthly_workouts(year: int, df_wk: pd.DataFrame):
    if df_wk.empty:
        return None

    df = df_wk.copy()
    df["month"] = df["start"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("month").agg(
        workout_count=("type_clean", "count"),
        total_duration=("duration", "sum"),
    )

    fig, ax1 = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)

    ax1.bar(monthly.index, monthly["workout_count"], alpha=0.6, color="#2196F3", label="Workout Count")
    ax1.set_ylabel("Workout Count", fontsize=12)
    ax1.set_xlabel("Month", fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["total_duration"], marker="o", color="#E91E63", linewidth=2, label="Total Duration (min)")
    ax2.set_ylabel("Total Duration (min)", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    return fig

def plot_weekday_workouts(year: int, df_wk: pd.DataFrame):
    # Ensure required columns
    df_wk["month"] = df_wk["start"].dt.month
    df_wk["weekday"] = df_wk["start"].dt.weekday  # 0 = Mon, 6 = Sun

    # Monthly × weekday workout counts
    monthly_weekday = (
        df_wk
        .groupby(["month", "weekday"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    WEEKDAY_COLORS = [
        "#C7D9F2",  # Mon - very light blue
        "#A9C4EA",  # Tue
        "#7FA8E0",  # Wed
        "#4F8AD8",  # Thu
        "#1F6FD2",  # Fri - deep blue
        "#FFB703",  # Sat - orange
        "#FB8500",  # Sun - deeper orange
    ]

    fig, ax = plt.subplots(figsize=(12, 5.3))
    months = monthly_weekday.index.tolist()
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    ax.stackplot(
        months,
        monthly_weekday.T.values,
        labels=weekday_labels,
        colors=WEEKDAY_COLORS,
        alpha=0.9
    )

    # X-axis: month labels
    ax.set_xticks(months)
    ax.set_xticklabels(
        [date(year, m, 1).strftime("%b") for m in months]
    )
    ax.set_ylabel("Workout Count")
    # Legend
    ax.legend(
        loc="best",
        ncol=4,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    return plt

# =========================
# Sidebar inputs
# =========================
available_years = list(range(2015, date.today().year + 1))
with st.sidebar:
    st.header("Inputs")
    YEAR = st.sidebar.selectbox( "Year",options=available_years,index=len(available_years) - 1 )
    uploaded_zip = st.file_uploader("Upload Apple Health export.zip", type=["zip"])
    st.divider()
    st.header("Sections")
    show_summary = st.checkbox("Summary", value=True)
    show_concentric = st.checkbox("Workout & Ring Summary", value=True)
    show_hardcore = st.checkbox("🏆 Hardcore Day", value=True)
    show_workout_types = st.checkbox("Workout Types", value=True)
    show_category_pie = st.checkbox("Workout Categories", value=True)
    show_ring_calendar = st.checkbox("Ring Completion Calendar", value=True)
    show_workout_calendar = st.checkbox("Daily Workout Calendar", value=True)
    show_monthly = st.checkbox("Monthly Workouts", value=True)
    show_weekday = st.checkbox("Workouts by Weekday", value=True)
    st.divider()
    st.subheader("Workout Type Threshold")
    workout_type_threshold = st.slider(
        "Show workout types with at least N sessions",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )
    st.caption("Parsing is done locally. Large exports are handled via streaming XML parse (iterparse).")
# =========================
# Main
# =========================
st.markdown(f"<h1 style='text-align:center; margin: 8px 0 22px 0;'>Fitness Dashboard {int(YEAR)}</h1>", unsafe_allow_html=True)

if uploaded_zip is None:
    st.info("Upload your **export.zip** (Apple Health export) to begin.")
    st.stop()

# Extract + locate export.xml
with st.spinner("Extracting export.zip and locating export.xml ..."):
    extract_dir = extract_zip_to_cache(uploaded_zip)
    xml_path = find_export_xml(extract_dir)

st.success(f"Found export.xml: {xml_path}")

# Parse (streaming)
with st.spinner("Parsing export.xml (streaming). This may take a bit for large exports..."):
    df_wk, df_ring = parse_export_xml_iter(str(xml_path), int(YEAR))

# Compute stats
activity_stats = compute_activity_stats(df_wk)

# =========================
# 1) Summary
# =========================
if show_summary:
    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    if df_wk.empty:
        c1.metric("Total workouts", "0")
        c2.metric("Workout days", "0")
        c3.metric("Total duration (hours)", "0.0")
    else:
        total_workouts = int(len(df_wk))
        workout_days = int(df_wk["date"].nunique())
        total_duration_min = float(df_wk["duration"].sum())
        c1.metric("Total workouts", f"{total_workouts}")
        c2.metric("Workout days", f"{workout_days}")
        c3.metric("Total duration (hours)", f"{total_duration_min/60:.1f}")

    if df_ring.empty:
        c4.metric("All rings closed days", "N/A")
    else:
        c4.metric("All rings closed days", f"{int(df_ring['all_closed'].sum())}")

    # Longest workout streak (consecutive workout days)
    wk_streak = longest_streak_from_dates(df_wk["date"]) if not df_wk.empty else {"length": 0, "start": None, "end": None}
    # Longest all-rings-closed streak
    if df_ring is not None and (not df_ring.empty) and ("all_closed" in df_ring.columns):
        ring_days = df_ring.loc[df_ring["all_closed"], "date"]
        ring_streak = longest_streak_from_dates(ring_days)
    else:
        ring_streak = {"length": 0, "start": None, "end": None}
    col1, col2 = st.columns(2)
    with col1:
        if wk_streak["length"] > 0:
            st.metric("🔥 Longest workout streak days", f"{wk_streak['length']}",
                    help=f"{wk_streak['start']} → {wk_streak['end']}")
        else:
            st.metric("🔥 Longest workout streak days", "0 days")

    with col2:
        if ring_streak["length"] > 0:
            st.metric("💪 Longest all-rings-closed streak days", f"{ring_streak['length']}",
                    help=f"{ring_streak['start']} → {ring_streak['end']}")
        else:
            st.metric("💪 Longest all-rings-closed streak days", "0 days")

    with st.expander("See workout type table"):
        if activity_stats.empty:
            st.write("No workouts found for this year.")
        else:
            show = activity_stats.copy()
            show.index = [pretty_name(x) for x in show.index]
            st.dataframe(show)
        st.caption(
        "ℹ️ Calorie data may be missing (0) for some workout types because "
        "Apple Health does not provide energy estimates for those activities."
        )

# =========================
# 2) Concentric rings
# =========================
if show_concentric:
    st.subheader("Workout & Ring Summary")
    fig = plot_concentric_rings(int(YEAR), df_wk, df_ring)
    st.pyplot(fig, clear_figure=True)

# =========================
# 3) Hardcore Day
# =========================
if show_hardcore:
    st.subheader("🏆 Hardcore Day")

    hardcore_date, hardcore_row, hardcore_workouts, kcal, kcal_source = compute_hardcore_day(df_wk, df_ring)
    if hardcore_date is None:
        st.info("No workouts found for this year.")
    else:
        st.markdown(
            f"""
    <div class="hardcore-card">
    <div class="hardcore-meta">
        <div><b>{hardcore_date}</b></div>
        <div>⏱️ {hardcore_row["total_duration"]:.0f} min · 🔥 {kcal:.0f} kcal</div>
        <div class="kcal-note">kcal source: {kcal_source}</div>
    </div>
    <ul class="emoji-list">
    """
            + "\n".join(
                [
                    f"<li><span class='icon'>{EMOJI_MAP.get(k,'•')}</span><span class='txt'>{pretty_name(k)}: {v:.0f} min</span></li>"
                    for k, v in hardcore_workouts.items()
                ]
            )
            + """
    </ul>
    </div>
    """,
            unsafe_allow_html=True,
        )

# =========================
# 4) Workout Types (>N sessions) — emoji list (Streamlit UI)
# =========================
if show_workout_types:
    st.subheader(f"Workout Types (>{workout_type_threshold} sessions)")

    if activity_stats.empty:
        st.write("No workout data.")
    else:
        df_overN = activity_stats[activity_stats["count"] > workout_type_threshold].copy()
        if df_overN.empty:
            st.write(f"No workout type exceeds {workout_type_threshold} sessions this year.")
        else:
            st.markdown(
                "<ul class='emoji-list'>"
                + "\n".join(
                    [
                        f"<li><span class='icon'>{EMOJI_MAP.get(k,'•')}</span>"
                        f"<span class='txt'>{pretty_name(k)} × {int(row['count'])} sessions ({row['avg_duration']:.1f} min/session)</span></li>"
                        for k, row in df_overN.iterrows()
                    ]
                )
                + "</ul>",
                unsafe_allow_html=True,
            )

# =========================
# 5) All Workout Types — emoji list
# =========================
if show_workout_types:
    st.subheader("All Workout Types")

    if activity_stats.empty:
        st.write("No workout data.")
    else:
        st.markdown(
            "<ul class='emoji-list'>"
            + "\n".join(
                [
                    f"<li><span class='icon'>{EMOJI_MAP.get(k,'•')}</span>"
                    f"<span class='txt'>{pretty_name(k)} × {int(row['count'])} sessions ({row['avg_duration']:.1f} min/session)</span></li>"
                    for k, row in activity_stats.iterrows()
                ]
            )
            + "</ul>",
            unsafe_allow_html=True,
        )

# =========================
# 6) Category pie
# =========================
if show_category_pie:
    st.subheader("Workout Categories (by Duration)")
    fig = plot_category_pie(df_wk, int(YEAR))
    if fig is None:
        st.write("No workouts.")
    else:
        st.pyplot(fig, clear_figure=True)

# =========================
# 7) Ring completion calendar
# =========================
if show_ring_calendar:
    st.subheader("Ring Completion Calendar")
    fig = plot_ring_calendar_heatmap(int(YEAR), df_ring)
    if fig is None:
        st.write("No ring data for this year.")
    else:
        st.pyplot(fig, clear_figure=True)

# =========================
# 8) Daily workout calendar heatmap
# =========================
if show_workout_calendar:
    st.subheader("Daily Workout Calendar")
    fig = plot_yearly_daily_workout_heatmap(int(YEAR), df_wk)
    if fig is None:
        st.write("No workouts.")
    else:
        st.pyplot(fig, clear_figure=True)

# =========================
# 9) Monthly workouts
# =========================
if show_monthly:
    st.subheader("Monthly Workouts (Count & Duration)")
    fig = plot_monthly_workouts(int(YEAR), df_wk)
    if fig is None:
        st.write("No workouts.")
    else:
        st.pyplot(fig, clear_figure=True)

# =========================
# 10) Workouts by weekday
# =========================
if show_weekday:
    st.subheader("Workouts by Weekday")
    fig = plot_weekday_workouts(int(YEAR), df_wk)
    if fig is None:
        st.write("No workouts.")
    else:
        st.pyplot(fig, clear_figure=True)

st.caption("Done. Tip: if you upload the same zip again, parsing is cached.")
