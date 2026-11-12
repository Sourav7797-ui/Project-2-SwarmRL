
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_CHECKPOINTS = PROJECT_ROOT / "checkpoints"

ORIGINAL_CHECKPOINTS = (
    Path.home()
    / "OneDrive"
    / "Desktop"
    / "SwarmRL"
    / "checkpoints"
)

CSV_FILES = {
    "Static World — Day13": ACTIVE_CHECKPOINTS / "day17_static_day13.csv",
    "Static World — Day16": ACTIVE_CHECKPOINTS / "day17_static_day16.csv",
    "Standard Benchmark — Day13": ORIGINAL_CHECKPOINTS / "day17_day13_benchmark.csv",
    "Standard Benchmark — Day16": ORIGINAL_CHECKPOINTS / "day17_day16_benchmark.csv",
    "20-Episode Validation — Day13": ORIGINAL_CHECKPOINTS / "day17_day13_validation.csv",
    "20-Episode Validation — Day16": ORIGINAL_CHECKPOINTS / "day17_day16_validation.csv",
    "Wind Generalization — Day13": ORIGINAL_CHECKPOINTS / "day17_day13_wind.csv",
    "Wind Generalization — Day16": ORIGINAL_CHECKPOINTS / "day17_day16_wind.csv",
}

REQUIRED_COLUMNS = [
    "episode",
    "reward_raw",
    "coverage_pct",
    "drone_drone_collisions",
    "drone_obstacle_collisions",
    "min_pair_distance_m",
    "steps_below_preferred_separation",
]


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SwarmRL Performance Dashboard",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 SwarmRL Performance Dashboard")

st.caption(
    "Day17 benchmark analysis | "
    "Day13 baseline vs Day16 separation reward"
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_data
def load_all_results():
    frames = []
    missing = []
    invalid = []

    for label, path in CSV_FILES.items():

        if not path.exists():
            missing.append(str(path))
            continue

        try:
            data = load_csv(str(path))

        except Exception as exc:
            invalid.append(f"{path}: {exc}")
            continue

        absent = [
            column
            for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if absent:
            invalid.append(
                f"{path.name}: missing columns {', '.join(absent)}"
            )
            continue

        data = data.copy()

        data["evaluation"] = label

        data["checkpoint"] = (
            "Day13" if "Day13" in label else "Day16"
        )

        frames.append(data)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, missing, invalid


results, missing_files, invalid_files = load_all_results()

if results.empty:
    st.error("No valid evaluation CSV files were found.")

    st.info(
        "Check that your Day17 CSV files exist in the configured folders."
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Dashboard Controls")

available_evaluations = (
    results["evaluation"].drop_duplicates().tolist()
)

selected_evaluation = st.sidebar.selectbox(
    "Select evaluation condition",
    available_evaluations,
)

selected_data = results[
    results["evaluation"] == selected_evaluation
].copy()

selected_data = selected_data.sort_values("episode")

st.sidebar.divider()

st.sidebar.write("**Data sources**")

st.sidebar.write(
    f"Loaded CSV files: {len(available_evaluations)} / {len(CSV_FILES)}"
)

st.sidebar.write(
    f"Total episodes: {len(results)}"
)

if missing_files:
    with st.sidebar.expander("Missing CSV files"):
        for filename in missing_files:
            st.write(filename)

if invalid_files:
    with st.sidebar.expander("Invalid CSV files"):
        for filename in invalid_files:
            st.write(filename)


# ---------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------

mean_coverage = selected_data["coverage_pct"].mean()

mean_reward = selected_data["reward_raw"].mean()

drone_collisions = selected_data[
    "drone_drone_collisions"
].sum()

obstacle_collisions = selected_data[
    "drone_obstacle_collisions"
].sum()

minimum_distance = selected_data[
    "min_pair_distance_m"
].min()

close_steps = selected_data[
    "steps_below_preferred_separation"
].sum()

st.subheader(selected_evaluation)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Coverage",
    f"{mean_coverage:.3f}%"
)

col2.metric(
    "Mean Episode Reward",
    f"{mean_reward:.3f}"
)

col3.metric(
    "Episodes",
    len(selected_data)
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Drone Collisions",
    int(drone_collisions)
)

col5.metric(
    "Obstacle Collisions",
    int(obstacle_collisions)
)

col6.metric(
    "Minimum Pair Distance",
    f"{minimum_distance:.3f} m"
)

st.metric(
    "Steps Below 4 m Preferred Separation",
    int(close_steps)
)


# ---------------------------------------------------------
# EPISODE CHARTS
# ---------------------------------------------------------

st.divider()

st.subheader("Episode-Level Performance")

chart_data = (
    selected_data
    .sort_values("episode")
    .set_index("episode")
)

tab1, tab2, tab3 = st.tabs(
    ["Coverage", "Episode Reward", "Safety"]
)

with tab1:

    st.line_chart(
        chart_data[["coverage_pct"]],
        y_label="Coverage (%)",
    )

with tab2:

    st.line_chart(
        chart_data[["reward_raw"]],
        y_label="Raw episode reward",
    )

with tab3:

    st.markdown("**Minimum Pair Distance**")

    st.line_chart(
        chart_data[["min_pair_distance_m"]],
        y_label="Distance (metres)",
    )

    st.markdown("**Steps Below Preferred Separation**")

    st.line_chart(
        chart_data[["steps_below_preferred_separation"]],
        y_label="Steps",
    )

    st.caption(
        "Close-proximity steps count steps with at least one "
        "drone pair below the 4 m preferred separation."
    )


# ---------------------------------------------------------
# DAY13 VS DAY16 COMPARISON
# ---------------------------------------------------------

st.divider()

st.subheader("Day13 vs Day16 Comparison")

comparison_rows = []

for evaluation_name in available_evaluations:

    subset = results[
        results["evaluation"] == evaluation_name
    ]

    comparison_rows.append({

        "Evaluation": evaluation_name,

        "Checkpoint": (
            "Day13"
            if "Day13" in evaluation_name
            else "Day16"
        ),

        "Episodes": len(subset),

        "Mean Coverage (%)": (
            subset["coverage_pct"].mean()
        ),

        "Mean Reward": (
            subset["reward_raw"].mean()
        ),

        "Drone Collisions": (
            subset["drone_drone_collisions"].sum()
        ),

        "Obstacle Collisions": (
            subset["drone_obstacle_collisions"].sum()
        ),

        "Minimum Distance (m)": (
            subset["min_pair_distance_m"].min()
        ),

        "Steps Below 4 m": (
            subset[
                "steps_below_preferred_separation"
            ].sum()
        ),
    })

comparison = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison.round(3),
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# COVERAGE COMPARISON
# ---------------------------------------------------------

st.subheader("Coverage Across Evaluation Conditions")

coverage_comparison = comparison.set_index(
    "Evaluation"
)[["Mean Coverage (%)"]]

st.bar_chart(
    coverage_comparison,
    y_label="Mean Coverage (%)",
)


# ---------------------------------------------------------
# DAY13 VS DAY16 COVERAGE BY CONDITION
# ---------------------------------------------------------

st.subheader("Day13 vs Day16 — Paired Coverage Comparison")

paired_coverage = (
    results.groupby(["evaluation", "checkpoint"])["coverage_pct"]
    .mean()
    .reset_index()
)

st.bar_chart(
    paired_coverage,
    x="evaluation",
    y="coverage_pct",
    color="checkpoint",
    y_label="Mean Coverage (%)",
)


# ---------------------------------------------------------
# RAW EPISODE DATA
# ---------------------------------------------------------

st.divider()

st.subheader("Episode Data")

display_columns = [
    "episode",
    "reward_raw",
    "coverage_pct",
    "drone_drone_collisions",
    "drone_obstacle_collisions",
    "min_pair_distance_m",
    "steps_below_preferred_separation",
]

st.dataframe(
    selected_data[display_columns].round(3),
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# CSV DOWNLOAD
# ---------------------------------------------------------

csv_download = selected_data.to_csv(
    index=False
).encode("utf-8")

safe_filename = (
    selected_evaluation
    .lower()
    .replace(" ", "_")
    .replace("—", "")
)

st.download_button(
    label="Download Selected Evaluation CSV",
    data=csv_download,
    file_name=f"{safe_filename}.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "SwarmRL Day18 Dashboard | "
    "Read-only evaluation data. "
    "No training or checkpoint modification is performed."
)
