import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE = ROOT / "checkpoints"
ORIGINAL = Path.home() / "OneDrive" / "Desktop" / "SwarmRL" / "checkpoints"

FILES = {
    "Static World - Day13": ACTIVE / "day17_static_day13.csv",
    "Static World - Day16": ACTIVE / "day17_static_day16.csv",
    "Standard Benchmark - Day13": ORIGINAL / "day17_day13_benchmark.csv",
    "Standard Benchmark - Day16": ORIGINAL / "day17_day16_benchmark.csv",
    "Validation - Day13": ORIGINAL / "day17_day13_validation.csv",
    "Validation - Day16": ORIGINAL / "day17_day16_validation.csv",
    "Wind - Day13": ORIGINAL / "day17_day13_wind.csv",
    "Wind - Day16": ORIGINAL / "day17_day16_wind.csv",
}

REQUIRED = [
    "episode", "reward_raw", "coverage_pct",
    "drone_drone_collisions", "drone_obstacle_collisions",
    "min_pair_distance_m", "steps_below_preferred_separation",
]

st.set_page_config(
    page_title="SwarmRL Dashboard",
    page_icon="🚁",
    layout="wide",
)

st.title("SwarmRL Performance Dashboard")
st.caption("Day17 | Day13 Baseline vs Day16 Separation Reward")


@st.cache_data
def load_data():
    frames = []
    missing = []

    for label, path in FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue

        df = pd.read_csv(path)
        absent = [c for c in REQUIRED if c not in df.columns]

        if absent:
            missing.append(f"{path}: missing columns {absent}")
            continue

        df["evaluation"] = label
        df["checkpoint"] = "Day13" if "Day13" in label else "Day16"
        frames.append(df)

    if not frames:
        return pd.DataFrame(), missing

    return pd.concat(frames, ignore_index=True), missing


data, missing = load_data()

if data.empty:
    st.error("No valid Day17 evaluation CSV files found.")
    st.write(missing)
    st.stop()

st.sidebar.header("Dashboard Controls")

condition = st.sidebar.selectbox(
    "Evaluation condition",
    data["evaluation"].drop_duplicates().tolist(),
)

selected = data[data["evaluation"] == condition].copy()
selected = selected.sort_values("episode")

st.sidebar.write(f"Loaded files: {len(FILES) - len(missing)} / {len(FILES)}")
st.sidebar.write(f"Total episodes: {len(data)}")

if missing:
    with st.sidebar.expander("Missing files or columns"):
        for item in missing:
            st.write(item)

st.header(condition)

c1, c2, c3 = st.columns(3)
c1.metric("Mean Coverage", f"{selected.coverage_pct.mean():.3f}%")
c2.metric("Mean Reward", f"{selected.reward_raw.mean():.3f}")
c3.metric("Episodes", len(selected))

c4, c5, c6 = st.columns(3)
c4.metric("Drone Collisions", int(selected.drone_drone_collisions.sum()))
c5.metric("Obstacle Collisions", int(selected.drone_obstacle_collisions.sum()))
c6.metric("Minimum Pair Distance", f"{selected.min_pair_distance_m.min():.3f} m")

st.metric(
    "Steps Below 4 m",
    int(selected.steps_below_preferred_separation.sum()),
)

st.divider()
st.subheader("Episode Performance")

tab1, tab2, tab3 = st.tabs(["Coverage", "Reward", "Safety"])

with tab1:
    st.line_chart(selected.set_index("episode")[["coverage_pct"]])

with tab2:
    st.line_chart(selected.set_index("episode")[["reward_raw"]])

with tab3:
    st.line_chart(
        selected.set_index("episode")[
            ["min_pair_distance_m", "steps_below_preferred_separation"]
        ]
    )

st.divider()
st.header("Day13 vs Day16 — All Conditions")

summary = data.groupby("evaluation").agg(
    Episodes=("episode", "count"),
    Mean_Coverage=("coverage_pct", "mean"),
    Mean_Reward=("reward_raw", "mean"),
    Drone_Collisions=("drone_drone_collisions", "sum"),
    Obstacle_Collisions=("drone_obstacle_collisions", "sum"),
    Minimum_Distance=("min_pair_distance_m", "min"),
    Steps_Below_4m=("steps_below_preferred_separation", "sum"),
).reset_index()

st.dataframe(summary.round(3), width="stretch", hide_index=True)

st.subheader("Coverage Comparison")
st.bar_chart(summary.set_index("evaluation")[["Mean_Coverage"]])

st.divider()
st.subheader("Episode-Level Data")

columns = [
    "episode", "reward_raw", "coverage_pct",
    "drone_drone_collisions", "drone_obstacle_collisions",
    "min_pair_distance_m", "steps_below_preferred_separation",
]

st.dataframe(selected[columns].round(3), width="stretch", hide_index=True)

st.download_button(
    "Download Selected Evaluation CSV",
    selected.to_csv(index=False).encode("utf-8"),
    file_name="swarmrl_selected_evaluation.csv",
    mime="text/csv",
)

st.caption("Read-only dashboard. No training or checkpoint modification.")

