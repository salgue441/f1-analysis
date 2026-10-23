"""Pitwall strategy explorer: pick a race, try strategies, see the uncertainty.

Run with `make app` (http://localhost:8501). Reads the lakehouse through delta-rs, so it
needs no Spark; without data it falls back to a clearly labelled demo race.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitwall import serving
from pitwall.simulation.field import points_for, simulate_field
from pitwall.simulation.race_simulator import Stint, Strategy
from pitwall.simulation.strategies import evaluate_for_driver

COMPOUND_COLORS = {
    "SOFT": "#d62728",
    "MEDIUM": "#e6b800",
    "HARD": "#9a9a9a",
    "INTERMEDIATE": "#2ca02c",
    "WET": "#1f77b4",
    "UNKNOWN": "#555555",
}

st.set_page_config(page_title="Pitwall", page_icon="🏁", layout="wide")
st.title("Pitwall: F1 strategy explorer")


@st.cache_data(show_spinner=False)
def _race_options():
    return serving.race_options()


@st.cache_resource(show_spinner=False)
def _setup(race_id: str):
    return serving.demo_setup() if race_id == "demo" else serving.load_setup(race_id)


@st.cache_data(show_spinner=False)
def _laps(race_id: str):
    return None if race_id == "demo" else serving.laps_for_race(race_id)


options = _race_options()
with st.sidebar:
    st.header("Race")
    if options is None or options.empty:
        st.warning(
            "No simulator inputs found yet: showing a demo race with made-up numbers. "
            "Run the pipeline and `pitwall research rq3` to explore real 2025 races."
        )
        race_id = "demo"
    else:
        labels = {
            r.race_id: f"{r.race_id.replace('_', ' R')} · {r.event_name}"
            for r in options.itertuples(index=False)
        }
        race_id = st.selectbox("Race", list(labels), format_func=labels.get, index=len(labels) - 1)
    n_sims = st.select_slider("Simulations", [500, 1000, 2000, 5000], value=2000)
    seed = st.number_input("Random seed", value=42, step=1)

setup = _setup(race_id)
if setup is None:
    st.error("Simulator inputs are incomplete for this race (RQ1 posterior missing?).")
    st.stop()

drivers = [c.driver_id for c in setup.cars]
explore, race_tab, findings_tab = st.tabs(["Strategy explorer", "Race laps", "Findings"])

with explore:
    left, right = st.columns([1, 2])
    with left:
        driver = st.selectbox("Driver", drivers)
        idx = drivers.index(driver)
        actual = setup.cars[idx].strategy
        st.caption(
            "Actual strategy: " + " → ".join(f"{s.compound} ({s.laps})" for s in actual.stints)
        )
        st.subheader("Try a strategy")
        stops = st.radio("Stops", [1, 2], horizontal=True)
        laps = setup.params.race_laps
        compounds = ["SOFT", "MEDIUM", "HARD"]
        c1 = st.selectbox("Stint 1", compounds, index=1)
        p1 = st.slider("Pit lap 1", 5, laps - 5, min(20, laps - 5))
        c2 = st.selectbox("Stint 2", compounds, index=2)
        stints = [Stint(c1, p1)]
        if stops == 2:
            p2 = st.slider("Pit lap 2", p1 + 3, laps - 3, min(p1 + 18, laps - 3))
            c3 = st.selectbox("Stint 3", compounds, index=1)
            stints += [Stint(c2, p2 - p1), Stint(c3, laps - p2)]
        else:
            stints.append(Stint(c2, laps - p1))
        custom = Strategy("custom", tuple(stints))
        legal = len({s.compound for s in custom.stints}) >= 2
        if not legal:
            st.warning("A dry race must use at least two compounds.")

    with right:
        if legal:
            with st.spinner(f"Simulating {n_sims:,} races..."):
                table = evaluate_for_driver(
                    list(setup.cars), idx, [custom], setup.params, n_sims=n_sims, seed=int(seed)
                )
            act, cus = table.iloc[0], table.iloc[1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected points (actual)", f"{act.expected_points:.2f}")
            m2.metric(
                "Expected points (yours)",
                f"{cus.expected_points:.2f}",
                f"{cus.points_gain_vs_actual:+.2f} ± {2 * cus.points_gain_se:.2f}",
            )
            m3.metric(
                "P(yours scores more)",
                f"{cus.p_more_points_than_actual:.0%}",
                f"P(less) {cus.p_fewer_points_than_actual:.0%}",
                delta_color="off",
            )

            rows = []
            for name, strategy in [("actual", actual), ("yours", custom)]:
                cars = list(setup.cars)
                cars[idx] = replace(cars[idx], strategy=strategy)
                pos = simulate_field(cars, setup.params, n_sims, np.random.default_rng(int(seed)))
                counts = np.bincount(pos[:, idx], minlength=len(cars) + 1)[1:] / n_sims
                rows += [
                    {"strategy": name, "position": p + 1, "probability": v}
                    for p, v in enumerate(counts)
                ]
            dist = pd.DataFrame(rows)
            fig = px.bar(
                dist,
                x="position",
                y="probability",
                color="strategy",
                barmode="group",
                color_discrete_map={"actual": "#4c72b0", "yours": "#dd8452"},
            )
            fig.update_layout(
                height=360,
                margin=dict(t=30, b=10),
                yaxis_tickformat=".0%",
                title=f"Finishing position of {driver} ({n_sims:,} simulated races)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Same random numbers for both strategies (safety cars, retirements, "
                "lap noise and tyre-wear draws), so the difference is the strategy, not luck. "
                "Degradation draws come from the RQ1 posterior for this circuit."
            )

    st.subheader("Simulator's view of the whole field (actual strategies)")
    pos = simulate_field(list(setup.cars), setup.params, n_sims, np.random.default_rng(int(seed)))
    pts = points_for(pos)
    field = pd.DataFrame(
        {
            "driver": drivers,
            "grid": [c.grid for c in setup.cars],
            "P(win)": (pos == 1).mean(axis=0),
            "P(podium)": (pos <= 3).mean(axis=0),
            "P(points)": (pos <= 10).mean(axis=0),
            "expected points": pts.mean(axis=0),
            "median finish": np.median(pos, axis=0),
        }
    )
    st.dataframe(
        field.style.format(
            {
                "P(win)": "{:.1%}",
                "P(podium)": "{:.1%}",
                "P(points)": "{:.1%}",
                "expected points": "{:.2f}",
                "median finish": "{:.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with race_tab:
    laps_df = _laps(race_id)
    if laps_df is None or laps_df.empty:
        st.info("Lap data appears here for real races once gold.fact_lap is built.")
    else:
        stints = laps_df.groupby(["driver_id", "stint_number", "compound"], as_index=False).agg(
            start=("lap_number", "min"), end=("lap_number", "max")
        )
        order = (
            laps_df[laps_df["lap_number"] == laps_df["lap_number"].max()]
            .sort_values("position_end_of_lap")["driver_id"]
            .tolist()
        )
        fig = go.Figure()
        for row in stints.itertuples(index=False):
            fig.add_trace(
                go.Bar(
                    x=[row.end - row.start + 1],
                    y=[row.driver_id],
                    base=row.start - 1,
                    orientation="h",
                    marker_color=COMPOUND_COLORS.get(row.compound),
                    name=row.compound,
                    showlegend=False,
                    hovertemplate=f"{row.compound}: laps {row.start}-{row.end}",
                )
            )
        fig.update_layout(
            barmode="overlay",
            height=560,
            xaxis_title="lap",
            yaxis=dict(categoryorder="array", categoryarray=order[::-1]),
            title="Stints (finishing order, top to bottom)",
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        picked = st.multiselect(
            "Lap times", sorted(laps_df["driver_id"].dropna().unique()), default=order[:3]
        )
        clean = laps_df[
            laps_df["driver_id"].isin(picked)
            & laps_df["is_green_flag"]
            & ~laps_df["is_pit_in_lap"]
            & ~laps_df["is_pit_out_lap"]
        ]
        fig = px.line(
            clean.sort_values("lap_number"),
            x="lap_number",
            y="lap_time_s",
            color="driver_id",
            markers=True,
        )
        fig.update_layout(height=380, yaxis_title="lap time (s)", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

with findings_tab:
    found = serving.findings()
    if not found:
        st.info("Findings appear after the research runs (`pitwall research rq1|rq2|rq3`).")
    if "rq1" in found:
        r = found["rq1"]
        st.subheader("RQ1 · How fast do tyres wear?")
        cols = st.columns(3)
        for col, (k, v) in zip(cols, r["beta_compound"].items(), strict=False):
            col.metric(
                f"{k} (s/lap per lap of age)",
                f"{v['mean']:.3f}",
                f"90% CI {v['lo']:.3f} – {v['hi']:.3f}",
                delta_color="off",
            )
        st.image(str(serving.REPORTS / "rq1" / "forest_plot.png"))
    if "rq2" in found:
        d = found["rq2"]["primary"]["dml"]
        st.subheader("RQ2 · Does the undercut work?")
        st.metric(
            "Effect of pitting first on P(ahead after both stop), DML",
            f"{d['effect']:+.1%}",
            f"95% CI {d['lo']:+.1%} to {d['hi']:+.1%}",
            delta_color="off",
        )
    if "rq3" in found:
        b = found["rq3"]["backtest"]
        st.subheader("RQ3 · What should teams have done?")
        c1, c2 = st.columns(2)
        c1.metric(
            "Backtest RPS (lower is better)",
            f"{b['rps_model']:.3f}",
            f"grid baseline {b['rps_grid_baseline']:.3f}",
            delta_color="off",
        )
        dec = found["rq3"]["decision"]
        c2.metric(
            "Expected points left on the table per driver-race",
            f"{dec['mean_gain_per_driver_race']:.2f}",
            f"95% CI {dec['mean_gain_ci'][0]:.2f} – {dec['mean_gain_ci'][1]:.2f}",
            delta_color="off",
        )
