from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitwall import serving
from pitwall.features.race_sim_params import RaceInputs, from_tables, to_tables

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_race_inputs_round_trip_through_tables():
    setup = serving.demo_setup()
    inputs = [
        RaceInputs(
            setup, [c.driver_id for c in setup.cars], np.arange(1, 21), np.arange(1, 21), "bahrain"
        )
    ]
    races, cars = to_tables(inputs, 2025)
    posterior = pd.DataFrame(
        [("bahrain", k, d, 0.05, 0.03) for k in ("SOFT", "MEDIUM", "HARD") for d in range(10)],
        columns=["circuit_id", "compound", "draw", "beta_s_per_lap", "fuel_effect_s_per_lap"],
    )
    rebuilt = from_tables(races.iloc[0], cars, posterior)
    assert [c.strategy.stints for c in rebuilt.cars] == [c.strategy.stints for c in setup.cars]
    assert rebuilt.params.race_laps == setup.params.race_laps
    assert len(rebuilt.params.fuel_draws) == 10


def test_app_runs_in_demo_mode(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PITWALL_LAKEHOUSE_ROOT", str(tmp_path))
    serving.get_settings.cache_clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        assert not at.exception
        assert any("demo race" in w.value for w in at.warning)
        assert len(at.metric) >= 3
    finally:
        serving.get_settings.cache_clear()
