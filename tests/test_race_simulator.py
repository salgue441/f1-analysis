import numpy as np
import pytest

from pitwall.simulation.race_simulator import RaceParams, Stint, Strategy, simulate_strategy

PARAMS = RaceParams(
    race_laps=50,
    base_lap_s=90.0,
    fuel_effect_s=0.0,
    deg_samples={"SOFT": np.array([0.10]), "HARD": np.array([0.02])},
    pit_loss_mean_s=20.0,
    pit_loss_sd_s=0.0,
    sc_hazard_per_lap=0.0,
)


def test_deterministic_case_matches_hand_calculation():
    one_stop = Strategy("S-H", (Stint("SOFT", 20), Stint("HARD", 30)))
    times = simulate_strategy(one_stop, PARAMS, n_sims=10, rng=np.random.default_rng(0))
    expected = 90.0 * 50 + 0.10 * sum(range(1, 21)) + 0.02 * sum(range(1, 31)) + 20.0
    assert np.allclose(times, expected)


def test_lap_count_must_match_race():
    with pytest.raises(ValueError):
        simulate_strategy(
            Strategy("bad", (Stint("SOFT", 10),)), PARAMS, 1, np.random.default_rng(0)
        )
