import numpy as np
import pandas as pd
import pytest

from pitwall.simulation import scoring
from pitwall.simulation.field import Car, FieldParams, points_for, simulate_field
from pitwall.simulation.race_simulator import Stint, Strategy
from pitwall.simulation.strategies import (
    candidate_strategies,
    evaluate_for_driver,
    strategy_from_stints,
)

LAPS = 40


def _params(**kw):
    base = dict(
        race_laps=LAPS,
        base_lap_s=90.0,
        deg_draws={"SOFT": np.array([0.12]), "MEDIUM": np.array([0.06]), "HARD": np.array([0.03])},
        fuel_draws=np.array([0.03]),
        pit_loss_mean_s=20.0,
        pit_loss_sd_s=0.0,
        sc_hazard_per_lap=0.0,
        lap_noise_sd_s=0.0,
        dnf_hazard_per_lap=0.0,
    )
    return FieldParams(**{**base, **kw})


ONE_STOP = Strategy("MH-20", (Stint("MEDIUM", 20), Stint("HARD", 20)))


def test_faster_car_wins_without_noise():
    cars = [
        Car("slow", 1, 0.5, ONE_STOP),
        Car("fast", 2, 0.0, ONE_STOP),
    ]
    pos = simulate_field(cars, _params(pass_threshold_s=0.1), 50, np.random.default_rng(0))
    assert (pos[:, 1] == 1).all()


def test_overtaking_difficulty_holds_up_a_slightly_faster_car():
    cars = [Car("lead", 1, 0.2, ONE_STOP), Car("chaser", 2, 0.0, ONE_STOP)]
    hard_to_pass = _params(pass_threshold_s=3.0, pass_scale_s=0.1)
    pos = simulate_field(cars, hard_to_pass, 200, np.random.default_rng(0))
    assert (pos[:, 0] == 1).mean() > 0.9


def test_strategy_lap_count_validated():
    with pytest.raises(ValueError):
        simulate_field(
            [Car("a", 1, 0, Strategy("x", (Stint("SOFT", 5),)))],
            _params(),
            1,
            np.random.default_rng(0),
        )


def test_same_seed_same_result_and_retirements_rank_last():
    cars = [Car(f"d{i}", i + 1, 0.1 * i, ONE_STOP) for i in range(6)]
    params = _params(lap_noise_sd_s=0.5, sc_hazard_per_lap=0.05, dnf_hazard_per_lap=0.01)
    a = simulate_field(cars, params, 300, np.random.default_rng(7))
    b = simulate_field(cars, params, 300, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a, axis=1), np.tile(np.arange(1, 7), (300, 1)))


def test_safety_car_bunches_the_field():
    cars = [Car("a", 1, 0.0, ONE_STOP), Car("b", 2, 2.0, ONE_STOP)]
    # Without SC the slow car is far behind; with a guaranteed SC, gaps reset (still P2).
    pos = simulate_field(cars, _params(sc_hazard_per_lap=1.0), 10, np.random.default_rng(0))
    assert (pos[:, 0] == 1).all()


def test_undercut_emerges_from_fresh_tyres():
    """Equal cars, high degradation, overtaking hard: pitting first beats pitting later."""
    deg = {"SOFT": np.array([0.3]), "MEDIUM": np.array([0.3]), "HARD": np.array([0.3])}
    params = _params(deg_draws=deg, pass_threshold_s=5.0, pass_scale_s=0.1)
    early = Strategy("early", (Stint("MEDIUM", 18), Stint("HARD", 22)))
    late = Strategy("late", (Stint("MEDIUM", 20), Stint("HARD", 20)))
    cars = [Car("ahead", 1, 0.0, late), Car("behind", 2, 0.0, early)]
    pos = simulate_field(cars, params, 20, np.random.default_rng(0))
    assert (pos[:, 1] == 1).all()


def test_candidates_are_legal():
    cands = candidate_strategies(LAPS, step=4)
    assert cands and all(s.total_laps == LAPS for s in cands)
    assert all(len({st.compound for st in s.stints}) >= 2 for s in cands)
    assert {len(s.stints) for s in cands} == {2, 3}


def test_strategy_from_stints_fixes_length_and_wet():
    stints = pd.DataFrame(
        {"stint_number": [1, 2], "compound": ["INTERMEDIATE", "HARD"], "laps": [10, 25]}
    )
    s = strategy_from_stints(stints, LAPS)
    assert s.total_laps == LAPS and s.stints[0].compound == "MEDIUM"


def test_evaluate_for_driver_uses_common_random_numbers():
    cars = [Car(f"d{i}", i + 1, 0.15 * i, ONE_STOP) for i in range(4)]
    params = _params(lap_noise_sd_s=0.4, sc_hazard_per_lap=0.03)
    table = evaluate_for_driver(cars, 2, [ONE_STOP], params, n_sims=200, seed=3)
    # The candidate identical to the actual strategy has exactly zero paired gain.
    same = table[(table["strategy"] == ONE_STOP.name) & ~table["is_actual"]].iloc[0]
    assert same["points_gain_vs_actual"] == 0 and same["p_more_points_than_actual"] == 0


def test_points_and_scores():
    assert list(points_for(np.array([[1, 10, 11]]))[0]) == [25, 1, 0]
    observed = np.array([1, 3])
    perfect = scoring.degenerate(observed, 4)
    assert np.allclose(scoring.rps(perfect, observed), 0)
    uniform = np.full((2, 4), 0.25)
    assert (scoring.rps(uniform, observed) > 0).all()
    pit = scoring.randomized_pit(perfect, observed, np.random.default_rng(0))
    assert ((pit >= 0) & (pit <= 1)).all()
    probs = scoring.position_distribution(np.array([[1, 2], [2, 1], [1, 2]]), 2)
    assert np.allclose(probs, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])


def test_distributed_matches_local(spark):
    from pitwall.simulation.distributed import (
        RaceSetup,
        evaluate_strategies_on_spark,
        position_probs,
        simulate_on_spark,
        summarize,
    )

    cars = tuple(Car(f"d{i}", i + 1, 0.2 * i, ONE_STOP) for i in range(4))
    race = RaceSetup("r1", cars, _params(lap_noise_sd_s=0.4))
    counts = simulate_on_spark(spark, [race], n_sims=600, batch_size=200, seed=0, partitions=2)
    probs = position_probs(counts, "r1", 4)
    assert np.allclose(probs.sum(axis=1), 1) and counts["count"].sum() == 600 * 4
    assert probs[0, 0] > probs[3, 0]

    sums = evaluate_strategies_on_spark(
        spark, [race], {"r1": [3]}, {"r1": [ONE_STOP]}, n_sims=300, batch_size=100
    )
    assert set(sums["split"]) == {0, 1}
    table = summarize(sums)
    assert set(table["n_sims"]) == {300}
    assert np.allclose(table["points_gain_vs_actual"], 0)
    assert set(table["strategy"]) == {"actual:MH-20", "MH-20"}
