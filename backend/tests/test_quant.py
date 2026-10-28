import math

import numpy as np
import pytest
from scipy.stats import poisson

from app.engines.models_mma import FighterProfile, predict_fight
from app.engines.quant_engine import (
    bookmaker_margin, evaluate_market, expected_value, implied_probability, kelly_fraction,
    remove_vig, run_backtest, simulate_fight, simulate_football,
)

MATRIX = np.outer(poisson.pmf(range(8), 1.6), poisson.pmf(range(8), 1.1))
MATRIX /= MATRIX.sum()


def test_football_converges_to_matrix():
    s = simulate_football(MATRIX, 10_000, seed=1).summary()
    assert s["home_win"] == pytest.approx(np.tril(MATRIX, -1).sum(), abs=0.02)
    assert s["draw"] == pytest.approx(np.trace(MATRIX), abs=0.02)
    assert s["away_win"] == pytest.approx(np.triu(MATRIX, 1).sum(), abs=0.02)
    assert s["home_win"] + s["draw"] + s["away_win"] == pytest.approx(1.0)
    lo, hi = s["confidence_intervals"]["draw"]
    assert lo < s["draw"] < hi
    assert s["over_under"]["2.5"]["over"] + s["over_under"]["2.5"]["under"] == pytest.approx(1.0)
    assert len(s["score_distribution"]) == 10
    assert sum(s["goal_histogram"]["home"]) == 10_000


def test_football_speed_and_reproducibility():
    sim = simulate_football(MATRIX, 10_000, seed=7)
    assert sim.elapsed_ms < 200
    again = simulate_football(MATRIX, 10_000, seed=7)
    assert np.array_equal(sim.home_goals, again.home_goals)
    assert np.array_equal(sim.away_goals, again.away_goals)


def test_football_validation():
    with pytest.raises(ValueError):
        simulate_football(MATRIX, 0)
    with pytest.raises(ValueError):
        simulate_football(MATRIX, 1_000_001)
    with pytest.raises(ValueError):
        simulate_football(-MATRIX)
    with pytest.raises(ValueError):
        simulate_football(np.zeros((3, 3)))


@pytest.mark.parametrize("method", ["proportional", "power"])
def test_remove_vig_sums_to_one(method):
    odds = [2.10, 3.40, 3.60]
    fair = remove_vig(odds, method)
    assert sum(fair) == pytest.approx(1.0)
    assert all(f < implied_probability(o) for f, o in zip(fair, odds))
    assert bookmaker_margin(odds) == pytest.approx(sum(1 / o for o in odds) - 1)


def test_power_removes_less_from_favourites_than_proportional():
    odds = [1.30, 3.80, 9.00]
    prop, power = remove_vig(odds, "proportional"), remove_vig(odds, "power")
    assert power[0] > prop[0] and power[2] < prop[2]


def test_odds_validation():
    with pytest.raises(ValueError):
        implied_probability(1.0)
    with pytest.raises(ValueError):
        remove_vig([2.0])
    with pytest.raises(ValueError):
        remove_vig([2.0, 2.0], "nope")
    with pytest.raises(ValueError):
        expected_value(1.2, 2.0)


def test_ev_and_kelly_known_values():
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 2.0, multiplier=1.0, cap=1.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.025)
    assert kelly_fraction(0.80, 3.0, multiplier=1.0) == pytest.approx(0.10)  # capped
    assert kelly_fraction(0.40, 2.0) == 0.0  # negative EV


def test_evaluate_market():
    bets = evaluate_market([("H", 0.5, 2.4), ("D", 0.3, 3.5), ("A", 0.2, 5.0)], bankroll=1000.0)
    assert [b.selection for b in bets] == ["H", "D", "A"]
    best = bets[0]
    assert best.expected_value == pytest.approx(0.2) and best.is_value
    assert best.stake == pytest.approx(1000 * 0.25 * 0.2 / 1.4)
    assert not bets[2].is_value and bets[2].stake == 0.0
    assert sum(b.fair_prob for b in bets) == pytest.approx(1.0)
    assert best.edge == pytest.approx(best.model_prob - best.fair_prob)


def test_backtest_exact_values():
    records = [
        {"probs": {"H": 0.5, "D": 0.3, "A": 0.2}, "odds": {"H": 2.4, "D": 3.5, "A": 5.0}, "outcome": "H"},
        {"probs": {"H": 0.4, "D": 0.3, "A": 0.3}, "odds": {"H": 2.0, "D": 3.0, "A": 4.0}, "outcome": "H"},
        {"probs": {"H": 0.3, "D": 0.4, "A": 0.3}, "odds": {"H": None, "D": None, "A": None}, "outcome": "A"},
    ]
    r = run_backtest(records, ["H", "D", "A"])
    # bet 1 on H: stake 1000*0.25*0.2/1.4, wins 1000*0.25*0.2 = 50
    # bet 2 on A (EV 0.2): stake 1050*0.25*0.2/3 = 17.5, loses
    stake1 = 1000 * 0.25 * 0.2 / 1.4
    assert (r.n_events, r.n_bets) == (3, 2)
    assert r.profit == pytest.approx(50.0 - 17.5)
    assert r.final_bankroll == pytest.approx(1032.5)
    assert r.bankroll_curve == pytest.approx([1000.0, 1050.0, 1032.5])
    assert r.roi == pytest.approx(32.5 / (stake1 + 17.5))
    assert r.yield_pct == pytest.approx(r.roi * 100)
    assert r.brier_score == pytest.approx((0.38 + 0.54 + 0.74) / 3)
    assert r.log_loss == pytest.approx(-(math.log(0.5) + math.log(0.4) + math.log(0.3)) / 3)
    assert r.accuracy == pytest.approx(2 / 3)
    assert r.max_drawdown == pytest.approx(17.5 / 1050)
    assert r.sharpe == pytest.approx(1 / 6, abs=1e-4)
    assert [b["selection"] for b in r.bets] == ["H", "A"]
    assert len(r.calibration) == 10 and sum(c["count"] for c in r.calibration) == 9
    assert isinstance(r.to_dict()["bets"], list)


def test_backtest_validation():
    with pytest.raises(ValueError):
        run_backtest([], ["H", "D", "A"])
    with pytest.raises(ValueError):
        run_backtest([{"probs": {"H": 0.5, "A": 0.5}, "odds": {}, "outcome": "D"}], ["H", "A"])


def test_fight_simulation_converges_to_prediction():
    p = predict_fight(FighterProfile(name="A", slpm=5.0, elo=1600.0), FighterProfile(name="B"), rounds=5)
    sim = simulate_fight(p.method_probs, p.finish_round_probs, 5, 100_000, seed=3)
    s = sim.summary()
    assert s["win_prob_a"] == pytest.approx(p.win_prob_a, abs=0.01)
    for side in ("a", "b"):
        for method, prob in p.method_probs[side].items():
            assert s["methods"][side][method] == pytest.approx(prob, abs=0.01)
    for key, prob in p.finish_round_probs.items():
        assert s["round_distribution"][key] == pytest.approx(prob, abs=0.01)
    assert s["finish_rate"] == pytest.approx(1 - p.finish_round_probs["decision"], abs=0.01)
    lo, hi = s["confidence_intervals"]["a"]
    assert lo < p.win_prob_a < hi
    assert sim.end_rounds.max() == 5 and sim.end_rounds.min() == 1
    assert (sim.end_rounds[sim.methods == 2] == 5).all()
