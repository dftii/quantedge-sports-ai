import datetime

import numpy as np
import pytest

from app.engines.models_football import (
    DixonColesModel,
    EloRating,
    MatchAdjustments,
    MatchRecord,
    TeamProfile,
    apply_adjustments,
    bivariate_poisson_matrix,
    dixon_coles_matrix,
    expected_goals_from_profiles,
    market_probabilities,
    poisson_pmf_vector,
    predict_match,
    walk_forward_predictions,
)

TEAMS = [f"T{i}" for i in range(8)]


def synthetic_matches(n: int = 300, seed: int = 7) -> list[MatchRecord]:
    """Matches drawn from known strengths: T0 has by far the best attack."""
    rng = np.random.default_rng(seed)
    attack = np.array([0.6, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.3])
    defence = np.array([-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])  # lower = better defence
    start = datetime.date(2023, 8, 1)
    out = []
    for i in range(n):
        h, a = rng.choice(len(TEAMS), size=2, replace=False)
        lam = np.exp(0.2 + 0.25 + attack[h] + defence[a])
        mu = np.exp(0.2 + attack[a] + defence[h])
        out.append(MatchRecord(start + datetime.timedelta(days=i), TEAMS[h], TEAMS[a],
                               int(rng.poisson(lam)), int(rng.poisson(mu))))
    return out


def test_matrices_sum_to_one():
    assert bivariate_poisson_matrix(1.2, 0.9, 0.2).sum() == pytest.approx(1.0)
    assert dixon_coles_matrix(1.6, 1.1, -0.1, lam3=0.15).sum() == pytest.approx(1.0)


def test_1x2_sums_to_one_and_ou_complements():
    m = market_probabilities(dixon_coles_matrix(1.6, 1.1, -0.1))
    assert m["home_win"] + m["draw"] + m["away_win"] == pytest.approx(1.0)
    assert m["btts_yes"] + m["btts_no"] == pytest.approx(1.0)
    assert len(m["over_under"]) == 5 and len(m["correct_scores"]) == 10
    for line in m["over_under"].values():
        assert line["over"] + line["under"] == pytest.approx(1.0)
    probs = [c["probability"] for c in m["correct_scores"]]
    assert probs == sorted(probs, reverse=True)


def test_dc_without_correction_is_independent_poisson():
    lam, mu = 1.5, 1.1
    outer = np.outer(poisson_pmf_vector(lam, 10), poisson_pmf_vector(mu, 10))
    np.testing.assert_allclose(dixon_coles_matrix(lam, mu, 0.0), outer / outer.sum(), atol=1e-12)


def test_bivariate_increases_draws_with_same_marginal_means():
    indep = market_probabilities(dixon_coles_matrix(1.5, 1.2, 0.0))
    biv = market_probabilities(dixon_coles_matrix(1.5, 1.2, 0.0, lam3=0.3))
    assert biv["draw"] > indep["draw"]
    assert biv["expected_home_goals"] == pytest.approx(1.5, abs=1e-3)
    assert biv["expected_away_goals"] == pytest.approx(1.2, abs=1e-3)


def test_rho_is_clipped_to_feasible_range_and_inputs_validated():
    assert (dixon_coles_matrix(3.0, 3.0, -5.0) >= 0).all()
    with pytest.raises(ValueError):
        dixon_coles_matrix(-1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        dixon_coles_matrix(1.0, 1.0, 0.0, max_goals=0)


def test_elo_conserves_rating_and_favourite_gains_less():
    elo = EloRating()
    elo.ratings = {"fav": 1700.0, "dog": 1500.0}
    before = sum(elo.ratings.values())
    fav_new, dog_new = elo.update("fav", "dog", 1, 0)
    assert fav_new + dog_new == pytest.approx(before)
    fav_gain = fav_new - 1700.0

    elo.ratings = {"fav": 1700.0, "dog": 1500.0}
    upset_home, _ = elo.update("dog", "fav", 1, 0)  # the underdog wins at home by the same margin
    assert fav_gain > 0
    assert upset_home - 1500.0 > fav_gain


def test_elo_fit_is_chronological():
    ms = synthetic_matches(60)
    assert EloRating().fit(list(reversed(ms))) == EloRating().fit(ms)


def test_dixon_coles_recovers_strongest_attack():
    model = DixonColesModel().fit(synthetic_matches())
    assert model.fitted
    assert max(model.attack, key=model.attack.get) == "T0"
    assert sum(model.attack.values()) == pytest.approx(0.0, abs=1e-9)
    assert model.home_advantage > 0
    p = model.predict("T0", "T7")
    assert p.lambda_home > p.lambda_away
    assert p.markets["home_win"] + p.markets["draw"] + p.markets["away_win"] == pytest.approx(1.0)
    assert p.to_dict()["matrix"][0][0] == round(float(p.matrix[0, 0]), 6)


def test_dixon_coles_errors():
    with pytest.raises(ValueError, match="no matches"):
        DixonColesModel().fit([])
    with pytest.raises(ValueError, match="at least 2"):
        DixonColesModel().fit([MatchRecord(datetime.date(2024, 1, 1), "A", "A", 1, 1)])
    model = DixonColesModel().fit(synthetic_matches(100))
    with pytest.raises(ValueError, match="unknown team"):
        model.expected_goals("T0", "Nobody")
    with pytest.raises(ValueError, match="not fitted"):
        DixonColesModel().expected_goals("T0", "T1")


def test_profiles_better_team_gets_higher_lambda():
    strong = TeamProfile("S", elo=1800, xg_for=2.0, xg_against=0.9, possession=60, defensive_rating=1.1)
    weak = TeamProfile("W", elo=1400, xg_for=1.0, xg_against=1.8, possession=42)
    lam, mu = expected_goals_from_profiles(strong, weak)
    assert lam > mu and 0.05 <= mu and lam <= 6.0
    pred = predict_match(strong, weak, lam3=0.1)
    assert pred.markets["home_win"] > pred.markets["away_win"]
    assert pred.elo_win_expectancy > 0.5


def test_walk_forward_is_leak_free_and_excludes_warmup():
    ms = synthetic_matches(120)
    rows = walk_forward_predictions(ms, warmup=30)
    assert len(rows) == 90
    assert rows[0]["date"] == sorted(ms, key=lambda m: m.date)[30].date
    for r in rows:
        assert r["p_home"] + r["p_draw"] + r["p_away"] == pytest.approx(1.0)
        assert r["outcome"] in "HDA"

    # Changing the last result must not change any prediction (state is updated after predicting).
    last = max(ms, key=lambda m: m.date)
    altered = [m if m is not last else MatchRecord(m.date, m.home, m.away, 9, 0) for m in ms]
    assert [r["p_home"] for r in walk_forward_predictions(altered, warmup=30)] == [r["p_home"] for r in rows]


def test_adjustments_neutral_is_identity():
    assert apply_adjustments(1.5, 1.2, MatchAdjustments()) == (1.5, 1.2)
    home, away = TeamProfile("H", elo=1600, xg_for=1.8), TeamProfile("A", xg_against=1.6)
    base = predict_match(home, away)
    neutral = predict_match(home, away, adjustments=MatchAdjustments())
    assert neutral.lambda_home == base.lambda_home and neutral.lambda_away == base.lambda_away
    assert np.array_equal(neutral.matrix, base.matrix)


def test_adjustments_known_values_and_clamp():
    adj = MatchAdjustments(home_advantage=1.1, home_form=0.1, away_form=-0.05, home_absences=2, away_absences=1)
    lam, mu = apply_adjustments(1.5, 1.2, adj)
    assert lam == pytest.approx(1.5 * 1.1 * (1 - 0.08) * (1 + 0.03) * 1.1)
    assert mu == pytest.approx(1.2 * 0.95 * (1 - 0.04) * (1 + 0.06) / 1.1)
    assert apply_adjustments(5.9, 0.06, MatchAdjustments(home_advantage=1.3, home_form=0.2)) == (6.0, 0.05)


def test_predict_match_uses_adjusted_lambdas():
    home, away = TeamProfile("H"), TeamProfile("A")
    adj = MatchAdjustments(home_absences=3)
    pred = predict_match(home, away, adjustments=adj)
    lam, mu = apply_adjustments(*expected_goals_from_profiles(home, away), adj)
    assert (pred.lambda_home, pred.lambda_away) == (lam, mu)
    assert pred.markets["home_win"] < predict_match(home, away).markets["home_win"]
