from dataclasses import replace

import pytest

from app.engines.models_mma import FighterProfile, classify_style, predict_fight

AVERAGE = FighterProfile(name="Average")
STRIKER = FighterProfile(name="Striker", slpm=6.0, sapm=2.5, striking_accuracy=0.55,
                         striking_defense=0.68, td_avg=0.3, td_accuracy=0.2, td_defense=0.85,
                         sub_avg=0.1, reach_cm=188.0, elo=1650.0)
WRESTLER = FighterProfile(name="Wrestler", slpm=2.5, sapm=3.0, td_avg=5.5, td_accuracy=0.55,
                          sub_avg=1.0, td_defense=0.80)


def test_probabilities_sum_to_one():
    for rounds in (3, 5):
        p = predict_fight(STRIKER, WRESTLER, rounds)
        assert p.win_prob_a + p.win_prob_b == pytest.approx(1.0, abs=1e-12)
        six = sum(v for side in p.method_probs.values() for v in side.values())
        assert six == pytest.approx(1.0, abs=1e-12)
        assert sum(p.finish_round_probs.values()) == pytest.approx(1.0, abs=1e-12)
        assert set(p.finish_round_probs) == {str(r) for r in range(1, rounds + 1)} | {"decision"}
        assert 0 < p.expected_rounds < rounds


def test_swapping_fighters_swaps_probabilities():
    p, q = predict_fight(STRIKER, WRESTLER), predict_fight(WRESTLER, STRIKER)
    assert p.win_prob_a == pytest.approx(q.win_prob_b, abs=1e-9)
    for method in p.method_probs["a"]:
        assert p.method_probs["a"][method] == pytest.approx(q.method_probs["b"][method], abs=1e-9)
        assert p.method_probs["b"][method] == pytest.approx(q.method_probs["a"][method], abs=1e-9)
    assert p.finish_round_probs == pytest.approx(q.finish_round_probs, abs=1e-9)
    assert p.factors["elo"] == pytest.approx(-q.factors["elo"], abs=1e-9)


def test_identical_fighters_are_even():
    assert predict_fight(AVERAGE, replace(AVERAGE, name="Twin")).win_prob_a == pytest.approx(0.5)


def test_better_striker_beats_average_fighter():
    assert predict_fight(STRIKER, AVERAGE).win_prob_a > 0.6


def test_takedown_threat_raises_sub_and_decision_share():
    poor_td_defense = replace(STRIKER, name="Poor", td_defense=0.30)
    p_poor = predict_fight(WRESTLER, poor_td_defense)
    p_high = predict_fight(WRESTLER, replace(poor_td_defense, name="High", td_defense=0.90))

    def grind(p):
        return sum(p.method_probs["a"][m] for m in ("submission", "decision"))

    assert grind(p_poor) > grind(p_high)
    assert p_poor.method_probs["a"]["submission"] > p_high.method_probs["a"]["submission"]
    assert p_poor.win_prob_a > p_high.win_prob_a


def test_five_rounds_raise_finish_rate():
    def finish(p):
        return 1.0 - p.finish_round_probs["decision"]

    assert finish(predict_fight(STRIKER, WRESTLER, 5)) > finish(predict_fight(STRIKER, WRESTLER, 3))


def test_ko_is_front_loaded_and_low_cardio_shifts_later():
    fresh = predict_fight(STRIKER, replace(AVERAGE, cardio=0.9)).finish_round_probs
    tired = predict_fight(STRIKER, replace(AVERAGE, cardio=0.3, name="Tired")).finish_round_probs
    assert fresh["1"] > fresh["2"] > fresh["3"]
    assert tired["3"] / tired["1"] > fresh["3"] / fresh["1"]


def test_style_classification():
    assert classify_style(STRIKER) == "striker"
    assert classify_style(WRESTLER) == "wrestler"
    assert classify_style(replace(WRESTLER, td_avg=1.0, sub_avg=3.0)) == "grappler"
    assert classify_style(AVERAGE) == "balanced"


def test_validation_errors():
    with pytest.raises(ValueError):
        FighterProfile(name="x", slpm=-1.0)
    with pytest.raises(ValueError):
        FighterProfile(name="x", td_defense=1.2)
    with pytest.raises(ValueError):
        FighterProfile(name="x", cardio=-0.1)
    with pytest.raises(ValueError):
        FighterProfile(name="x", reach_cm=0)
    with pytest.raises(ValueError):
        predict_fight(STRIKER, WRESTLER, rounds=4)
