import math
from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.engines.models_football import dixon_coles_matrix, market_probabilities, walk_forward_predictions
from app.engines.quant_engine import expected_value, kelly_fraction, run_backtest
from app.main import match_records
from app.models import Competition, Event, EventStatus, Fighter, Sport, Team
from app.schemas import (
    CompetitionRead,
    EventRead,
    FighterRead,
    FightSimulationRequest,
    FootballSimulationRequest,
    OddsInput1X2,
    OddsInputFight,
    TeamRead,
)
from app.seed import seed_database


def test_seed_counts(seeded_session):
    db = seeded_session
    football = (Event.sport == Sport.FOOTBALL,)
    count = lambda *w: len(db.scalars(select(Event).where(*w)).all())  # noqa: E731
    assert len(db.scalars(select(Competition)).all()) == 2
    assert len(db.scalars(select(Team)).all()) == 20
    assert len(db.scalars(select(Fighter)).all()) == 16
    assert count(*football, Event.status == EventStatus.FINISHED) == 150
    assert count(*football, Event.status == EventStatus.SCHEDULED) == 4
    assert count(Event.sport == Sport.MMA, Event.status == EventStatus.FINISHED) == 40
    assert count(Event.sport == Sport.MMA, Event.status == EventStatus.SCHEDULED) == 3
    assert all(t.matches_played == 15 and len(t.form) == 5 for t in db.scalars(select(Team)))


def test_seed_is_idempotent(seeded_session):
    first = seed_database(seeded_session)
    assert first["seeded"] is False
    assert first["teams"] == 20 and first["events"] == 197


def test_seed_force_reseeds_identically(seeded_session):
    before = [(t.name, t.elo) for t in seeded_session.scalars(select(Team).order_by(Team.id))]
    result = seed_database(seeded_session, force=True)
    after = [(t.name, t.elo) for t in seeded_session.scalars(select(Team).order_by(Team.id))]
    assert result["seeded"] is True and result["events"] == 197 and result["competitions"] == 2
    assert before == after


def test_football_event_requires_teams(db_session):
    comp = Competition(name="X", sport=Sport.FOOTBALL, season="2025")
    db_session.add(comp)
    db_session.flush()
    db_session.add(Event(sport=Sport.FOOTBALL, competition_id=comp.id, event_date=date(2025, 1, 1)))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_foreign_keys_enforced(db_session):
    db_session.add(Team(name="Orphan", short_name="ORP", competition_id=999))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_read_schemas_validate_from_orm(seeded_session):
    db = seeded_session
    assert CompetitionRead.model_validate(db.scalars(select(Competition)).first()).id
    assert TeamRead.model_validate(db.scalars(select(Team)).first()).elo > 0
    assert FighterRead.model_validate(db.scalars(select(Fighter)).first()).weight_class
    for event in db.scalars(select(Event)):
        EventRead.model_validate(event)


def test_request_schemas_reject_bad_input():
    with pytest.raises(ValidationError):
        OddsInput1X2(home=1.0, draw=3.2, away=4.0)
    with pytest.raises(ValidationError):
        OddsInputFight(fighter_a=1.5, fighter_b=0.9)
    with pytest.raises(ValidationError):
        FootballSimulationRequest(home_team_id=1, away_team_id=2, n_simulations=0)
    with pytest.raises(ValidationError):
        FightSimulationRequest(fighter_a_id=1, fighter_b_id=2, rounds=4)
    req = FootballSimulationRequest(home_team_id=1, away_team_id=2)
    assert req.n_simulations == 10_000 and req.rho == -0.08 and req.save is True


# --------------------------------------------------------------------------- seed data integrity
def test_seed_team_stats_and_history(seeded_session):
    db = seeded_session
    for t in db.scalars(select(Team)):
        assert t.elo > 0 and t.ppda > 0 and t.defensive_rating > 0
        assert 0 <= t.possession_avg <= 100 and 0 <= t.clean_sheet_rate <= 1
        assert min(t.xg_for_avg, t.xg_against_avg, t.goals_for_avg, t.goals_against_avg) >= 0
        assert 0 <= t.shots_on_target_per_match <= t.shots_per_match
        assert set(t.form) <= set("WDL")

    finished = db.scalars(
        select(Event).where(Event.sport == Sport.FOOTBALL, Event.status == EventStatus.FINISHED)
    ).all()
    goals = [e.home_goals + e.away_goals for e in finished]
    league = db.scalars(select(Competition).where(Competition.sport == Sport.FOOTBALL)).one()
    assert 2.2 <= sum(goals) / len(goals) <= 3.6
    assert league.avg_goals_per_match == pytest.approx(sum(goals) / len(goals), abs=1e-3)
    assert max(e.event_date for e in finished) < date(2025, 6, 1)

    games = {t.id: 0 for t in db.scalars(select(Team))}
    pairs = set()
    for e in finished:
        games[e.home_team_id] += 1
        games[e.away_team_id] += 1
        pairs.add(frozenset((e.home_team_id, e.away_team_id)))
    assert set(games.values()) == {15} and len(pairs) == 150  # nobody meets twice


def test_seed_fighters_and_odds(seeded_session):
    db = seeded_session
    for f in db.scalars(select(Fighter)):
        assert f.ko_wins + f.sub_wins + f.dec_wins <= f.wins
        assert 0 < f.cardio <= 1 and f.weight_class in ("Lightweight", "Welterweight")
    assert len(db.scalars(select(Fighter).where(Fighter.weight_class == "Lightweight")).all()) == 8

    for e in db.scalars(select(Event)):
        odds, margin = ((e.odds_home, e.odds_draw, e.odds_away), 0.05) if e.sport == Sport.FOOTBALL else (
            (e.odds_fighter_a, e.odds_fighter_b), 0.045)
        assert all(o > 1 for o in odds)
        assert sum(1 / o for o in odds) - 1 == pytest.approx(margin, abs=0.02)  # 2-decimal rounding + noise

    for e in db.scalars(select(Event).where(Event.sport == Sport.MMA, Event.status == EventStatus.FINISHED)):
        assert e.fighter_a.weight_class == e.fighter_b.weight_class
        assert e.winner_fighter_id in (e.fighter_a_id, e.fighter_b_id, None)


def test_backtest_over_seeded_history_is_finite(seeded_session):
    records = match_records(seeded_session)
    assert len(records) == 150
    rows = walk_forward_predictions(records, warmup=30)
    report = run_backtest(
        [{"probs": {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]},
          "odds": {"H": r["odds_home"], "D": r["odds_draw"], "A": r["odds_away"]},
          "outcome": r["outcome"]} for r in rows],
        ("H", "D", "A"),
    )
    assert report.n_events == len(rows) == 120
    assert math.isfinite(report.brier_score) and 0 <= report.brier_score <= 2
    assert math.isfinite(report.log_loss) and report.log_loss > 0


# --------------------------------------------------------------------------- explicit math checks
SCORE_GRID_CASES = [
    (1.4, 1.1, -0.08, 0.0), (2.6, 0.7, -0.15, 0.1), (0.2, 4.5, -0.10, 0.05), (4.5, 0.2, 0.10, 0.0),
    (1.0, 1.0, 0.00, 0.3), (3.0, 3.0, -0.30, 0.9), (0.2, 0.2, 0.50, 0.1), (0.5, 0.5, -5.0, 0.0),
]


@pytest.mark.parametrize(("lam", "mu", "rho", "lam3"), SCORE_GRID_CASES)
def test_dixon_coles_matrix_is_a_valid_pmf(lam, mu, rho, lam3):
    matrix = dixon_coles_matrix(lam, mu, rho, lam3, max_goals=15)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert (matrix >= 0).all()
    markets = market_probabilities(matrix)
    assert markets["home_win"] + markets["draw"] + markets["away_win"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(("lam", "mu"), [(1.4, 1.1), (0.2, 4.5), (4.5, 0.2), (2.0, 2.0)])
def test_independent_poisson_recovers_expected_goals(lam, mu):
    markets = market_probabilities(dixon_coles_matrix(lam, mu, rho=0.0, lam3=0.0, max_goals=15))
    assert markets["expected_home_goals"] == pytest.approx(lam, abs=1e-3)
    assert markets["expected_away_goals"] == pytest.approx(mu, abs=1e-3)


def test_expected_value_and_kelly_known_values():
    assert expected_value(0.55, 2.0) == pytest.approx(0.10, abs=1e-12)
    assert kelly_fraction(0.55, 2.0, multiplier=1.0, cap=1.0) == pytest.approx(0.10, abs=1e-12)


@pytest.mark.parametrize(("p", "odds"), [(0.55, 2.0), (0.40, 3.0), (0.30, 4.5), (0.62, 1.8), (0.10, 12.0)])
def test_expected_value_and_kelly_closed_forms(p, odds):
    assert expected_value(p, odds) == pytest.approx(p * odds - 1, abs=1e-12)
    assert kelly_fraction(p, odds, multiplier=1.0, cap=1.0) == pytest.approx((p * odds - 1) / (odds - 1), abs=1e-12)
    assert kelly_fraction(p, odds, multiplier=0.5, cap=1.0) == pytest.approx(0.5 * (p * odds - 1) / (odds - 1), abs=1e-12)


def test_kelly_is_zero_for_no_edge_and_capped():
    assert expected_value(0.40, 2.0) == pytest.approx(-0.20, abs=1e-12)
    assert kelly_fraction(0.40, 2.0) == 0.0  # negative EV
    assert kelly_fraction(0.50, 2.0) == 0.0  # zero EV
    assert kelly_fraction(0.90, 3.0, multiplier=1.0, cap=1.0) == pytest.approx(0.85, abs=1e-12)
    assert kelly_fraction(0.90, 3.0) == 0.10  # 0.25 * 0.85 > default cap
    assert kelly_fraction(0.90, 3.0, multiplier=1.0, cap=0.4) == 0.4
