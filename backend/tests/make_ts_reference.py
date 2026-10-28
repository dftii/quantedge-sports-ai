"""Write reference outputs of the Python engines for the TypeScript port's parity tests.

Run from backend/:  python -m tests.make_ts_reference ../frontend/src/engine/__fixtures__/reference.json
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import init_db, make_engine
from app.engines.models_football import (
    MatchAdjustments,
    TeamProfile,
    apply_adjustments,
    dixon_coles_matrix,
    expected_goals_from_profiles,
    market_probabilities,
    predict_match,
    walk_forward_predictions,
)
from app.engines.models_mma import FighterProfile, predict_fight
from app.engines.quant_engine import expected_value, kelly_fraction, remove_vig, run_backtest
from app.main import match_records
from app.seed import seed_database

FOOTBALL_CASES = [
    {  # balanced sides, neutral adjustments
        "home": {"name": "Alpha", "elo": 1500, "xg_for": 1.35, "xg_against": 1.35, "possession": 50, "defensive_rating": 1.0},
        "away": {"name": "Beta", "elo": 1500, "xg_for": 1.35, "xg_against": 1.35, "possession": 50, "defensive_rating": 1.0},
        "league_avg_goals": 1.35, "rho": -0.08, "lam3": 0.0, "max_goals": 10, "adjustments": {},
    },
    {  # strong home side, bivariate term, mixed adjustments
        "home": {"name": "Gamma", "elo": 1780, "xg_for": 2.1, "xg_against": 0.8, "possession": 62, "defensive_rating": 1.2},
        "away": {"name": "Delta", "elo": 1560, "xg_for": 1.1, "xg_against": 1.6, "possession": 44, "defensive_rating": 0.9},
        "league_avg_goals": 1.4, "rho": -0.1, "lam3": 0.12, "max_goals": 8,
        "adjustments": {"home_advantage": 1.1, "home_form": 0.1, "away_form": -0.05, "home_absences": 2, "away_absences": 1},
    },
    {  # extreme profile: lambdas hit the clamps, rho gets clipped, lam3 gets capped
        "home": {"name": "Eps", "elo": 2100, "xg_for": 4.5, "xg_against": 0.4, "possession": 75, "defensive_rating": 1.5},
        "away": {"name": "Zeta", "elo": 1200, "xg_for": 0.3, "xg_against": 3.8, "possession": 25, "defensive_rating": 0.7},
        "league_avg_goals": 1.2, "rho": 0.3, "lam3": 0.5, "max_goals": 10,
        "adjustments": {"home_advantage": 1.3, "home_form": 0.2, "away_form": -0.2, "home_absences": 0, "away_absences": 5},
    },
]

DC_CASES = [
    {"lam": 1.6, "mu": 1.1, "rho": -0.1, "lam3": 0.15, "max_goals": 10},
    {"lam": 2.4, "mu": 0.7, "rho": 0.05, "lam3": 0.3, "max_goals": 7},
]

MMA_CASES = [
    {"a": {"name": "A1"}, "b": {"name": "B1"}, "rounds": 3},
    {
        "a": {"name": "Striker", "slpm": 5.8, "sapm": 3.0, "striking_accuracy": 0.52, "striking_defense": 0.62,
              "td_avg": 0.4, "td_accuracy": 0.3, "td_defense": 0.8, "sub_avg": 0.1, "reach_cm": 191, "height_cm": 185,
              "cardio": 0.85, "age": 27, "wins": 22, "losses": 2, "ko_wins": 16, "sub_wins": 1, "dec_wins": 5, "elo": 1720},
        "b": {"name": "Grappler", "slpm": 2.6, "sapm": 4.2, "striking_accuracy": 0.41, "striking_defense": 0.5,
              "td_avg": 4.1, "td_accuracy": 0.55, "td_defense": 0.7, "sub_avg": 2.0, "reach_cm": 178, "height_cm": 175,
              "cardio": 0.6, "age": 35, "wins": 18, "losses": 6, "ko_wins": 2, "sub_wins": 11, "dec_wins": 5, "elo": 1610},
        "rounds": 5,
    },
    {
        "a": {"name": "Veteran", "slpm": 3.1, "sapm": 2.4, "striking_accuracy": 0.48, "striking_defense": 0.66,
              "td_avg": 1.9, "td_accuracy": 0.45, "td_defense": 0.75, "sub_avg": 0.9, "reach_cm": 183, "height_cm": 180,
              "cardio": 0.7, "age": 39, "wins": 30, "losses": 9, "ko_wins": 7, "sub_wins": 8, "dec_wins": 15, "elo": 1540},
        "b": {"name": "Prospect", "slpm": 4.4, "sapm": 3.9, "striking_accuracy": 0.46, "striking_defense": 0.54,
              "td_avg": 2.2, "td_accuracy": 0.38, "td_defense": 0.58, "sub_avg": 0.7, "reach_cm": 185, "height_cm": 181,
              "cardio": 0.8, "age": 24, "wins": 9, "losses": 0, "ko_wins": 5, "sub_wins": 2, "dec_wins": 2, "elo": 1565},
        "rounds": 3,
    },
]


def football_case(c: dict) -> dict:
    home, away, adj = TeamProfile(**c["home"]), TeamProfile(**c["away"]), MatchAdjustments(**c["adjustments"])
    lam, mu = expected_goals_from_profiles(home, away, c["league_avg_goals"])
    pred = predict_match(home, away, c["league_avg_goals"], c["rho"], c["lam3"], c["max_goals"], adjustments=adj)
    return {
        **c,
        "expected_goals": [lam, mu],
        "adjusted_goals": list(apply_adjustments(lam, mu, adj)),
        "prediction": {
            "lambda_home": pred.lambda_home, "lambda_away": pred.lambda_away, "rho": pred.rho, "lam3": pred.lam3,
            "elo_win_expectancy": pred.elo_win_expectancy, "matrix": pred.matrix.tolist(), "markets": pred.markets,
        },
    }


def dc_case(c: dict) -> dict:
    m = dixon_coles_matrix(c["lam"], c["mu"], c["rho"], c["lam3"], c["max_goals"])
    return {**c, "matrix": m.tolist(), "markets": market_probabilities(m)}


def backtest_case() -> dict:
    engine = make_engine("sqlite://", poolclass=StaticPool)
    init_db(engine)
    with sessionmaker(engine)() as db:
        seed_database(db)
        records = match_records(db)[:80]
    warmup, bankroll, kelly, min_ev = 30, 1000.0, 0.25, 0.02
    rows = walk_forward_predictions(records, warmup)
    report = run_backtest(
        [
            {"probs": {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]},
             "odds": {"H": r["odds_home"], "D": r["odds_draw"], "A": r["odds_away"]}, "outcome": r["outcome"]}
            for r in rows
        ],
        ("H", "D", "A"), bankroll, kelly, min_ev,
    )
    iso = lambda d: d.isoformat()  # noqa: E731
    return {
        "matches": [{**asdict(m), "date": iso(m.date)} for m in records],
        "warmup": warmup, "bankroll": bankroll, "kelly_multiplier": kelly, "min_ev": min_ev,
        "rows": [{**r, "date": iso(r["date"])} for r in rows],
        "report": report.to_dict(),
    }


def build() -> dict:
    odds_sets = [[2.1, 3.4, 3.6], [1.5, 4.2, 7.0], [1.91, 1.91], [1.2, 6.0, 15.0, 41.0]]
    pk = [(0.55, 2.1), (0.3, 3.0), (0.5, 1.9), (0.8, 1.2), (0.05, 30.0)]
    return {
        "football": [football_case(c) for c in FOOTBALL_CASES],
        "dixon_coles": [dc_case(c) for c in DC_CASES],
        "mma": [
            {**c, "prediction": predict_fight(FighterProfile(**c["a"]), FighterProfile(**c["b"]), c["rounds"]).to_dict()}
            for c in MMA_CASES
        ],
        "remove_vig": [
            {"odds": o, "method": m, "fair": remove_vig(o, m)} for o in odds_sets for m in ("proportional", "power")
        ],
        "expected_value": [{"prob": p, "odds": o, "value": expected_value(p, o)} for p, o in pk],
        "kelly_fraction": [
            {"prob": p, "odds": o, "multiplier": k, "cap": cap, "value": kelly_fraction(p, o, k, cap)}
            for p, o in pk for k, cap in ((0.25, 0.10), (1.0, 0.5))
        ],
        "backtest": backtest_case(),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "../frontend/src/engine/__fixtures__/reference.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), allow_nan=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
