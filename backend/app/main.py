"""QuantEdge Sports AI - FastAPI application (REST layer over the quant engines)."""
import csv
import io
import json
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import fields, replace
from datetime import date, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db, init_db
from app.engines.models_football import (
    DixonColesModel,
    FootballPrediction,
    MatchAdjustments,
    MatchRecord,
    TeamProfile,
    _rho_bounds,
    apply_adjustments,
    dixon_coles_matrix,
    market_probabilities,
    predict_match,
    walk_forward_predictions,
)
from app.engines.models_mma import FighterProfile, classify_style, predict_fight
from app.engines.quant_engine import (
    bookmaker_margin,
    evaluate_market,
    run_backtest,
    simulate_fight,
    simulate_football,
)
from app.models import Competition, Event, EventStatus, Fighter, Simulation, Sport, Team
from app.schemas import (
    BacktestRequest,
    CompetitionRead,
    EventCreate,
    EventRead,
    FighterCreate,
    FighterRead,
    FighterUpdate,
    FightSimulationRequest,
    FootballSimulationRequest,
    SimulationRead,
    TeamCreate,
    TeamRead,
    TeamUpdate,
)
from app.seed import seed_database

MODEL_VERSION = "quantedge-1.0.0"
DEFAULT_LEAGUE_GOALS = 1.35

Db = Annotated[Session, Depends(get_db)]


# --------------------------------------------------------------------------- helpers
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def to_jsonable(obj):
    """Round-trip through JSON so numpy scalars/arrays and dates become plain Python types."""
    return json.loads(json.dumps(obj, default=_json_default))


@contextmanager
def engine_errors():
    """Engines reject impossible inputs with ValueError; surface that as HTTP 422."""
    try:
        yield
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def get_or_404(db: Session, model, obj_id: int):
    obj = db.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {obj_id} not found")
    return obj


def require_refs(db: Session, *refs: tuple[type, int | None]) -> None:
    for model, obj_id in refs:
        if obj_id is not None:
            get_or_404(db, model, obj_id)


def require_distinct(a: int, b: int, what: str) -> None:
    if a == b:
        raise HTTPException(422, f"{what} must be two different ones")


def team_profile(team: Team) -> TeamProfile:
    # defensive_rating is derived from xg_against_avg, so passing both would count defence twice.
    return TeamProfile(
        name=team.name, elo=team.elo, xg_for=team.xg_for_avg, xg_against=team.xg_against_avg,
        possession=team.possession_avg,
    )


_FIGHTER_FIELDS = [f.name for f in fields(FighterProfile) if f.name != "name"]


def fighter_profile(fighter: Fighter) -> FighterProfile:
    """Nullable physical columns (age, height, reach) fall back to the profile defaults."""
    values = {k: getattr(fighter, k) for k in _FIGHTER_FIELDS if getattr(fighter, k) is not None}
    with engine_errors():
        return FighterProfile(name=fighter.name, **values)


def league_goals(team: Team) -> float:
    """Average goals per team per match in the team's competition."""
    return team.competition.avg_goals_per_match / 2 if team.competition else DEFAULT_LEAGUE_GOALS


def match_records(db: Session, competition_id: int | None = None) -> list[MatchRecord]:
    """Finished football events with a score, oldest first, as engine input."""
    names = dict(db.execute(select(Team.id, Team.name)).all())
    query = (
        select(Event)
        .where(
            Event.sport == Sport.FOOTBALL,
            Event.status == EventStatus.FINISHED,
            Event.home_goals.is_not(None),
            Event.away_goals.is_not(None),
        )
        .order_by(Event.event_date, Event.id)
    )
    if competition_id is not None:
        query = query.where(Event.competition_id == competition_id)
    return [
        MatchRecord(
            e.event_date, names[e.home_team_id], names[e.away_team_id], e.home_goals, e.away_goals,
            e.home_xg, e.away_xg, e.odds_home, e.odds_draw, e.odds_away,
        )
        for e in db.scalars(query)
    ]


def create_row(db: Session, model, data: BaseModel):
    row = model(**data.model_dump())
    db.add(row)
    db.commit()
    return row


def patch_row(db: Session, model, obj_id: int, data: BaseModel):
    row = get_or_404(db, model, obj_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(row, key, value)
    db.commit()
    return row


def save_and_respond(db: Session, req, sport: Sport, prediction: dict, simulation: dict,
                     bets: list | None, margin: float | None, **participant_ids) -> dict:
    """Build the simulation response and persist a Simulation row when req.save is set."""
    result = to_jsonable({"prediction": prediction, "simulation": simulation})
    value_bets = to_jsonable([b.to_dict() for b in bets]) if bets else None
    simulation_id = None
    if req.save:
        row = Simulation(
            sport=sport, label=req.label, n_simulations=req.n_simulations, seed=req.seed,
            model_version=MODEL_VERSION, input_params=req.model_dump(mode="json"),
            result=result, value_bets=value_bets, **participant_ids,
        )
        db.add(row)
        db.commit()
        simulation_id = row.id
    return {**result, "value_bets": value_bets, "market_margin": margin, "simulation_id": simulation_id}


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if os.environ.get("QUANTEDGE_AUTO_SEED", "1") != "0":
        with SessionLocal() as db:
            seed_database(db)  # no-op when the database already has data
    yield


app = FastAPI(
    title="QuantEdge Sports AI",
    version="1.0.0",
    description="Probabilistic football and MMA models, Monte Carlo simulation, value-bet detection and backtesting.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(IntegrityError)
async def integrity_error_handler(_request: Request, exc: IntegrityError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": f"Conflict: {exc.orig}"})


# --------------------------------------------------------------------------- health
health_router = APIRouter(prefix="/api", tags=["health"])


@health_router.get("/health")
def health(db: Db) -> dict:
    counts = {
        name: db.scalar(select(func.count()).select_from(model))
        for name, model in (
            ("competitions", Competition), ("teams", Team), ("fighters", Fighter),
            ("events", Event), ("simulations", Simulation),
        )
    }
    return {"status": "ok", "version": app.version, "model_version": MODEL_VERSION, "counts": counts}


# --------------------------------------------------------------------------- catalogue
catalogue = APIRouter(prefix="/api", tags=["catalogue"])


@catalogue.get("/competitions", response_model=list[CompetitionRead])
def list_competitions(db: Db):
    return db.scalars(select(Competition).order_by(Competition.id)).all()


@catalogue.get("/teams", response_model=list[TeamRead])
def list_teams(db: Db, competition_id: int | None = None):
    query = select(Team).order_by(Team.id)
    if competition_id is not None:
        query = query.where(Team.competition_id == competition_id)
    return db.scalars(query).all()


@catalogue.get("/teams/{team_id}", response_model=TeamRead)
def read_team(team_id: int, db: Db):
    return get_or_404(db, Team, team_id)


@catalogue.post("/teams", response_model=TeamRead, status_code=201)
def create_team(body: TeamCreate, db: Db):
    require_refs(db, (Competition, body.competition_id))
    return create_row(db, Team, body)


@catalogue.patch("/teams/{team_id}", response_model=TeamRead)
def update_team(team_id: int, body: TeamUpdate, db: Db):
    return patch_row(db, Team, team_id, body)


@catalogue.get("/fighters", response_model=list[FighterRead])
def list_fighters(db: Db, weight_class: str | None = None):
    query = select(Fighter).order_by(Fighter.id)
    if weight_class is not None:
        query = query.where(func.lower(Fighter.weight_class) == weight_class.lower())
    return db.scalars(query).all()


@catalogue.get("/fighters/{fighter_id}", response_model=FighterRead)
def read_fighter(fighter_id: int, db: Db):
    return get_or_404(db, Fighter, fighter_id)


@catalogue.post("/fighters", response_model=FighterRead, status_code=201)
def create_fighter(body: FighterCreate, db: Db):
    require_refs(db, (Competition, body.competition_id))
    return create_row(db, Fighter, body)


@catalogue.patch("/fighters/{fighter_id}", response_model=FighterRead)
def update_fighter(fighter_id: int, body: FighterUpdate, db: Db):
    return patch_row(db, Fighter, fighter_id, body)


@catalogue.get("/events", response_model=list[EventRead])
def list_events(
    db: Db,
    sport: Sport | None = None,
    status: EventStatus | None = None,
    competition_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    query = select(Event).order_by(Event.event_date.desc(), Event.id.desc()).limit(limit).offset(offset)
    for column, value in ((Event.sport, sport), (Event.status, status), (Event.competition_id, competition_id)):
        if value is not None:
            query = query.where(column == value)
    return db.scalars(query).all()


@catalogue.post("/events", response_model=EventRead, status_code=201)
def create_event(body: EventCreate, db: Db):
    require_refs(
        db, (Competition, body.competition_id), (Team, body.home_team_id), (Team, body.away_team_id),
        (Fighter, body.fighter_a_id), (Fighter, body.fighter_b_id), (Fighter, body.winner_fighter_id),
    )
    return create_row(db, Event, body)


# --------------------------------------------------------------------------- simulation
simulate = APIRouter(prefix="/api/simulate", tags=["simulation"])


@lru_cache(maxsize=8)
def fit_dixon_coles(records: tuple[MatchRecord, ...]) -> DixonColesModel:
    """Keyed on the match data itself, so any new or edited result triggers a refit."""
    return DixonColesModel().fit(records)


def dixon_coles_prediction(db: Session, home: Team, away: Team, req: FootballSimulationRequest) -> FootballPrediction:
    """Fit Dixon-Coles on the home team's competition history; keep the request's rho / lam3 / max_goals."""
    model = fit_dixon_coles(tuple(match_records(db, home.competition_id)))
    lam, mu = model.expected_goals(home.name, away.name)
    lam, mu = apply_adjustments(lam, mu, MatchAdjustments(**req.adjustments.model_dump()))
    rho = float(np.clip(req.rho, *_rho_bounds(lam, mu)))
    lam3 = min(req.lam3, 0.5 * min(lam, mu))
    matrix = dixon_coles_matrix(lam, mu, rho, lam3, req.max_goals)
    elo_home, elo_away = model.elo.get(home.name), model.elo.get(away.name)
    return FootballPrediction(
        home=home.name, away=away.name, lambda_home=lam, lambda_away=mu, rho=rho, lam3=lam3,
        elo_home=elo_home, elo_away=elo_away, elo_win_expectancy=model.elo.expected_score(elo_home, elo_away),
        matrix=matrix, markets=market_probabilities(matrix),
    )


@simulate.post("/football")
def simulate_football_match(req: FootballSimulationRequest, db: Db) -> dict:
    home, away = get_or_404(db, Team, req.home_team_id), get_or_404(db, Team, req.away_team_id)
    require_distinct(home.id, away.id, "home and away team")
    with engine_errors():
        if req.use_dixon_coles_fit:
            prediction = dixon_coles_prediction(db, home, away, req)
        else:
            prediction = predict_match(
                team_profile(home), team_profile(away), league_goals(home), req.rho, req.lam3, req.max_goals,
                adjustments=MatchAdjustments(**req.adjustments.model_dump()),
            )
        summary = simulate_football(prediction.matrix, req.n_simulations, req.seed).summary()
        bets, margin = None, None
        if req.odds:
            o, p = req.odds, prediction.markets
            bets = evaluate_market(
                [("home", p["home_win"], o.home), ("draw", p["draw"], o.draw), ("away", p["away_win"], o.away)],
                req.bankroll, req.kelly_multiplier,
            )
            margin = bookmaker_margin([o.home, o.draw, o.away])
    return save_and_respond(
        db, req, Sport.FOOTBALL, prediction.to_dict(), summary, bets, margin,
        home_team_id=home.id, away_team_id=away.id,
    )


@simulate.post("/fight")
def simulate_mma_fight(req: FightSimulationRequest, db: Db) -> dict:
    a, b = get_or_404(db, Fighter, req.fighter_a_id), get_or_404(db, Fighter, req.fighter_b_id)
    require_distinct(a.id, b.id, "fighter_a and fighter_b")
    with engine_errors():
        pa, pb = fighter_profile(a), fighter_profile(b)
        adj = req.adjustments  # form shifts Elo by 250 points per unit
        prediction = predict_fight(
            replace(pa, elo=pa.elo + 250 * adj.form_a), replace(pb, elo=pb.elo + 250 * adj.form_b), req.rounds
        )
        summary = simulate_fight(
            prediction.method_probs, prediction.finish_round_probs, req.rounds, req.n_simulations, req.seed
        ).summary()
        bets, margin = None, None
        if req.odds:
            o = req.odds
            bets = evaluate_market(
                [(a.name, prediction.win_prob_a, o.fighter_a), (b.name, prediction.win_prob_b, o.fighter_b)],
                req.bankroll, req.kelly_multiplier,
            )
            margin = bookmaker_margin([o.fighter_a, o.fighter_b])
    return save_and_respond(
        db, req, Sport.MMA, prediction.to_dict(), summary, bets, margin, fighter_a_id=a.id, fighter_b_id=b.id
    )


simulations = APIRouter(prefix="/api/simulations", tags=["simulation"])


@simulations.get("", response_model=list[SimulationRead])
def list_simulations(db: Db, sport: Sport | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50):
    query = select(Simulation).order_by(Simulation.created_at.desc(), Simulation.id.desc()).limit(limit)
    if sport is not None:
        query = query.where(Simulation.sport == sport)
    return db.scalars(query).all()


@simulations.get("/{simulation_id}", response_model=SimulationRead)
def read_simulation(simulation_id: int, db: Db):
    return get_or_404(db, Simulation, simulation_id)


@simulations.delete("/{simulation_id}", status_code=204)
def delete_simulation(simulation_id: int, db: Db) -> None:
    db.delete(get_or_404(db, Simulation, simulation_id))
    db.commit()


# --------------------------------------------------------------------------- head to head
h2h = APIRouter(prefix="/api/h2h", tags=["head-to-head"])

TEAM_METRICS = (
    "elo", "xg_for_avg", "xg_against_avg", "goals_for_avg", "goals_against_avg", "possession_avg",
    "shots_per_match", "shots_on_target_per_match", "ppda", "defensive_rating", "clean_sheet_rate",
)
FIGHTER_METRICS = (
    "age", "height_cm", "reach_cm", "wins", "losses", "slpm", "sapm", "striking_accuracy", "striking_defense",
    "td_avg", "td_accuracy", "td_defense", "sub_avg", "cardio", "elo",
)


def differentials(a, b, metrics: tuple[str, ...]) -> dict:
    """a minus b for each metric; None when either side is unknown."""
    out = {}
    for m in metrics:
        va, vb = getattr(a, m), getattr(b, m)
        out[m] = None if va is None or vb is None else va - vb
    return out


@h2h.get("/football")
def h2h_football(team_a_id: int, team_b_id: int, db: Db) -> dict:
    a, b = get_or_404(db, Team, team_a_id), get_or_404(db, Team, team_b_id)
    require_distinct(a.id, b.id, "team_a and team_b")
    meetings = db.scalars(
        select(Event)
        .where(
            Event.sport == Sport.FOOTBALL,
            Event.status == EventStatus.FINISHED,
            or_(
                (Event.home_team_id == a.id) & (Event.away_team_id == b.id),
                (Event.home_team_id == b.id) & (Event.away_team_id == a.id),
            ),
        )
        .order_by(Event.event_date.desc(), Event.id.desc())
    ).all()

    history, summary = [], {"matches": len(meetings), "team_a_wins": 0, "draws": 0, "team_b_wins": 0,
                            "goals_team_a": 0, "goals_team_b": 0}
    for e in meetings:
        goals_a, goals_b = (e.home_goals, e.away_goals) if e.home_team_id == a.id else (e.away_goals, e.home_goals)
        summary["goals_team_a"] += goals_a
        summary["goals_team_b"] += goals_b
        summary["team_a_wins" if goals_a > goals_b else "draws" if goals_a == goals_b else "team_b_wins"] += 1
        history.append({
            "event_id": e.id, "date": e.event_date, "home": a.name if e.home_team_id == a.id else b.name,
            "away": b.name if e.home_team_id == a.id else a.name, "home_goals": e.home_goals,
            "away_goals": e.away_goals,
        })

    league = league_goals(a)
    with engine_errors():
        prediction = {
            "team_a_home": predict_match(team_profile(a), team_profile(b), league).to_dict(),
            "team_b_home": predict_match(team_profile(b), team_profile(a), league).to_dict(),
        }
    return to_jsonable({
        "team_a": TeamRead.model_validate(a).model_dump(mode="json"),
        "team_b": TeamRead.model_validate(b).model_dump(mode="json"),
        "differentials": differentials(a, b, TEAM_METRICS),
        "meetings": history,
        "summary": summary,
        "prediction": prediction,
    })


@h2h.get("/mma")
def h2h_mma(fighter_a_id: int, fighter_b_id: int, db: Db) -> dict:
    a, b = get_or_404(db, Fighter, fighter_a_id), get_or_404(db, Fighter, fighter_b_id)
    require_distinct(a.id, b.id, "fighter_a and fighter_b")
    pa, pb = fighter_profile(a), fighter_profile(b)
    fights = db.scalars(
        select(Event)
        .where(
            Event.sport == Sport.MMA,
            Event.status == EventStatus.FINISHED,
            or_(
                (Event.fighter_a_id == a.id) & (Event.fighter_b_id == b.id),
                (Event.fighter_a_id == b.id) & (Event.fighter_b_id == a.id),
            ),
        )
        .order_by(Event.event_date.desc(), Event.id.desc())
    ).all()
    previous = [
        {
            "event_id": e.id, "date": e.event_date, "winner": e.winner.name if e.winner else None,
            "method": e.method, "end_round": e.end_round, "scheduled_rounds": e.scheduled_rounds,
        }
        for e in fights
    ]
    with engine_errors():
        prediction = {
            "three_rounds": predict_fight(pa, pb, 3).to_dict(),
            "five_rounds": predict_fight(pa, pb, 5).to_dict(),
        }
    return to_jsonable({
        "fighter_a": FighterRead.model_validate(a).model_dump(mode="json"),
        "fighter_b": FighterRead.model_validate(b).model_dump(mode="json"),
        "styles": {"fighter_a": classify_style(pa), "fighter_b": classify_style(pb)},
        "differentials": differentials(a, b, FIGHTER_METRICS),
        "previous_fights": previous,
        "prediction": prediction,
    })


# --------------------------------------------------------------------------- backtest
backtest = APIRouter(prefix="/api/backtest", tags=["backtest"])


@backtest.post("/football")
def backtest_football(req: BacktestRequest, db: Db) -> dict:
    records = match_records(db, req.competition_id)
    if len(records) <= req.warmup:
        raise HTTPException(422, f"need more than {req.warmup} finished matches, found {len(records)}")
    with engine_errors():
        rows = walk_forward_predictions(records, req.warmup)
        report = run_backtest(
            [
                {
                    "probs": {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]},
                    "odds": {"H": r["odds_home"], "D": r["odds_draw"], "A": r["odds_away"]},
                    "outcome": r["outcome"],
                }
                for r in rows
            ],
            ("H", "D", "A"), req.bankroll, req.kelly_multiplier, req.min_ev,
        )
    return to_jsonable({**report.to_dict(), "n_matches": len(records), "warmup": req.warmup, "predictions": rows})


# --------------------------------------------------------------------------- export
export = APIRouter(prefix="/api/export", tags=["export"])


class Dataset(StrEnum):
    TEAMS = "teams"
    FIGHTERS = "fighters"
    EVENTS = "events"
    SIMULATIONS = "simulations"


DATASETS = {
    Dataset.TEAMS: (Team, TeamRead),
    Dataset.FIGHTERS: (Fighter, FighterRead),
    Dataset.EVENTS: (Event, EventRead),
    Dataset.SIMULATIONS: (Simulation, SimulationRead),
}


@export.get("/{dataset}", response_model=None)
def export_dataset(dataset: Dataset, db: Db, format: Literal["csv", "json"] = "csv"):
    model, schema = DATASETS[dataset]
    rows = [schema.model_validate(r).model_dump(mode="json") for r in db.scalars(select(model).order_by(model.id))]
    if format == "json":
        return rows
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(schema.model_fields))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="quantedge_{dataset.value}.csv"'},
    )


# --------------------------------------------------------------------------- admin
admin = APIRouter(prefix="/api/admin", tags=["admin"])


@admin.post("/seed")
def admin_seed(db: Db, force: bool = False) -> dict:
    return seed_database(db, force)


for router in (health_router, catalogue, simulate, simulations, h2h, backtest, export, admin):
    app.include_router(router)
