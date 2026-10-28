"""Pydantic v2 schemas: CRUD payloads and simulation / backtest requests."""
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models import EventStatus, FightMethod, Sport

Rate = Annotated[float, Field(ge=0, le=1)]
NonNeg = Annotated[float, Field(ge=0)]
Positive = Annotated[float, Field(gt=0)]
Count = Annotated[int, Field(ge=0)]
DecimalOdds = Annotated[float, Field(gt=1)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Competition -----------------------------------------------------------

class CompetitionBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    sport: Sport
    country: str | None = Field(default=None, max_length=80)
    season: str = Field(min_length=1, max_length=20)
    avg_goals_per_match: Positive = 2.7


class CompetitionCreate(CompetitionBase):
    pass


class CompetitionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    sport: Sport | None = None
    country: str | None = Field(default=None, max_length=80)
    season: str | None = Field(default=None, min_length=1, max_length=20)
    avg_goals_per_match: Positive | None = None


class CompetitionRead(CompetitionBase, ORMModel):
    id: int
    created_at: datetime


# --- Team ------------------------------------------------------------------

class TeamBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    short_name: str = Field(min_length=1, max_length=10)
    competition_id: int
    elo: Positive = 1500.0
    matches_played: Count = 0
    xg_for_avg: NonNeg = 1.35
    xg_against_avg: NonNeg = 1.35
    goals_for_avg: NonNeg = 1.35
    goals_against_avg: NonNeg = 1.35
    possession_avg: float = Field(default=50.0, ge=0, le=100)
    shots_per_match: NonNeg = 12.0
    shots_on_target_per_match: NonNeg = 4.0
    ppda: Positive = 10.0
    defensive_rating: Positive = 1.0
    clean_sheet_rate: Rate = 0.25
    form: str = Field(default="", max_length=10, pattern=r"^[WDL]*$")


class TeamCreate(TeamBase):
    pass


class TeamUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    short_name: str | None = Field(default=None, min_length=1, max_length=10)
    competition_id: int | None = None
    elo: Positive | None = None
    matches_played: Count | None = None
    xg_for_avg: NonNeg | None = None
    xg_against_avg: NonNeg | None = None
    goals_for_avg: NonNeg | None = None
    goals_against_avg: NonNeg | None = None
    possession_avg: float | None = Field(default=None, ge=0, le=100)
    shots_per_match: NonNeg | None = None
    shots_on_target_per_match: NonNeg | None = None
    ppda: Positive | None = None
    defensive_rating: Positive | None = None
    clean_sheet_rate: Rate | None = None
    form: str | None = Field(default=None, max_length=10, pattern=r"^[WDL]*$")


class TeamRead(TeamBase, ORMModel):
    id: int
    updated_at: datetime


# --- Fighter ---------------------------------------------------------------

class FighterBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    nickname: str | None = Field(default=None, max_length=80)
    competition_id: int | None = None
    weight_class: str = Field(min_length=1, max_length=40)
    stance: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=18, le=60)
    height_cm: Positive | None = None
    reach_cm: Positive | None = None
    wins: Count = 0
    losses: Count = 0
    draws: Count = 0
    ko_wins: Count = 0
    sub_wins: Count = 0
    dec_wins: Count = 0
    slpm: NonNeg = 3.5
    sapm: NonNeg = 3.5
    striking_accuracy: Rate = 0.45
    striking_defense: Rate = 0.55
    td_avg: NonNeg = 1.0
    td_accuracy: Rate = 0.40
    td_defense: Rate = 0.65
    sub_avg: NonNeg = 0.5
    cardio: Rate = 0.7
    elo: Positive = 1500.0

    @model_validator(mode="after")
    def _methods_within_wins(self):
        if self.ko_wins + self.sub_wins + self.dec_wins > self.wins:
            raise ValueError("ko_wins + sub_wins + dec_wins cannot exceed wins")
        return self


class FighterCreate(FighterBase):
    pass


class FighterUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    nickname: str | None = Field(default=None, max_length=80)
    competition_id: int | None = None
    weight_class: str | None = Field(default=None, min_length=1, max_length=40)
    stance: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=18, le=60)
    height_cm: Positive | None = None
    reach_cm: Positive | None = None
    wins: Count | None = None
    losses: Count | None = None
    draws: Count | None = None
    ko_wins: Count | None = None
    sub_wins: Count | None = None
    dec_wins: Count | None = None
    slpm: NonNeg | None = None
    sapm: NonNeg | None = None
    striking_accuracy: Rate | None = None
    striking_defense: Rate | None = None
    td_avg: NonNeg | None = None
    td_accuracy: Rate | None = None
    td_defense: Rate | None = None
    sub_avg: NonNeg | None = None
    cardio: Rate | None = None
    elo: Positive | None = None


class FighterRead(FighterBase, ORMModel):
    id: int
    updated_at: datetime


# --- Event -----------------------------------------------------------------

class EventBase(BaseModel):
    sport: Sport
    competition_id: int
    event_date: date
    status: EventStatus = EventStatus.SCHEDULED
    home_team_id: int | None = None
    away_team_id: int | None = None
    fighter_a_id: int | None = None
    fighter_b_id: int | None = None
    home_goals: Count | None = None
    away_goals: Count | None = None
    home_xg: NonNeg | None = None
    away_xg: NonNeg | None = None
    home_possession: float | None = Field(default=None, ge=0, le=100)
    winner_fighter_id: int | None = None
    method: FightMethod | None = None
    end_round: int | None = Field(default=None, ge=1, le=5)
    scheduled_rounds: Literal[3, 5] = 3
    odds_home: DecimalOdds | None = None
    odds_draw: DecimalOdds | None = None
    odds_away: DecimalOdds | None = None
    odds_fighter_a: DecimalOdds | None = None
    odds_fighter_b: DecimalOdds | None = None


class EventCreate(EventBase):
    @model_validator(mode="after")
    def _participants_match_sport(self):
        if self.sport == Sport.FOOTBALL:
            if self.home_team_id is None or self.away_team_id is None:
                raise ValueError("football events require home_team_id and away_team_id")
            if self.home_team_id == self.away_team_id:
                raise ValueError("home and away team must differ")
        else:
            if self.fighter_a_id is None or self.fighter_b_id is None:
                raise ValueError("mma events require fighter_a_id and fighter_b_id")
            if self.fighter_a_id == self.fighter_b_id:
                raise ValueError("fighter_a and fighter_b must differ")
        return self


class EventUpdate(BaseModel):
    competition_id: int | None = None
    event_date: date | None = None
    status: EventStatus | None = None
    home_goals: Count | None = None
    away_goals: Count | None = None
    home_xg: NonNeg | None = None
    away_xg: NonNeg | None = None
    home_possession: float | None = Field(default=None, ge=0, le=100)
    winner_fighter_id: int | None = None
    method: FightMethod | None = None
    end_round: int | None = Field(default=None, ge=1, le=5)
    scheduled_rounds: Literal[3, 5] | None = None
    odds_home: DecimalOdds | None = None
    odds_draw: DecimalOdds | None = None
    odds_away: DecimalOdds | None = None
    odds_fighter_a: DecimalOdds | None = None
    odds_fighter_b: DecimalOdds | None = None


class EventRead(EventBase, ORMModel):
    id: int


# --- Simulation (saved) ----------------------------------------------------

class SimulationRead(ORMModel):
    id: int
    sport: Sport
    created_at: datetime
    label: str | None
    home_team_id: int | None
    away_team_id: int | None
    fighter_a_id: int | None
    fighter_b_id: int | None
    n_simulations: int
    seed: int | None
    model_version: str
    input_params: dict
    result: dict
    value_bets: list | None


# --- API requests ----------------------------------------------------------

class OddsInput1X2(BaseModel):
    home: DecimalOdds
    draw: DecimalOdds
    away: DecimalOdds


class OddsInputFight(BaseModel):
    fighter_a: DecimalOdds
    fighter_b: DecimalOdds


class FootballAdjustmentsIn(BaseModel):
    home_advantage: float = Field(default=1.0, ge=0.8, le=1.3)
    home_form: float = Field(default=0.0, ge=-0.2, le=0.2)
    away_form: float = Field(default=0.0, ge=-0.2, le=0.2)
    home_absences: int = Field(default=0, ge=0, le=5)
    away_absences: int = Field(default=0, ge=0, le=5)


class FightAdjustmentsIn(BaseModel):
    form_a: float = Field(default=0.0, ge=-0.2, le=0.2)
    form_b: float = Field(default=0.0, ge=-0.2, le=0.2)


class FootballSimulationRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    n_simulations: int = Field(default=10_000, ge=1, le=1_000_000)
    seed: int | None = None
    rho: float = Field(default=-0.08, ge=-0.3, le=0.3)
    lam3: float = Field(default=0.0, ge=0, le=1)
    max_goals: int = Field(default=10, ge=4, le=15)
    use_dixon_coles_fit: bool = False
    adjustments: FootballAdjustmentsIn = Field(default_factory=FootballAdjustmentsIn)
    odds: OddsInput1X2 | None = None
    bankroll: Positive = 1000
    kelly_multiplier: float = Field(default=0.25, gt=0, le=1)
    save: bool = True
    label: str | None = Field(default=None, max_length=200)


class FightSimulationRequest(BaseModel):
    fighter_a_id: int
    fighter_b_id: int
    rounds: Literal[3, 5] = 3
    adjustments: FightAdjustmentsIn = Field(default_factory=FightAdjustmentsIn)
    n_simulations: int = Field(default=10_000, ge=1, le=1_000_000)
    seed: int | None = None
    odds: OddsInputFight | None = None
    bankroll: Positive = 1000
    kelly_multiplier: float = Field(default=0.25, gt=0, le=1)
    save: bool = True
    label: str | None = Field(default=None, max_length=200)


class BacktestRequest(BaseModel):
    competition_id: int | None = None
    warmup: int = Field(default=30, ge=5)
    bankroll: Positive = 1000
    kelly_multiplier: float = Field(default=0.25, gt=0, le=1)
    min_ev: float = Field(default=0.02, ge=0, le=1)


class ValueBetOut(BaseModel):
    selection: str
    model_prob: float
    decimal_odds: float
    implied_prob: float
    fair_prob: float
    edge: float
    expected_value: float
    kelly_fraction: float
    stake: float
    is_value: bool
