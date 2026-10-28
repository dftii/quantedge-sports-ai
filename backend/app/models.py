"""SQLAlchemy 2.0 ORM models for QuantEdge Sports AI."""
import enum
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Sport(enum.StrEnum):
    FOOTBALL = "football"
    MMA = "mma"


class EventStatus(enum.StrEnum):
    SCHEDULED = "scheduled"
    FINISHED = "finished"


class FightMethod(enum.StrEnum):
    KO_TKO = "ko_tko"
    SUBMISSION = "submission"
    DECISION = "decision"
    DRAW = "draw"
    NO_CONTEST = "no_contest"


def _enum(cls: type[enum.Enum]) -> Enum:
    """Store enum values (not member names) as plain VARCHAR."""
    return Enum(
        cls,
        values_callable=lambda e: [m.value for m in e],
        native_enum=False,
        create_constraint=False,
        length=20,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Competition(Base):
    __tablename__ = "competitions"
    __table_args__ = (
        UniqueConstraint("name", "season", name="uq_competition_name_season"),
        CheckConstraint("avg_goals_per_match > 0", name="ck_competition_avg_goals"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    sport: Mapped[Sport] = mapped_column(_enum(Sport), index=True)
    country: Mapped[str | None] = mapped_column(String(80))
    season: Mapped[str] = mapped_column(String(20))
    avg_goals_per_match: Mapped[float] = mapped_column(default=2.7)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    teams: Mapped[list["Team"]] = relationship(back_populates="competition")
    fighters: Mapped[list["Fighter"]] = relationship(back_populates="competition")
    events: Mapped[list["Event"]] = relationship(back_populates="competition")


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (
        CheckConstraint("matches_played >= 0", name="ck_team_matches_played"),
        CheckConstraint("elo > 0", name="ck_team_elo"),
        CheckConstraint(
            "xg_for_avg >= 0 AND xg_against_avg >= 0 AND goals_for_avg >= 0 AND goals_against_avg >= 0",
            name="ck_team_goal_stats",
        ),
        CheckConstraint("possession_avg >= 0 AND possession_avg <= 100", name="ck_team_possession"),
        CheckConstraint("shots_per_match >= 0 AND shots_on_target_per_match >= 0", name="ck_team_shots"),
        CheckConstraint("ppda > 0", name="ck_team_ppda"),
        CheckConstraint("defensive_rating > 0", name="ck_team_defensive_rating"),
        CheckConstraint("clean_sheet_rate >= 0 AND clean_sheet_rate <= 1", name="ck_team_clean_sheet_rate"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    short_name: Mapped[str] = mapped_column(String(10))
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    elo: Mapped[float] = mapped_column(default=1500.0)
    matches_played: Mapped[int] = mapped_column(default=0)
    xg_for_avg: Mapped[float] = mapped_column(default=1.35)
    xg_against_avg: Mapped[float] = mapped_column(default=1.35)
    goals_for_avg: Mapped[float] = mapped_column(default=1.35)
    goals_against_avg: Mapped[float] = mapped_column(default=1.35)
    possession_avg: Mapped[float] = mapped_column(default=50.0)
    shots_per_match: Mapped[float] = mapped_column(default=12.0)
    shots_on_target_per_match: Mapped[float] = mapped_column(default=4.0)
    ppda: Mapped[float] = mapped_column(default=10.0)  # passes per defensive action; lower = more pressing
    defensive_rating: Mapped[float] = mapped_column(default=1.0)  # >1 means better than league-average defence
    clean_sheet_rate: Mapped[float] = mapped_column(default=0.25)
    form: Mapped[str] = mapped_column(String(10), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    competition: Mapped[Competition] = relationship(back_populates="teams")


class Fighter(Base):
    __tablename__ = "fighters"
    __table_args__ = (
        CheckConstraint("elo > 0", name="ck_fighter_elo"),
        CheckConstraint("age IS NULL OR (age >= 18 AND age <= 60)", name="ck_fighter_age"),
        CheckConstraint("height_cm IS NULL OR height_cm > 0", name="ck_fighter_height"),
        CheckConstraint("reach_cm IS NULL OR reach_cm > 0", name="ck_fighter_reach"),
        CheckConstraint(
            "wins >= 0 AND losses >= 0 AND draws >= 0 AND ko_wins >= 0 AND sub_wins >= 0 AND dec_wins >= 0",
            name="ck_fighter_record",
        ),
        CheckConstraint("ko_wins + sub_wins + dec_wins <= wins", name="ck_fighter_win_methods"),
        CheckConstraint("slpm >= 0 AND sapm >= 0 AND td_avg >= 0 AND sub_avg >= 0", name="ck_fighter_rates"),
        CheckConstraint(
            "striking_accuracy BETWEEN 0 AND 1 AND striking_defense BETWEEN 0 AND 1 "
            "AND td_accuracy BETWEEN 0 AND 1 AND td_defense BETWEEN 0 AND 1",
            name="ck_fighter_percentages",
        ),
        CheckConstraint("cardio BETWEEN 0 AND 1", name="ck_fighter_cardio"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    nickname: Mapped[str | None] = mapped_column(String(80))
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"), index=True)
    weight_class: Mapped[str] = mapped_column(String(40), index=True)
    stance: Mapped[str | None] = mapped_column(String(20))
    age: Mapped[int | None] = mapped_column()
    height_cm: Mapped[float | None] = mapped_column()
    reach_cm: Mapped[float | None] = mapped_column()
    wins: Mapped[int] = mapped_column(default=0)
    losses: Mapped[int] = mapped_column(default=0)
    draws: Mapped[int] = mapped_column(default=0)
    ko_wins: Mapped[int] = mapped_column(default=0)
    sub_wins: Mapped[int] = mapped_column(default=0)
    dec_wins: Mapped[int] = mapped_column(default=0)
    slpm: Mapped[float] = mapped_column(default=3.5)  # significant strikes landed per minute
    sapm: Mapped[float] = mapped_column(default=3.5)  # significant strikes absorbed per minute
    striking_accuracy: Mapped[float] = mapped_column(default=0.45)
    striking_defense: Mapped[float] = mapped_column(default=0.55)
    td_avg: Mapped[float] = mapped_column(default=1.0)  # takedowns per 15 minutes
    td_accuracy: Mapped[float] = mapped_column(default=0.40)
    td_defense: Mapped[float] = mapped_column(default=0.65)
    sub_avg: Mapped[float] = mapped_column(default=0.5)  # submission attempts per 15 minutes
    cardio: Mapped[float] = mapped_column(default=0.7)
    elo: Mapped[float] = mapped_column(default=1500.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    competition: Mapped[Competition | None] = relationship(back_populates="fighters")


class Event(Base):
    """A football match or an MMA fight, historical or scheduled."""

    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(
            "(sport = 'football' AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL) "
            "OR (sport = 'mma' AND fighter_a_id IS NOT NULL AND fighter_b_id IS NOT NULL)",
            name="ck_event_participants",
        ),
        CheckConstraint("home_team_id IS NULL OR home_team_id != away_team_id", name="ck_event_distinct_teams"),
        CheckConstraint("fighter_a_id IS NULL OR fighter_a_id != fighter_b_id", name="ck_event_distinct_fighters"),
        CheckConstraint("home_goals IS NULL OR home_goals >= 0", name="ck_event_home_goals"),
        CheckConstraint("away_goals IS NULL OR away_goals >= 0", name="ck_event_away_goals"),
        CheckConstraint("home_xg IS NULL OR home_xg >= 0", name="ck_event_home_xg"),
        CheckConstraint("away_xg IS NULL OR away_xg >= 0", name="ck_event_away_xg"),
        CheckConstraint(
            "home_possession IS NULL OR (home_possession >= 0 AND home_possession <= 100)",
            name="ck_event_possession",
        ),
        CheckConstraint("scheduled_rounds IN (3, 5)", name="ck_event_scheduled_rounds"),
        CheckConstraint("end_round IS NULL OR (end_round >= 1 AND end_round <= 5)", name="ck_event_end_round"),
        CheckConstraint(
            "(odds_home IS NULL OR odds_home > 1) AND (odds_draw IS NULL OR odds_draw > 1) "
            "AND (odds_away IS NULL OR odds_away > 1) AND (odds_fighter_a IS NULL OR odds_fighter_a > 1) "
            "AND (odds_fighter_b IS NULL OR odds_fighter_b > 1)",
            name="ck_event_odds",
        ),
        Index("ix_events_competition_date", "competition_id", "event_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[Sport] = mapped_column(_enum(Sport), index=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    event_date: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[EventStatus] = mapped_column(_enum(EventStatus), default=EventStatus.SCHEDULED, index=True)

    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    fighter_a_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"), index=True)
    fighter_b_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"), index=True)

    home_goals: Mapped[int | None] = mapped_column()
    away_goals: Mapped[int | None] = mapped_column()
    home_xg: Mapped[float | None] = mapped_column()
    away_xg: Mapped[float | None] = mapped_column()
    home_possession: Mapped[float | None] = mapped_column()

    winner_fighter_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"))
    method: Mapped[FightMethod | None] = mapped_column(_enum(FightMethod))
    end_round: Mapped[int | None] = mapped_column()
    scheduled_rounds: Mapped[int] = mapped_column(default=3)

    odds_home: Mapped[float | None] = mapped_column()
    odds_draw: Mapped[float | None] = mapped_column()
    odds_away: Mapped[float | None] = mapped_column()
    odds_fighter_a: Mapped[float | None] = mapped_column()
    odds_fighter_b: Mapped[float | None] = mapped_column()

    competition: Mapped[Competition] = relationship(back_populates="events")
    home_team: Mapped[Team | None] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team | None] = relationship(foreign_keys=[away_team_id])
    fighter_a: Mapped[Fighter | None] = relationship(foreign_keys=[fighter_a_id])
    fighter_b: Mapped[Fighter | None] = relationship(foreign_keys=[fighter_b_id])
    winner: Mapped[Fighter | None] = relationship(foreign_keys=[winner_fighter_id])


class Simulation(Base):
    """A saved simulation run (inputs and results as JSON)."""

    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[Sport] = mapped_column(_enum(Sport), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    label: Mapped[str | None] = mapped_column(String(200))
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    fighter_a_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"))
    fighter_b_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"))
    n_simulations: Mapped[int] = mapped_column()
    seed: Mapped[int | None] = mapped_column()
    model_version: Mapped[str] = mapped_column(String(40))
    input_params: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    value_bets: Mapped[list | None] = mapped_column(JSON)

    home_team: Mapped[Team | None] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team | None] = relationship(foreign_keys=[away_team_id])
    fighter_a: Mapped[Fighter | None] = relationship(foreign_keys=[fighter_a_id])
    fighter_b: Mapped[Fighter | None] = relationship(foreign_keys=[fighter_b_id])
