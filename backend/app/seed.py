"""Deterministic demo dataset: real club and fighter NAMES, approximate real STATS, SIMULATED history.

What is real and what is not:
* Entity names (clubs, fighters, nicknames, weight classes) are real.
* Per-team and per-fighter statistics (CLUBS / FIGHTERS tables below) are approximate values transcribed
  from public sources (FBref / Understat-style 2023/24 domestic-league season data, UFCStats career
  stats around mid-2024). They were typed in by hand and may contain inaccuracies.
* The historical matches and fights, their scores, methods and bookmaker odds are SIMULATED (seeded NumPy
  RNG, default_rng(42)) from those statistics. These specific fixtures did not happen, and the odds are
  not real market prices. The scheduled fixtures are hypothetical as well.
* This project is not affiliated with, endorsed by or connected to any club, league or the UFC.

How the simulated history is built:
* Football: 150 matches among the 20 clubs (15 rounds of a circle-method round robin, so each club plays
  15 opponents once, home/away balanced as far as possible). Goals are Poisson with
      home = L * attack_home * weakness_away * H,   away = L * attack_away * weakness_home / H,
  attack = xG for / L, weakness = xG against / L, L = LEAGUE_XG (average xG per team per match in the big-five
  leagues) and H = 1.10. Defence enters only through the opponent's xG against (no extra defensive factor).
  Stats are not adjusted for league strength, so clubs from smaller leagues look better than they are.
* MMA: 40 fights inside the two weight classes; win probability is a logistic of the starting-Elo difference,
  the method follows the winner's finishing profile (KO / submission / decision shares of career wins).
  Simulated results only update Elo; career records and rate stats stay as transcribed.
* Odds are the generating probabilities with a bookmaker margin (5 % football, 4.5 % MMA) and small noise.

Derived fields: team defensive_rating = LEAGUE_XG / xG against (descriptive only); fighter starting Elo =
1500 + 12 * (wins - losses); fighter cardio = 0.60 + 0.30 * (decision share of wins) +
0.10 * min(1, fights / 40), i.e. fighters who go the distance often and have more experience get more.

Run from backend/:  python -m app.seed
"""
import math
from collections import Counter, deque
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import (
    Competition,
    Event,
    EventStatus,
    Fighter,
    FightMethod,
    Simulation,
    Sport,
    Team,
)

SEED = 42
MAX_GOALS = 10
FOOTBALL_MARGIN = 0.05
MMA_MARGIN = 0.045
LEAGUE_XG = 1.45          # average xG per team per match, big-five leagues 2023/24
HOME_ADVANTAGE = 1.10
HISTORY_ROUNDS = 15       # 15 rounds x 10 matches = 150 matches, 15 per club
FOOTBALL_START = date(2025, 2, 15)  # weekly rounds; the last one is 2025-05-24
MMA_START = date(2024, 1, 6)
MMA_FIGHTS = 40
MMA_SPACING_DAYS = 8


class Club(NamedTuple):
    name: str
    short: str
    elo: float            # clubelo.com-style rating, approximate
    xg_for: float         # per match, domestic league 2023/24
    xg_against: float
    goals_for: float
    goals_against: float
    possession: float     # %
    shots: float
    shots_on_target: float
    ppda: float           # opponent passes per defensive action (lower = more pressing)
    clean_sheet_rate: float


CLUBS = [
    #    name                    short  elo   xGF   xGA   GF    GA    poss  shots sot  ppda  CS
    Club("Real Madrid",          "RMA", 2000, 2.00, 0.90, 2.29, 0.68, 58.0, 16.0, 6.0, 9.8, 0.42),
    Club("FC Barcelona",         "BAR", 1920, 1.85, 1.20, 2.08, 1.42, 64.0, 15.5, 5.6, 8.5, 0.29),
    Club("Atlético de Madrid",   "ATM", 1860, 1.70, 1.20, 1.84, 1.13, 51.0, 14.0, 5.0, 10.8, 0.34),
    Club("Manchester City",      "MCI", 2050, 2.20, 0.95, 2.53, 0.89, 66.0, 17.5, 6.8, 8.0, 0.47),
    Club("Arsenal",              "ARS", 2000, 2.10, 0.82, 2.39, 0.76, 60.0, 17.5, 6.3, 9.0, 0.47),
    Club("Liverpool",            "LIV", 1950, 2.25, 1.05, 2.26, 1.08, 60.0, 18.5, 6.6, 9.2, 0.34),
    Club("Chelsea",              "CHE", 1830, 1.90, 1.45, 2.03, 1.66, 56.0, 15.5, 5.5, 10.0, 0.21),
    Club("Aston Villa",          "AVL", 1820, 1.75, 1.45, 2.00, 1.61, 52.0, 14.5, 5.0, 10.8, 0.26),
    Club("Bayern München",       "BAY", 1940, 2.45, 1.20, 2.76, 1.32, 63.0, 18.0, 6.8, 8.5, 0.26),
    Club("Bayer Leverkusen",     "B04", 1950, 2.20, 0.95, 2.62, 0.71, 62.0, 17.0, 6.1, 9.0, 0.47),
    Club("Borussia Dortmund",    "BVB", 1830, 1.90, 1.35, 2.00, 1.26, 56.0, 15.5, 5.5, 10.5, 0.29),
    Club("Inter",                "INT", 1970, 2.05, 0.85, 2.34, 0.58, 55.0, 15.8, 6.0, 10.5, 0.55),
    Club("AC Milan",             "MIL", 1840, 1.80, 1.20, 2.00, 1.29, 55.0, 15.0, 5.3, 10.2, 0.29),
    Club("Juventus",             "JUV", 1850, 1.45, 1.00, 1.53, 0.92, 54.0, 13.5, 4.5, 11.5, 0.45),
    Club("Napoli",               "NAP", 1800, 1.55, 1.25, 1.45, 1.26, 55.0, 13.5, 4.6, 10.5, 0.29),
    Club("Atalanta",             "ATA", 1870, 1.85, 1.20, 1.89, 0.97, 52.0, 15.5, 5.2, 9.5, 0.37),
    Club("Paris Saint-Germain",  "PSG", 1900, 2.15, 1.05, 2.38, 1.18, 65.0, 17.0, 6.0, 9.0, 0.35),
    Club("SL Benfica",           "BEN", 1790, 2.00, 0.95, 2.20, 0.88, 62.0, 16.0, 6.0, 9.5, 0.47),
    Club("Sporting CP",          "SCP", 1800, 2.25, 0.90, 2.65, 0.88, 59.0, 16.0, 6.2, 9.8, 0.47),
    Club("PSV Eindhoven",        "PSV", 1790, 2.70, 0.95, 3.26, 0.82, 61.0, 17.5, 6.8, 9.0, 0.53),
]


class FighterRow(NamedTuple):
    weight_class: str
    name: str
    nickname: str | None
    stance: str
    age: int              # as of 2024
    height_cm: float
    reach_cm: float
    wins: int
    losses: int
    draws: int
    ko_wins: int
    sub_wins: int
    dec_wins: int
    slpm: float           # significant strikes landed per minute
    sapm: float           # significant strikes absorbed per minute
    striking_accuracy: float
    striking_defense: float
    td_avg: float         # takedowns per 15 minutes
    td_accuracy: float
    td_defense: float
    sub_avg: float        # submission attempts per 15 minutes


FIGHTERS = [
    # weight class, name, nickname, stance, age, height, reach, W, L, D, KO, SUB, DEC, SLpM, SApM, Str.Acc, Str.Def, TD avg, TD acc, TD def, Sub avg
    FighterRow("Lightweight", "Islam Makhachev", None, "Southpaw", 32, 178.0, 177.0, 26, 1, 0, 5, 11, 10,
               2.60, 1.85, 0.56, 0.62, 3.20, 0.58, 0.90, 1.0),
    FighterRow("Lightweight", "Charles Oliveira", "do Bronx", "Orthodox", 34, 178.0, 188.0, 34, 9, 0, 10, 21, 3,
               3.45, 3.35, 0.51, 0.53, 2.00, 0.38, 0.55, 2.6),
    FighterRow("Lightweight", "Arman Tsarukyan", "Ahalkalakets", "Orthodox", 27, 170.0, 183.0, 22, 3, 0, 8, 4, 10,
               3.60, 2.55, 0.47, 0.55, 2.70, 0.43, 0.81, 0.7),
    FighterRow("Lightweight", "Dustin Poirier", "The Diamond", "Southpaw", 35, 175.0, 183.0, 30, 9, 0, 14, 7, 9,
               5.55, 4.55, 0.50, 0.54, 0.70, 0.34, 0.68, 1.1),
    FighterRow("Lightweight", "Justin Gaethje", "The Highlight", "Orthodox", 35, 180.0, 178.0, 25, 4, 0, 20, 1, 4,
               7.30, 7.00, 0.55, 0.47, 0.30, 0.30, 0.75, 0.0),
    FighterRow("Lightweight", "Max Holloway", "Blessed", "Orthodox", 32, 180.0, 175.0, 26, 8, 0, 12, 3, 11,
               7.10, 4.60, 0.46, 0.59, 0.30, 0.38, 0.83, 0.3),
    FighterRow("Lightweight", "Dan Hooker", "The Hangman", "Orthodox", 34, 183.0, 193.0, 23, 12, 0, 12, 6, 5,
               4.65, 4.55, 0.45, 0.55, 0.50, 0.30, 0.66, 0.5),
    FighterRow("Lightweight", "Mateusz Gamrot", "Gamer", "Orthodox", 34, 178.0, 180.0, 24, 2, 0, 5, 6, 13,
               4.70, 3.70, 0.45, 0.53, 3.40, 0.45, 0.70, 0.4),
    FighterRow("Welterweight", "Belal Muhammad", "Remember the Name", "Orthodox", 36, 180.0, 183.0, 24, 3, 0, 6, 1, 17,
               4.40, 3.70, 0.50, 0.61, 1.70, 0.37, 0.83, 0.2),
    FighterRow("Welterweight", "Leon Edwards", "Rocky", "Orthodox", 32, 183.0, 188.0, 22, 3, 0, 8, 2, 12,
               3.90, 2.45, 0.46, 0.64, 1.15, 0.34, 0.73, 0.3),
    FighterRow("Welterweight", "Kamaru Usman", "The Nigerian Nightmare", "Orthodox", 37, 183.0, 193.0, 20, 4, 0, 8, 1, 11,
               4.45, 2.65, 0.50, 0.60, 3.10, 0.52, 0.93, 0.3),
    FighterRow("Welterweight", "Shavkat Rakhmonov", "Nomad", "Orthodox", 30, 185.0, 196.0, 18, 0, 0, 8, 9, 1,
               3.75, 1.85, 0.57, 0.70, 0.85, 0.55, 0.93, 2.2),
    FighterRow("Welterweight", "Jack Della Maddalena", "JDM", "Orthodox", 27, 180.0, 185.0, 17, 2, 0, 11, 2, 4,
               6.55, 3.35, 0.50, 0.64, 0.20, 0.33, 0.85, 0.2),
    FighterRow("Welterweight", "Sean Brady", "Wrecking Ball", "Orthodox", 31, 178.0, 184.0, 16, 0, 0, 3, 5, 8,
               4.25, 2.50, 0.50, 0.60, 3.00, 0.43, 0.75, 1.0),
    FighterRow("Welterweight", "Ian Machado Garry", "The Future", "Orthodox", 26, 188.0, 191.0, 15, 0, 0, 5, 1, 9,
               4.45, 3.25, 0.50, 0.62, 0.30, 0.38, 0.85, 0.3),
    FighterRow("Welterweight", "Gilbert Burns", "Durinho", "Orthodox", 38, 178.0, 188.0, 22, 7, 0, 7, 9, 6,
               4.30, 3.50, 0.50, 0.56, 1.90, 0.41, 0.73, 1.7),
]

METHOD_ENUMS = (FightMethod.KO_TKO, FightMethod.SUBMISSION, FightMethod.DECISION)

_FACTORIALS = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


def _r(x, digits: int = 3) -> float:
    return round(float(x), digits)


def _book_odds(probs: np.ndarray, margin: float, noise_sd: float, rng: np.random.Generator) -> list[float]:
    """Decimal odds whose implied probabilities sum to 1 + margin, with multiplicative noise."""
    noisy = probs * np.exp(rng.normal(0.0, noise_sd, probs.size))
    noisy /= noisy.sum()
    return [_r(max(1.01, 1.0 / (p * (1.0 + margin))), 2) for p in noisy]


def _probs_1x2(lam_home: float, lam_away: float) -> np.ndarray:
    """P(home, draw, away) from independent Poisson goals truncated at MAX_GOALS."""
    k = np.arange(MAX_GOALS + 1)
    home = np.exp(-lam_home) * lam_home**k / _FACTORIALS
    away = np.exp(-lam_away) * lam_away**k / _FACTORIALS
    m = np.outer(home, away)
    m /= m.sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _lambdas(home: Club, away: Club) -> tuple[float, float]:
    """Expected goals: attack x opponent weakness x home advantage (defence enters only via xG against)."""
    attack_h, weakness_h = home.xg_for / LEAGUE_XG, home.xg_against / LEAGUE_XG
    attack_a, weakness_a = away.xg_for / LEAGUE_XG, away.xg_against / LEAGUE_XG
    return (LEAGUE_XG * attack_h * weakness_a * HOME_ADVANTAGE,
            LEAGUE_XG * attack_a * weakness_h / HOME_ADVANTAGE)


def _history_fixtures(n: int) -> list[list[tuple[int, int]]]:
    """Last HISTORY_ROUNDS rounds of a circle-method round robin, home/away balanced by path reversal.

    Taking the last rounds (not the first) makes club 0 meet club 1 (Real Madrid v Barcelona).
    """
    ids, rounds = list(range(n)), []
    for _ in range(n - 1):
        rounds.append([(ids[i], ids[n - 1 - i]) for i in range(n // 2)])
        ids = [ids[0], ids[-1], *ids[1:-1]]
    rounds = rounds[-HISTORY_ROUNDS:]

    # Orientation (home, away) per game; reverse a home->away path from an over-hosting club to an
    # under-hosting one until every club's home count is within 1 of any other's.
    games = [list(g) for rnd in rounds for g in rnd]
    while True:
        home = Counter(g[0] for g in games)
        moved = False
        for start in sorted(range(n), key=lambda t: -home[t]):
            parent: dict[int, tuple[int, int]] = {start: (-1, -1)}  # club -> (game index, previous club)
            queue = deque([start])
            while queue and not moved:
                club = queue.popleft()
                for gi, g in enumerate(games):
                    if g[0] != club or g[1] in parent:
                        continue
                    parent[g[1]] = (gi, club)
                    if home[g[1]] <= home[start] - 2:
                        node = g[1]
                        while node != start:
                            gi, node = parent[node]
                            games[gi].reverse()
                        moved = True
                        break
                    queue.append(g[1])
            if moved:
                break
        if not moved:
            break
    it = iter(games)
    return [[tuple(next(it)) for _ in rnd] for rnd in rounds]


def _football_elo_update(r_home: float, r_away: float, hg: int, ag: int) -> tuple[float, float]:
    """Elo with k=20, home advantage 65 and a goal-difference multiplier."""
    expected = 1.0 / (1.0 + 10 ** (-(r_home - r_away + 65.0) / 400.0))
    score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    diff = abs(hg - ag)
    mult = 1.0 if diff <= 1 else 1.5 if diff == 2 else (11.0 + diff) / 8.0
    delta = 20.0 * mult * (score - expected)
    return r_home + delta, r_away - delta


def _seed_football(db: Session, rng: np.random.Generator) -> None:
    comp = Competition(name="European Elite Clubs", sport=Sport.FOOTBALL, country="Europe", season="2024/25")
    db.add(comp)
    db.flush()

    teams = [Team(name=c.name, short_name=c.short, competition_id=comp.id) for c in CLUBS]
    db.add_all(teams)
    db.flush()

    played: list[tuple[Event, int, int]] = []
    for rnd_idx, fixtures in enumerate(_history_fixtures(len(CLUBS))):
        match_date = FOOTBALL_START + timedelta(days=7 * rnd_idx)
        for h, a in fixtures:
            lh, la = _lambdas(CLUBS[h], CLUBS[a])
            hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
            hxg = lh * rng.gamma(20.0, 1 / 20.0)
            axg = la * rng.gamma(20.0, 1 / 20.0)
            share = 100 * CLUBS[h].possession / (CLUBS[h].possession + CLUBS[a].possession)
            poss = float(np.clip(share + 1.0 + rng.normal(0, 3), 25, 75))
            odds = _book_odds(_probs_1x2(lh, la), FOOTBALL_MARGIN, 0.03, rng)
            event = Event(sport=Sport.FOOTBALL, competition_id=comp.id, event_date=match_date,
                          status=EventStatus.FINISHED, home_team_id=teams[h].id, away_team_id=teams[a].id,
                          home_goals=hg, away_goals=ag, home_xg=_r(hxg), away_xg=_r(axg),
                          home_possession=_r(poss, 1), odds_home=odds[0], odds_draw=odds[1], odds_away=odds[2])
            played.append((event, h, a))

    # Four hypothetical upcoming fixtures: Real Madrid-Barcelona, Man City-Arsenal, Bayern-Leverkusen, Inter-Milan.
    upcoming = []
    for (h, a), when in (
        ((0, 1), date(2026, 11, 22)), ((3, 4), date(2026, 11, 29)),
        ((8, 9), date(2026, 12, 6)), ((11, 12), date(2026, 12, 13)),
    ):
        odds = _book_odds(_probs_1x2(*_lambdas(CLUBS[h], CLUBS[a])), FOOTBALL_MARGIN, 0.03, rng)
        upcoming.append(Event(sport=Sport.FOOTBALL, competition_id=comp.id, event_date=when,
                              status=EventStatus.SCHEDULED, home_team_id=teams[h].id, away_team_id=teams[a].id,
                              odds_home=odds[0], odds_draw=odds[1], odds_away=odds[2]))
    db.add_all([e for e, _, _ in played] + upcoming)

    # Sequential Elo from the hard-coded ratings, plus matches played and form, in chronological order.
    elo = [c.elo for c in CLUBS]
    results: list[list[str]] = [[] for _ in CLUBS]
    for event, h, a in played:
        hg, ag = event.home_goals, event.away_goals
        elo[h], elo[a] = _football_elo_update(elo[h], elo[a], hg, ag)
        results[h].append("W" if hg > ag else "D" if hg == ag else "L")
        results[a].append("W" if ag > hg else "D" if hg == ag else "L")

    for i, (team, c) in enumerate(zip(teams, CLUBS)):
        team.elo = _r(elo[i], 1)
        team.matches_played = len(results[i])
        team.form = "".join(results[i][-5:])
        team.xg_for_avg, team.xg_against_avg = c.xg_for, c.xg_against
        team.goals_for_avg, team.goals_against_avg = c.goals_for, c.goals_against
        team.possession_avg = c.possession
        team.shots_per_match, team.shots_on_target_per_match = c.shots, c.shots_on_target
        team.ppda = c.ppda
        team.defensive_rating = _r(LEAGUE_XG / c.xg_against)
        team.clean_sheet_rate = c.clean_sheet_rate
    comp.avg_goals_per_match = _r(np.mean([e.home_goals + e.away_goals for e, _, _ in played]))


def _cardio(f: FighterRow) -> float:
    return _r(min(0.98, 0.60 + 0.30 * f.dec_wins / f.wins + 0.10 * min(1.0, (f.wins + f.losses) / 40)), 2)


def _seed_mma(db: Session, rng: np.random.Generator) -> None:
    comp = Competition(name="UFC Lightweight & Welterweight", sport=Sport.MMA, country="International",
                       season="2024")
    db.add(comp)
    db.flush()

    start_elo = [1500.0 + 12 * (f.wins - f.losses) for f in FIGHTERS]
    fighters = [
        Fighter(
            name=f.name, nickname=f.nickname, competition_id=comp.id, weight_class=f.weight_class,
            stance=f.stance, age=f.age, height_cm=f.height_cm, reach_cm=f.reach_cm,
            wins=f.wins, losses=f.losses, draws=f.draws, ko_wins=f.ko_wins, sub_wins=f.sub_wins,
            dec_wins=f.dec_wins, slpm=f.slpm, sapm=f.sapm, striking_accuracy=f.striking_accuracy,
            striking_defense=f.striking_defense, td_avg=f.td_avg, td_accuracy=f.td_accuracy,
            td_defense=f.td_defense, sub_avg=f.sub_avg, cardio=_cardio(f), elo=start_elo[i],
        )
        for i, f in enumerate(FIGHTERS)
    ]
    db.add_all(fighters)
    db.flush()

    by_class = {wc: [i for i, f in enumerate(FIGHTERS) if f.weight_class == wc]
                for wc in ("Lightweight", "Welterweight")}
    classes = list(by_class)

    def win_prob(a: int, b: int) -> float:
        """Generating model: logistic of the (fixed) starting-Elo difference."""
        return 1.0 / (1.0 + 10 ** ((start_elo[b] - start_elo[a]) / 400.0))

    def fight_odds(a: int, b: int) -> list[float]:
        p = win_prob(a, b)
        return _book_odds(np.array([p, 1.0 - p]), MMA_MARGIN, 0.04, rng)

    def finishing_profile(i: int) -> np.ndarray:
        f = FIGHTERS[i]
        shares = np.array([f.ko_wins, f.sub_wins, f.dec_wins], dtype=float)
        return shares / shares.sum()

    elo = list(start_elo)  # only Elo changes during the simulated history; records stay as transcribed
    events: list[Event] = []
    for i in range(MMA_FIGHTS):
        a, b = (int(x) for x in rng.choice(by_class[classes[i % 2]], size=2, replace=False))
        rounds = 5 if i % 8 == 7 else 3
        odds = fight_odds(a, b)
        u = rng.random()
        if u < 0.02:
            winner, method, end_round = None, FightMethod.DRAW, rounds
        else:
            winner = a if u < 0.02 + 0.98 * win_prob(a, b) else b
            method = METHOD_ENUMS[int(rng.choice(3, p=finishing_profile(winner)))]
            end_round = rounds if method == FightMethod.DECISION else int(min(rounds, rng.geometric(0.45)))

        score = 0.5 if winner is None else 1.0 if winner == a else 0.0
        delta = 32.0 * (score - 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0)))
        elo[a] += delta
        elo[b] -= delta

        events.append(Event(
            sport=Sport.MMA, competition_id=comp.id, event_date=MMA_START + timedelta(days=MMA_SPACING_DAYS * i),
            status=EventStatus.FINISHED, fighter_a_id=fighters[a].id, fighter_b_id=fighters[b].id,
            winner_fighter_id=fighters[winner].id if winner is not None else None, method=method,
            end_round=end_round, scheduled_rounds=rounds, odds_fighter_a=odds[0], odds_fighter_b=odds[1],
        ))

    # Three hypothetical upcoming bouts: Makhachev-Tsarukyan, Oliveira-Gaethje, Muhammad-Rakhmonov.
    for (a, b), when, rounds in (
        ((0, 2), date(2026, 11, 7), 5),
        ((1, 4), date(2026, 11, 7), 3),
        ((8, 11), date(2026, 11, 14), 5),
    ):
        odds = fight_odds(a, b)
        events.append(Event(
            sport=Sport.MMA, competition_id=comp.id, event_date=when, status=EventStatus.SCHEDULED,
            fighter_a_id=fighters[a].id, fighter_b_id=fighters[b].id, scheduled_rounds=rounds,
            odds_fighter_a=odds[0], odds_fighter_b=odds[1],
        ))
    db.add_all(events)
    for f, rating in zip(fighters, elo):
        f.elo = _r(rating, 1)


def _counts(db: Session) -> dict:
    def n(model, *where) -> int:
        return db.scalar(select(func.count()).select_from(model).where(*where))

    return {
        "competitions": n(Competition),
        "teams": n(Team),
        "fighters": n(Fighter),
        "events": n(Event),
        "football_finished": n(Event, Event.sport == Sport.FOOTBALL, Event.status == EventStatus.FINISHED),
        "football_scheduled": n(Event, Event.sport == Sport.FOOTBALL, Event.status == EventStatus.SCHEDULED),
        "mma_finished": n(Event, Event.sport == Sport.MMA, Event.status == EventStatus.FINISHED),
        "mma_scheduled": n(Event, Event.sport == Sport.MMA, Event.status == EventStatus.SCHEDULED),
    }


def seed_database(db: Session, force: bool = False) -> dict:
    """Populate the dataset. Returns row counts plus `seeded` (False when skipped).

    Idempotent: does nothing if competitions already exist, unless `force` is set, in which
    case all tables are cleared (children first) and regenerated identically.
    """
    if db.scalar(select(func.count()).select_from(Competition)):
        if not force:
            return {**_counts(db), "seeded": False}
        for model in (Simulation, Event, Team, Fighter, Competition):
            db.execute(delete(model))
        db.flush()

    rng = np.random.default_rng(SEED)
    _seed_football(db, rng)
    _seed_mma(db, rng)
    db.commit()
    return {**_counts(db), "seeded": True}


if __name__ == "__main__":
    init_db()
    with SessionLocal() as session:
        print(seed_database(session))
