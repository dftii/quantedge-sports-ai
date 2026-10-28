"""Football (soccer) match models: bivariate Poisson, Dixon-Coles, Elo and a profile-based predictor.

Pure computation (NumPy / SciPy only). Score matrices have rows = home goals, columns = away goals.

Model summary
-------------
* Independent Poisson:      P(x, y) = Pois(x; lam) * Pois(y; mu)
* Bivariate Poisson (Karlis & Ntzoufras 2003): X = X1 + X3, Y = X2 + X3 with Xi ~ Pois(lam_i)
  independent, so Cov(X, Y) = lam3 > 0 and
      P(x, y) = sum_k Pois(k; lam3) * Pois(x - k; lam1) * Pois(y - k; lam2)
  which is a matrix product of two lower-triangular Toeplitz matrices (no loops over the grid).
* Dixon-Coles (1997): multiply the base grid by tau(x, y) for the four low scores, which corrects
  the independence model's under-estimation of 0-0 / 1-1 and over-estimation of 1-0 / 0-1.
"""
from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Elo-implied log goal ratio per rating point (400 Elo -> goal ratio e^0.8 ~ 2.2).
ELO_LOG_SCALE = 500.0
# Log-lambda tilt per possession percentage point of difference (10 pts -> ~3%).
POSSESSION_COEF = 0.003
# Smoothing factor of the exponentially weighted xG averages in the walk-forward backtest.
EWMA_ALPHA = 0.15


# --------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class TeamProfile:
    name: str
    elo: float = 1500.0
    xg_for: float = 1.35  # avg xG created per match
    xg_against: float = 1.35  # avg xG conceded per match
    possession: float = 50.0  # avg possession %
    defensive_rating: float = 1.0  # >1 = better than league-average defence


@dataclass(frozen=True)
class MatchRecord:
    """Historical result used for fitting / backtesting."""

    date: datetime.date
    home: str
    away: str
    home_goals: int
    away_goals: int
    home_xg: float | None = None
    away_xg: float | None = None
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None


@dataclass(frozen=True)
class MatchAdjustments:
    """Live what-if tweaks applied to the expected goals: home advantage multiplier, form (+-fraction) and absences."""

    home_advantage: float = 1.0
    home_form: float = 0.0
    away_form: float = 0.0
    home_absences: int = 0
    away_absences: int = 0


@dataclass
class FootballPrediction:
    home: str
    away: str
    lambda_home: float  # expected home goals (marginal mean)
    lambda_away: float
    rho: float  # effective (feasibility-clipped) Dixon-Coles rho
    lam3: float  # bivariate covariance term
    elo_home: float
    elo_away: float
    elo_win_expectancy: float  # home expected score under Elo, home advantage included
    matrix: np.ndarray = field(repr=False)
    markets: dict = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "home": self.home,
            "away": self.away,
            "lambda_home": float(self.lambda_home),
            "lambda_away": float(self.lambda_away),
            "rho": float(self.rho),
            "lam3": float(self.lam3),
            "elo_home": float(self.elo_home),
            "elo_away": float(self.elo_away),
            "elo_win_expectancy": float(self.elo_win_expectancy),
            "matrix": np.round(self.matrix, 6).tolist(),
            "markets": self.markets,
        }


# --------------------------------------------------------------------------- distributions
def _check_max_goals(max_goals: int) -> None:
    if int(max_goals) != max_goals or max_goals < 1:
        raise ValueError(f"max_goals must be an integer >= 1, got {max_goals}")


def poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    """Poisson pmf for 0..max_goals (the tail beyond max_goals is dropped)."""
    _check_max_goals(max_goals)
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    return poisson.pmf(np.arange(max_goals + 1), lam)


def bivariate_poisson_matrix(lam1: float, lam2: float, lam3: float, max_goals: int = 10) -> np.ndarray:
    """Karlis-Ntzoufras bivariate Poisson grid, truncated at max_goals and renormalised to sum 1.

    P(x, y) = sum_k p3[k] p1[x-k] p2[y-k]  =  (T1 @ diag(p3) @ T2^T)[x, y],
    where T[x, k] = p[x-k] for x >= k (else 0).
    """
    _check_max_goals(max_goals)
    if lam1 <= 0 or lam2 <= 0 or lam3 < 0:
        raise ValueError(f"need lam1, lam2 > 0 and lam3 >= 0, got {lam1}, {lam2}, {lam3}")
    n = max_goals + 1
    lag = np.arange(n)[:, None] - np.arange(n)[None, :]  # x - k

    def toeplitz(lam: float) -> np.ndarray:
        return np.where(lag >= 0, poisson_pmf_vector(lam, max_goals)[np.clip(lag, 0, None)], 0.0)

    p3 = poisson_pmf_vector(lam3, max_goals)
    matrix = (toeplitz(lam1) * p3) @ toeplitz(lam2).T
    return matrix / matrix.sum()


def _rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Range of rho for which every Dixon-Coles tau is >= 0 (and so the grid is a valid pmf)."""
    return max(-1.0 / lam, -1.0 / mu), min(1.0, 1.0 / (lam * mu))


def dixon_coles_tau(x, y, lam, mu, rho):
    """Vectorised Dixon-Coles low-score correction (1 for every score other than 0-0, 1-0, 0-1, 1-1)."""
    x, y = np.asarray(x), np.asarray(y)
    return np.where(
        (x == 0) & (y == 0), 1.0 - lam * mu * rho,
        np.where(
            (x == 0) & (y == 1), 1.0 + lam * rho,
            np.where((x == 1) & (y == 0), 1.0 + mu * rho, np.where((x == 1) & (y == 1), 1.0 - rho, 1.0)),
        ),
    )


def dixon_coles_matrix(
    lam: float, mu: float, rho: float, lam3: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Score grid with marginal means (lam, mu), shared covariance term lam3 and DC correction rho.

    The base grid is a bivariate Poisson with lam1 = lam - lam3, lam2 = mu - lam3 (so the marginal
    means stay lam and mu); lam3 = 0 gives independent Poissons. rho is clipped to the feasible range.
    """
    _check_max_goals(max_goals)
    if lam <= 0 or mu <= 0:
        raise ValueError(f"lam and mu must be > 0, got {lam}, {mu}")
    if not 0 <= lam3 < min(lam, mu):
        raise ValueError(f"lam3 must satisfy 0 <= lam3 < min(lam, mu), got {lam3}")
    rho = float(np.clip(rho, *_rho_bounds(lam, mu)))
    base = bivariate_poisson_matrix(lam - lam3, mu - lam3, lam3, max_goals)
    goals = np.arange(max_goals + 1)
    matrix = base * dixon_coles_tau(goals[:, None], goals[None, :], lam, mu, rho)
    return matrix / matrix.sum()


# --------------------------------------------------------------------------- markets
def market_probabilities(
    matrix: np.ndarray, ou_lines: Sequence[float] = (0.5, 1.5, 2.5, 3.5, 4.5), top_n: int = 10
) -> dict:
    """Derive betting-market probabilities from a (home x away) score grid (normalised internally)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.sum() <= 0:
        raise ValueError("matrix must be a square, non-empty score grid")
    matrix = matrix / matrix.sum()
    n = matrix.shape[0]
    goals = np.arange(n)
    total = goals[:, None] + goals[None, :]

    home_win = float(np.tril(matrix, -1).sum())  # rows (home goals) > columns (away goals)
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    btts_yes = float(matrix[1:, 1:].sum())

    over_under = {}
    for line in ou_lines:
        over = float(matrix[total > line].sum())
        over_under[f"{line:g}"] = {"over": over, "under": 1.0 - over}

    flat = matrix.ravel()
    best = np.argsort(-flat, kind="stable")[: min(top_n, flat.size)]
    correct_scores = [
        {"score": f"{i}-{j}", "home_goals": int(i), "away_goals": int(j), "probability": float(flat[k])}
        for k in best
        for i, j in [divmod(int(k), n)]
    ]

    decisive = home_win + away_win
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "over_under": over_under,
        "expected_home_goals": float(matrix.sum(axis=1) @ goals),
        "expected_away_goals": float(matrix.sum(axis=0) @ goals),
        "correct_scores": correct_scores,
        "double_chance": {"1X": home_win + draw, "X2": draw + away_win, "12": decisive},
        "asian_handicap_0": {"home": home_win / decisive, "away": away_win / decisive},
    }


# --------------------------------------------------------------------------- Elo
class EloRating:
    """World-Football-Elo style ratings with a goal-difference multiplier."""

    def __init__(self, k: float = 20.0, home_advantage: float = 65.0, initial: float = 1500.0):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def expected_score(self, rating_home: float, rating_away: float) -> float:
        """Home expected score (win = 1, draw = 0.5) with the home-advantage bonus added."""
        return 1.0 / (1.0 + 10.0 ** (-(rating_home + self.home_advantage - rating_away) / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> tuple[float, float]:
        """Apply one result: delta = k * G * (W - We). Zero-sum, so the rating total is conserved.

        G = 1 for a 1-goal margin, 1.5 for 2 goals, (11 + margin) / 8 for 3 or more.
        """
        r_home, r_away = self.get(home), self.get(away)
        margin = abs(home_goals - away_goals)
        g = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        result = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = self.k * g * (result - self.expected_score(r_home, r_away))
        self.ratings[home], self.ratings[away] = r_home + delta, r_away - delta
        return self.ratings[home], self.ratings[away]

    def fit(self, matches: Sequence[MatchRecord]) -> dict[str, float]:
        """Reset and replay matches chronologically."""
        self.ratings = {}
        for m in sorted(matches, key=lambda m: m.date):
            self.update(m.home, m.away, m.home_goals, m.away_goals)
        return self.ratings


# --------------------------------------------------------------------------- Dixon-Coles MLE
class DixonColesModel:
    """MLE fit of attack/defence/home/rho with exponential time decay xi (Dixon & Coles 1997).

        log lam_home = intercept + home_advantage + attack[home] + defence[away]
        log lam_away = intercept + attack[away]   + defence[home]

    attack and defence both sum to zero across teams (identifiability); defence is the log-propensity
    to concede, so a LOWER value is a better defence. Each match is weighted by exp(-xi * days_ago),
    xi = 0.0019/day giving a half-life of about a year. The log-likelihood is
        sum_m w_m [log tau + x log lam - lam + y log mu - mu]   (factorial constants dropped).
    """

    def __init__(self, xi: float = 0.0019, max_goals: int = 10):
        _check_max_goals(max_goals)
        self.xi = xi
        self.max_goals = max_goals
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.intercept = 0.0
        self.home_advantage = 0.0
        self.rho = 0.0
        self.fitted = False
        self.elo = EloRating()

    def fit(self, matches: Sequence[MatchRecord], reference_date: datetime.date | None = None) -> "DixonColesModel":
        if not matches:
            raise ValueError("cannot fit Dixon-Coles model: no matches given")
        reference_date = reference_date or max(m.date for m in matches)
        used = [m for m in matches if m.date <= reference_date]
        if not used:
            raise ValueError(f"cannot fit Dixon-Coles model: no matches on or before {reference_date}")
        teams = sorted({m.home for m in used} | {m.away for m in used})
        if len(teams) < 2:
            raise ValueError("cannot fit Dixon-Coles model: need at least 2 distinct teams")

        n = len(teams)
        index = {t: i for i, t in enumerate(teams)}
        hi = np.array([index[m.home] for m in used])
        ai = np.array([index[m.away] for m in used])
        x = np.array([m.home_goals for m in used], dtype=float)
        y = np.array([m.away_goals for m in used], dtype=float)
        w = np.exp(-self.xi * np.array([(reference_date - m.date).days for m in used], dtype=float))

        def unpack(theta):
            a, d = theta[3 : n + 2], theta[n + 2 :]
            return theta[0], theta[1], theta[2], np.append(a, -a.sum()), np.append(d, -d.sum())

        def neg_loglik(theta):
            intercept, gamma, rho, att, dfn = unpack(theta)
            log_lam = intercept + gamma + att[hi] + dfn[ai]
            log_mu = intercept + att[ai] + dfn[hi]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            tau = np.maximum(dixon_coles_tau(x, y, lam, mu, rho), 1e-10)
            return -np.sum(w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu))

        theta0 = np.zeros(3 + 2 * (n - 1))
        theta0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.1))
        bounds = [(-3, 3), (-1, 1), (-0.3, 0.3)] + [(-3, 3)] * (2 * (n - 1))
        res = minimize(neg_loglik, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})

        intercept, gamma, rho, att, dfn = unpack(res.x)
        self.intercept, self.home_advantage, self.rho = float(intercept), float(gamma), float(rho)
        self.attack = dict(zip(teams, map(float, att)))
        self.defence = dict(zip(teams, map(float, dfn)))
        self.elo = EloRating()
        self.elo.fit(used)
        self.fitted = True
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if not self.fitted:
            raise ValueError("model is not fitted; call fit() first")
        for team in (home, away):
            if team not in self.attack:
                raise ValueError(f"unknown team: {team!r}")
        lam = np.exp(self.intercept + self.home_advantage + self.attack[home] + self.defence[away])
        mu = np.exp(self.intercept + self.attack[away] + self.defence[home])
        return float(lam), float(mu)

    def predict(self, home: str, away: str) -> FootballPrediction:
        lam, mu = self.expected_goals(home, away)
        rho = float(np.clip(self.rho, *_rho_bounds(lam, mu)))
        matrix = dixon_coles_matrix(lam, mu, rho, 0.0, self.max_goals)
        elo_home, elo_away = self.elo.get(home), self.elo.get(away)
        return FootballPrediction(
            home=home, away=away, lambda_home=lam, lambda_away=mu, rho=rho, lam3=0.0,
            elo_home=elo_home, elo_away=elo_away,
            elo_win_expectancy=self.elo.expected_score(elo_home, elo_away),
            matrix=matrix, markets=market_probabilities(matrix),
        )


# --------------------------------------------------------------------------- profile-based predictor
def expected_goals_from_profiles(
    home: TeamProfile,
    away: TeamProfile,
    league_avg_goals: float = 1.35,
    home_advantage: float = 1.12,
    elo_weight: float = 0.5,
) -> tuple[float, float]:
    """Expected goals (home, away) from team profiles.

    1. Strengths relative to the league: attack = xg_for / L, weakness = xg_against / (L * defensive_rating).
    2. Stats lambdas (L = league_avg_goals per team, h = home_advantage, p = POSSESSION_COEF):
           lam = L * attack_h * weakness_a * h * exp( p * (poss_h - poss_a))
           mu  = L * attack_a * weakness_h / h * exp(-p * (poss_h - poss_a))
    3. Elo blend: keep the stats' log-total g = (ln lam + ln mu) / 2 but blend the log goal ratio
           r = (1 - w) * ln(lam / mu) + w * (2 ln h + (elo_h - elo_a) / ELO_LOG_SCALE),   w = elo_weight
       and set ln lam = g + r/2, ln mu = g - r/2.
    Both lambdas are clamped to [0.05, 6.0].
    """
    if league_avg_goals <= 0 or home_advantage <= 0 or not 0 <= elo_weight <= 1:
        raise ValueError("need league_avg_goals > 0, home_advantage > 0 and 0 <= elo_weight <= 1")
    for t in (home, away):
        if t.xg_for <= 0 or t.xg_against <= 0 or t.defensive_rating <= 0:
            raise ValueError(f"xg_for, xg_against and defensive_rating must be > 0 ({t.name})")

    league = league_avg_goals
    possession_tilt = POSSESSION_COEF * (home.possession - away.possession)
    log_lam = np.log(league * (home.xg_for / league) * (away.xg_against / (league * away.defensive_rating))
                     * home_advantage) + possession_tilt
    log_mu = np.log(league * (away.xg_for / league) * (home.xg_against / (league * home.defensive_rating))
                    / home_advantage) - possession_tilt

    log_total = (log_lam + log_mu) / 2
    ratio_stats = log_lam - log_mu
    ratio_elo = 2 * np.log(home_advantage) + (home.elo - away.elo) / ELO_LOG_SCALE
    ratio = (1 - elo_weight) * ratio_stats + elo_weight * ratio_elo
    lam, mu = np.exp(log_total + ratio / 2), np.exp(log_total - ratio / 2)
    return float(np.clip(lam, 0.05, 6.0)), float(np.clip(mu, 0.05, 6.0))


def apply_adjustments(lam: float, mu: float, adj: MatchAdjustments) -> tuple[float, float]:
    """Scale expected goals by form (+-fraction), absences (-4% own attack, +3% opponent's) and home advantage.

    Both results are clamped to [0.05, 6.0]; neutral adjustments leave an in-range pair unchanged.
    """
    lam *= (1 + adj.home_form) * (1 - 0.04 * adj.home_absences) * (1 + 0.03 * adj.away_absences) * adj.home_advantage
    mu *= (1 + adj.away_form) * (1 - 0.04 * adj.away_absences) * (1 + 0.03 * adj.home_absences) / adj.home_advantage
    return float(np.clip(lam, 0.05, 6.0)), float(np.clip(mu, 0.05, 6.0))


def predict_match(
    home: TeamProfile,
    away: TeamProfile,
    league_avg_goals: float = 1.35,
    rho: float = -0.08,
    lam3: float = 0.0,
    max_goals: int = 10,
    ou_lines: Sequence[float] = (0.5, 1.5, 2.5, 3.5, 4.5),
    adjustments: MatchAdjustments | None = None,
) -> FootballPrediction:
    lam, mu = expected_goals_from_profiles(home, away, league_avg_goals)
    if adjustments is not None:
        lam, mu = apply_adjustments(lam, mu, adjustments)
    if lam3 < 0:
        raise ValueError(f"lam3 must be >= 0, got {lam3}")
    lam3 = min(lam3, 0.5 * min(lam, mu))  # keep lam1, lam2 = lam - lam3, mu - lam3 comfortably positive
    rho = float(np.clip(rho, *_rho_bounds(lam, mu)))
    matrix = dixon_coles_matrix(lam, mu, rho, lam3, max_goals)
    elo = EloRating()
    return FootballPrediction(
        home=home.name, away=away.name, lambda_home=lam, lambda_away=mu, rho=rho, lam3=lam3,
        elo_home=home.elo, elo_away=away.elo,
        elo_win_expectancy=elo.expected_score(home.elo, away.elo),
        matrix=matrix, markets=market_probabilities(matrix, ou_lines),
    )


# --------------------------------------------------------------------------- walk-forward backtest feed
def walk_forward_predictions(
    matches: Sequence[MatchRecord], warmup: int = 30, league_avg_goals: float | None = None
) -> list[dict]:
    """Leak-free predictions: each match is predicted from state built only from earlier matches.

    State per team: Elo (sequential updates) and exponentially weighted xG for/against
    (alpha = EWMA_ALPHA; actual goals stand in when xG is missing). The league average is the running
    mean of per-team goals/xG unless league_avg_goals is fixed. The first `warmup` matches only
    update state and are not emitted.
    """
    elo = EloRating()
    form: dict[str, list[float]] = {}  # team -> [xg_for, xg_against]
    league_sum, league_n = 0.0, 0
    rows: list[dict] = []

    def profile(team: str, league: float) -> TeamProfile:
        xg_for, xg_against = form.get(team, (league, league))
        return TeamProfile(name=team, elo=elo.get(team), xg_for=xg_for, xg_against=xg_against)

    for i, m in enumerate(sorted(matches, key=lambda m: m.date)):
        home_xg = m.home_goals if m.home_xg is None else m.home_xg
        away_xg = m.away_goals if m.away_xg is None else m.away_xg

        if i >= warmup:
            league = league_avg_goals or (league_sum / league_n if league_n else 1.35)
            pred = predict_match(
                profile(m.home, league), profile(m.away, league), league_avg_goals=league, ou_lines=(2.5,)
            )
            mk = pred.markets
            rows.append({
                "date": m.date, "home": m.home, "away": m.away,
                "home_goals": m.home_goals, "away_goals": m.away_goals,
                "p_home": mk["home_win"], "p_draw": mk["draw"], "p_away": mk["away_win"],
                "p_over_2_5": mk["over_under"]["2.5"]["over"], "p_btts": mk["btts_yes"],
                "outcome": "H" if m.home_goals > m.away_goals else "D" if m.home_goals == m.away_goals else "A",
                "odds_home": m.odds_home, "odds_draw": m.odds_draw, "odds_away": m.odds_away,
            })

        # Update state with the result only after predicting it.
        for team, scored, conceded in ((m.home, home_xg, away_xg), (m.away, away_xg, home_xg)):
            old = form.get(team, (scored, conceded))
            form[team] = (old[0] + EWMA_ALPHA * (scored - old[0]), old[1] + EWMA_ALPHA * (conceded - old[1]))
        elo.update(m.home, m.away, m.home_goals, m.away_goals)
        league_sum += home_xg + away_xg
        league_n += 2
    return rows
