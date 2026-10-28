"""Quant toolkit: Monte Carlo simulation, odds maths (vig removal, EV, Kelly) and backtesting.

Works only on score matrices and probability dicts, so it is independent of the sport models.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

WILSON_Z = 1.96  # 95% interval
MAX_SIMULATIONS = 1_000_000
CALIBRATION_BINS = 10
LOG_LOSS_EPS = 1e-15


def _check_simulations(n: int) -> None:
    if not 1 <= n <= MAX_SIMULATIONS:
        raise ValueError(f"n_simulations must be between 1 and {MAX_SIMULATIONS}")


def _check_prob(p: float, name: str = "probability") -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def _check_odds(odds: float) -> None:
    if not odds > 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {odds}")


def _wilson(successes: int, n: int) -> list[float]:
    """95% Wilson score interval for a binomial proportion."""
    p = successes / n
    z2 = WILSON_Z ** 2
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


# --- Football Monte Carlo -----------------------------------------------------------------

@dataclass
class FootballSimulation:
    n_simulations: int
    elapsed_ms: float
    home_goals: np.ndarray
    away_goals: np.ndarray

    def summary(self, ou_lines=(0.5, 1.5, 2.5, 3.5, 4.5)) -> dict:
        h, a, n = self.home_goals, self.away_goals, self.n_simulations
        total = h + a
        wins, draws, losses = int((h > a).sum()), int((h == a).sum()), int((h < a).sum())

        width = int(a.max()) + 1
        counts = np.bincount(h * width + a)
        top = np.argsort(counts)[::-1][:10]
        score_distribution = [
            {"score": f"{i // width}-{i % width}", "frequency": float(counts[i] / n)}
            for i in top if counts[i] > 0
        ]

        size = int(max(h.max(), a.max())) + 1
        return {
            "n_simulations": n,
            "elapsed_ms": self.elapsed_ms,
            "home_win": wins / n,
            "draw": draws / n,
            "away_win": losses / n,
            "btts_yes": float(((h > 0) & (a > 0)).mean()),
            "over_under": {
                str(line): {"over": float((total > line).mean()), "under": float((total < line).mean())}
                for line in ou_lines
            },
            "mean_home_goals": float(h.mean()),
            "mean_away_goals": float(a.mean()),
            "std_home_goals": float(h.std()),
            "std_away_goals": float(a.std()),
            "confidence_intervals": {
                "home_win": _wilson(wins, n),
                "draw": _wilson(draws, n),
                "away_win": _wilson(losses, n),
            },
            "score_distribution": score_distribution,
            "goal_histogram": {
                "home": np.bincount(h, minlength=size).tolist(),
                "away": np.bincount(a, minlength=size).tolist(),
            },
        }


def simulate_football(score_matrix: np.ndarray, n_simulations: int = 10_000,
                      seed: int | None = None) -> FootballSimulation:
    """Sample scorelines from the exact joint distribution.

    score_matrix[i, j] = P(home scores i, away scores j). Cells are drawn in one vectorised
    call over the flattened matrix and decoded back to (home, away) with divmod.
    """
    _check_simulations(n_simulations)
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or not np.isfinite(m).all() or (m < 0).any() or m.sum() <= 0:
        raise ValueError("score_matrix must be a 2-D, finite, non-negative array with positive sum")

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    cells = rng.choice(m.size, size=n_simulations, p=(m / m.sum()).ravel())
    home, away = np.divmod(cells, m.shape[1])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return FootballSimulation(n_simulations, elapsed_ms, home, away)


# --- Fight Monte Carlo --------------------------------------------------------------------

_METHOD_KEYS = ("ko_tko", "submission", "decision")


@dataclass
class FightSimulation:
    n_simulations: int
    elapsed_ms: float
    winners: np.ndarray     # 0 = a, 1 = b
    methods: np.ndarray     # 0 = ko_tko, 1 = submission, 2 = decision
    end_rounds: np.ndarray  # decisions = scheduled rounds
    rounds: int = 3         # scheduled rounds (needed to label the round distribution)

    def summary(self) -> dict:
        n, w, m = self.n_simulations, self.winners, self.methods
        a_wins = int((w == 0).sum())
        finished = m < 2
        round_distribution = {
            str(r): float((finished & (self.end_rounds == r)).mean()) for r in range(1, self.rounds + 1)
        }
        round_distribution["decision"] = float((m == 2).mean())
        return {
            "n_simulations": n,
            "elapsed_ms": self.elapsed_ms,
            "win_prob_a": a_wins / n,
            "win_prob_b": 1.0 - a_wins / n,
            "confidence_intervals": {"a": _wilson(a_wins, n)},
            "methods": {
                side: {name: float(((w == i) & (m == j)).mean()) for j, name in enumerate(_METHOD_KEYS)}
                for i, side in enumerate(("a", "b"))
            },
            "round_distribution": round_distribution,
            "finish_rate": float(finished.mean()),
        }


def simulate_fight(method_probs: dict, finish_round_probs: dict, rounds: int,
                   n_simulations: int = 10_000, seed: int | None = None) -> FightSimulation:
    """Sample (winner, method) among the 6 categories, then a round for each finish.

    Inputs are MMAPrediction.method_probs and .finish_round_probs. The finish round is drawn
    from the numbered-round entries only, renormalised (the "decision" entry is ignored).
    """
    _check_simulations(n_simulations)
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    categories = np.array([method_probs[s][k] for s in ("a", "b") for k in _METHOD_KEYS], dtype=float)
    round_p = np.array([finish_round_probs.get(str(r), 0.0) for r in range(1, rounds + 1)], dtype=float)
    if (categories < 0).any() or (round_p < 0).any() or categories.sum() <= 0:
        raise ValueError("probabilities must be non-negative with positive total")
    if round_p.sum() <= 0 and categories[[0, 1, 3, 4]].sum() > 0:
        raise ValueError("finish_round_probs has no mass on numbered rounds but finishes are possible")

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    cats = rng.choice(6, size=n_simulations, p=categories / categories.sum())
    winners, methods = np.divmod(cats, 3)
    end_rounds = np.full(n_simulations, rounds, dtype=int)
    finished = methods < 2
    if finished.any():
        end_rounds[finished] = rng.choice(rounds, size=int(finished.sum()), p=round_p / round_p.sum()) + 1
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return FightSimulation(n_simulations, elapsed_ms, winners, methods, end_rounds, rounds)


# --- Odds maths ---------------------------------------------------------------------------

def implied_probability(decimal_odds: float) -> float:
    _check_odds(decimal_odds)
    return 1.0 / decimal_odds


def remove_vig(decimal_odds: Sequence[float], method: str = "proportional") -> list[float]:
    """Fair probabilities from one mutually exclusive market's odds.

    proportional: p_i / sum(p).
    power: find k with sum(p_i^k) = 1 (brentq); shrinks long shots more than favourites.
    """
    if len(decimal_odds) < 2:
        raise ValueError("a market needs at least two selections")
    p = np.array([implied_probability(o) for o in decimal_odds])
    if method == "proportional":
        return (p / p.sum()).tolist()
    if method == "power":
        def excess(k: float) -> float:
            return float((p ** k).sum() - 1.0)
        hi = 2.0
        while excess(hi) > 0:  # f is decreasing in k; widen until the root is bracketed
            hi *= 2.0
        return (p ** brentq(excess, 0.0, hi)).tolist()
    raise ValueError("method must be 'proportional' or 'power'")


def bookmaker_margin(decimal_odds: Sequence[float]) -> float:
    return sum(implied_probability(o) for o in decimal_odds) - 1.0


def expected_value(prob: float, decimal_odds: float) -> float:
    """Expected profit per unit staked: p * odds - 1."""
    _check_prob(prob)
    _check_odds(decimal_odds)
    return prob * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float, multiplier: float = 0.25,
                   cap: float = 0.10) -> float:
    """Fractional Kelly stake as a share of bankroll: (b*p - q)/b * multiplier, in [0, cap]."""
    _check_prob(prob)
    _check_odds(decimal_odds)
    b = decimal_odds - 1.0
    full = (b * prob - (1.0 - prob)) / b
    return min(cap, max(0.0, full * multiplier))


@dataclass
class ValueBet:
    selection: str
    model_prob: float
    decimal_odds: float
    implied_prob: float
    fair_prob: float        # vig-removed market probability
    edge: float             # model_prob - fair_prob
    expected_value: float
    kelly_fraction: float
    stake: float            # bankroll * kelly_fraction
    is_value: bool          # EV > min_ev

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_market(selections: Sequence[tuple[str, float, float]], bankroll: float = 1000.0,
                    kelly_multiplier: float = 0.25, min_ev: float = 0.0) -> list[ValueBet]:
    """Evaluate one mutually exclusive market; selections are (label, model_prob, decimal_odds).
    Returns bets sorted by expected value, best first."""
    if bankroll <= 0:
        raise ValueError("bankroll must be > 0")
    fair = remove_vig([odds for _, _, odds in selections])
    bets = []
    for (label, prob, odds), fair_p in zip(selections, fair):
        ev = expected_value(prob, odds)
        kelly = kelly_fraction(prob, odds, kelly_multiplier)
        bets.append(ValueBet(label, prob, odds, implied_probability(odds), fair_p, prob - fair_p,
                             ev, kelly, bankroll * kelly, ev > min_ev))
    return sorted(bets, key=lambda b: b.expected_value, reverse=True)


# --- Backtesting --------------------------------------------------------------------------

@dataclass
class BacktestReport:
    n_events: int
    n_bets: int
    brier_score: float      # multiclass Brier, averaged over events
    log_loss: float
    accuracy: float         # argmax hit rate
    calibration: list[dict]  # bins over all outcome probabilities
    roi: float              # profit / total staked
    yield_pct: float
    profit: float
    final_bankroll: float
    max_drawdown: float     # fraction peak-to-trough
    sharpe: float           # mean/std of per-bet returns * sqrt(n_bets); 0 if < 2 bets
    bankroll_curve: list[float]
    bets: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _calibration(probs: np.ndarray, hits: np.ndarray) -> list[dict]:
    p, y = probs.ravel(), hits.ravel()
    bin_idx = np.minimum((p * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    rows = []
    for i in range(CALIBRATION_BINS):
        in_bin = bin_idx == i
        count = int(in_bin.sum())
        rows.append({
            "bin_low": i / CALIBRATION_BINS,
            "bin_high": (i + 1) / CALIBRATION_BINS,
            "mean_predicted": float(p[in_bin].mean()) if count else None,
            "observed_frequency": float(y[in_bin].mean()) if count else None,
            "count": count,
        })
    return rows


def run_backtest(records: Sequence[dict], outcome_keys: Sequence[str], bankroll: float = 1000.0,
                 kelly_multiplier: float = 0.25, min_ev: float = 0.02) -> BacktestReport:
    """Score probabilities and simulate flat-rules betting over historical records.

    Each record: {"probs": {key: p}, "odds": {key: decimal or None}, "outcome": key}.
    Metrics use every record. Betting: when odds exist for every outcome, place at most the
    single best bet with EV > min_ev, sized by fractional Kelly on the current bankroll.
    """
    if not records:
        raise ValueError("records must not be empty")
    keys = list(outcome_keys)
    probs = np.array([[r["probs"][k] for k in keys] for r in records], dtype=float)
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    outcomes = []
    for r in records:
        if r["outcome"] not in keys:
            raise ValueError(f"unknown outcome {r['outcome']!r}")
        outcomes.append(keys.index(r["outcome"]))
    outcomes = np.array(outcomes)
    hits = np.zeros_like(probs)
    hits[np.arange(len(records)), outcomes] = 1.0

    p_true = probs[np.arange(len(records)), outcomes]
    brier = float(((probs - hits) ** 2).sum(axis=1).mean())
    log_loss = float(-np.log(np.maximum(p_true, LOG_LOSS_EPS)).mean())
    accuracy = float((probs.argmax(axis=1) == outcomes).mean())

    balance = bankroll
    curve = [balance]
    bets: list[dict] = []
    returns: list[float] = []
    staked_total = 0.0
    for i, r in enumerate(records):
        odds = [r["odds"].get(k) for k in keys]
        if any(o is None for o in odds):
            continue
        best = evaluate_market(list(zip(keys, probs[i].tolist(), odds)), balance, kelly_multiplier, min_ev)[0]
        if not best.is_value or best.stake <= 0:
            continue
        profit = best.stake * (best.decimal_odds - 1.0) if best.selection == r["outcome"] else -best.stake
        balance += profit
        staked_total += best.stake
        returns.append(profit / best.stake)
        curve.append(balance)
        bets.append({"index": i, "selection": best.selection, "model_prob": best.model_prob,
                     "odds": best.decimal_odds, "stake": best.stake, "profit": profit,
                     "bankroll": balance})

    curve_arr = np.array(curve)
    drawdown = float(((np.maximum.accumulate(curve_arr) - curve_arr) / np.maximum.accumulate(curve_arr)).max())
    total_profit = balance - bankroll
    roi = total_profit / staked_total if staked_total > 0 else 0.0
    std = float(np.std(returns, ddof=1)) if len(returns) >= 2 else 0.0
    sharpe = float(np.mean(returns)) / std * math.sqrt(len(returns)) if std > 0 else 0.0

    return BacktestReport(
        n_events=len(records), n_bets=len(bets), brier_score=brier, log_loss=log_loss,
        accuracy=accuracy, calibration=_calibration(probs, hits), roi=roi, yield_pct=roi * 100.0,
        profit=total_profit, final_bankroll=balance, max_drawdown=drawdown, sharpe=sharpe,
        bankroll_curve=curve, bets=bets,
    )
