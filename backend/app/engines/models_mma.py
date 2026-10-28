"""MMA fight outcome model: multifactor logistic win probability + method/round distribution.

Win probability
---------------
Every factor is an antisymmetric function of (a, b), i.e. f(a, b) = -f(b, a), expressed in
logit space (positive favours fighter A). The sum of the factors is the log-odds of A winning:

    P(A wins) = sigmoid(sum_k factor_k)

Antisymmetry guarantees P(A beats B) + P(B beats A) = 1 and exact symmetry under swapping.

Method and round distribution
-----------------------------
For each fighter X (opponent Y) the method split *given that X wins* is a Bayesian-smoothed
finishing history, re-weighted by the matchup, then normalised:

    w_ko  = share_ko  * ko_mult(X, Y)      striking output vs Y's defence and Y's durability
    w_sub = share_sub * sub_mult(X, Y)     X's sub rate x takedown success vs Y's td defence
    w_dec = share_dec * dec_mult(X, Y)     grows with both fighters' defence and cardio

P(X wins by method) = P(X wins) * w_method / sum(w). Finish probability is spread over the
rounds with a front-loaded profile for KO/TKO and a slightly back-loaded one for submissions;
low cardio tilts both toward later rounds (fighters fade, so stoppages come later).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# --- Win-probability weights (logit units) -------------------------------------------------
W_STRIKING = 0.40       # per net significant strike/min (soft-capped at +-3 strikes/min)
W_GRAPPLING = 0.50      # per unit of net takedown+submission threat (soft-capped at +-2)
W_REACH = 0.25          # saturates: a 10 cm effective-length edge gives tanh(1) ~ 0.76 of it
W_CARDIO = 0.80         # per unit cardio difference in a 3-round fight; x2 for 5 rounds
W_AGE = 0.012           # per year^2 beyond AGE_PRIME, so a 38-year-old is penalised ~0.43
W_EXP_VOLUME = 0.12     # per unit difference of ln(1 + total fights)
W_EXP_RECORD = 1.20     # per unit difference of smoothed win percentage
W_ELO = 0.60            # Elo already encodes results, so it is damped (full scale would be 1.0)
W_STYLE = 0.35          # max edge for a favourable style matchup

AGE_PRIME = 32.0        # no age penalty up to this age
AGE_MAX_PENALTY = 1.5
RECORD_PRIOR_FIGHTS = 4  # pseudo-fights (2-2) pulling win% toward 0.5

# Reference "average UFC fighter" values used to normalise rates to ~1.0
REF_STRIKING_DEF = 0.55
REF_STRIKE_RATE = 3.5
REF_TD_AVG = 1.5
REF_TD_ACC = 0.40
REF_TD_DEF = 0.60
REF_SUB_AVG = 0.5

# --- Method / round constants --------------------------------------------------------------
PRIOR_SHARES = {"ko": 0.32, "sub": 0.18, "dec": 0.50}  # divisional win-method priors
PRIOR_WINS = 8              # pseudo-wins behind the prior (Bayesian smoothing strength)
DEC_DEFENSE_W = 3.0         # log-weight of mean defence (vs 0.575 baseline) on decision share
DEC_CARDIO_W = 1.0          # log-weight of mean cardio (vs 0.7 baseline) on decision share
FIVE_ROUND_DEC_SCALE = 0.80  # decision weight in 5-round fights (more time to finish)
KO_ROUND_DECAY = 0.30       # per-round decay of KO intensity (front-loaded)
SUB_ROUND_GROWTH = 0.10     # per-round growth of submission intensity (slightly later)
CARDIO_FADE = 1.5           # per round, per unit of cardio below 0.7: tilts finishes later

_METHODS = ("ko_tko", "submission", "decision")


@dataclass(frozen=True)
class FighterProfile:
    name: str
    slpm: float = 3.5               # significant strikes landed per minute
    sapm: float = 3.5               # significant strikes absorbed per minute
    striking_accuracy: float = 0.45  # 0-1
    striking_defense: float = 0.55  # 0-1
    td_avg: float = 1.5             # takedowns landed per 15 min
    td_accuracy: float = 0.40       # 0-1
    td_defense: float = 0.60        # 0-1
    sub_avg: float = 0.5            # submission attempts per 15 min
    reach_cm: float = 180.0
    height_cm: float = 178.0
    cardio: float = 0.7             # 0-1 conditioning index (output retention in late rounds)
    age: float = 30.0
    wins: int = 10
    losses: int = 3
    ko_wins: int = 4
    sub_wins: int = 3
    dec_wins: int = 3
    elo: float = 1500.0

    def __post_init__(self) -> None:
        for field in ("slpm", "sapm", "td_avg", "sub_avg", "wins", "losses",
                      "ko_wins", "sub_wins", "dec_wins"):
            if getattr(self, field) < 0:
                raise ValueError(f"{self.name}: {field} must be >= 0")
        for field in ("striking_accuracy", "striking_defense", "td_accuracy",
                      "td_defense", "cardio"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"{self.name}: {field} must be in [0, 1]")
        for field in ("reach_cm", "height_cm", "age"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{self.name}: {field} must be > 0")


def _striking_index(f: FighterProfile) -> float:
    return f.slpm / REF_STRIKE_RATE


def _grappling_index(f: FighterProfile) -> float:
    # takedown output + half-weighted submission attempts, relative to an average fighter
    ref = REF_TD_AVG * REF_TD_ACC + 0.5 * REF_SUB_AVG
    return (f.td_avg * f.td_accuracy + 0.5 * f.sub_avg) / ref


def classify_style(f: FighterProfile) -> str:
    """Classify as "striker", "wrestler", "grappler" or "balanced".

    r = striking_index / grappling_index, each normalised so an average fighter scores 1.0.
      r >= 1.4        -> striker
      r <= 0.7        -> grappling-based: "grappler" if submissions make up >= 40% of the
                         grappling output (sub_avg * 0.5), otherwise "wrestler"
      otherwise       -> balanced
    """
    grappling = _grappling_index(f)
    ratio = _striking_index(f) / grappling if grappling > 0 else math.inf
    if ratio >= 1.4:
        return "striker"
    if ratio <= 0.7:
        sub_share = 0.5 * f.sub_avg / (f.td_avg * f.td_accuracy + 0.5 * f.sub_avg)
        return "grappler" if sub_share >= 0.4 else "wrestler"
    return "balanced"


# --- Win-probability factors (each antisymmetric in (a, b)) ----------------------------------

def _strikes_landed(x: FighterProfile, y: FighterProfile) -> float:
    """Expected strikes/min X lands on Y: mean of X's output through Y's defence
    (normalised to an average defence) and what Y typically absorbs."""
    offence = x.slpm * (1.0 - y.striking_defense) / (1.0 - REF_STRIKING_DEF)
    return 0.5 * (offence + y.sapm)


def _grappling_threat(x: FighterProfile, y: FighterProfile) -> float:
    """Successful takedowns per 15 min plus submission threat. Submissions come from the
    guard (0.3 floor) and grow with takedown success via tanh saturation."""
    takedowns = x.td_avg * x.td_accuracy * (1.0 - y.td_defense)
    return takedowns + 0.5 * x.sub_avg * (0.3 + math.tanh(takedowns))


def _age_penalty(age: float) -> float:
    return min(AGE_MAX_PENALTY, W_AGE * max(0.0, age - AGE_PRIME) ** 2)


def _record_strength(f: FighterProfile) -> float:
    n = f.wins + f.losses
    return (f.wins + RECORD_PRIOR_FIGHTS / 2) / (n + RECORD_PRIOR_FIGHTS)


def _style_edge(x: FighterProfile, sx: str, y: FighterProfile, sy: str) -> float:
    """Non-negative edge of X's style over Y's. A grappling-based X vs a striker grows as the
    striker's takedown defence drops below 0.60; a striker X vs a grappling-based Y grows as
    X's own takedown defence rises above 0.60 (he can keep it standing)."""
    if sx in ("wrestler", "grappler") and sy == "striker":
        return W_STYLE * min(1.0, max(0.0, (REF_TD_DEF - y.td_defense) / 0.30))
    if sx == "striker" and sy in ("wrestler", "grappler"):
        return W_STYLE * min(1.0, max(0.0, (x.td_defense - REF_TD_DEF) / 0.30))
    return 0.0


def _factors(a: FighterProfile, b: FighterProfile, rounds: int,
             style_a: str, style_b: str) -> dict[str, float]:
    striking_diff = _strikes_landed(a, b) - _strikes_landed(b, a)
    grappling_diff = _grappling_threat(a, b) - _grappling_threat(b, a)
    length_diff = (a.reach_cm - b.reach_cm) + 0.5 * (a.height_cm - b.height_cm)
    cardio_scale = (rounds - 1) / 2.0  # 1.0 for 3 rounds, 2.0 for 5 rounds
    return {
        "striking_differential": W_STRIKING * 3.0 * math.tanh(striking_diff / 3.0),
        "grappling_matchup": W_GRAPPLING * 2.0 * math.tanh(grappling_diff / 2.0),
        "reach_advantage": W_REACH * math.tanh(length_diff / 10.0),
        "cardio": W_CARDIO * cardio_scale * (a.cardio - b.cardio),
        "age": _age_penalty(b.age) - _age_penalty(a.age),
        "experience": (W_EXP_VOLUME * (math.log1p(a.wins + a.losses) - math.log1p(b.wins + b.losses))
                       + W_EXP_RECORD * (_record_strength(a) - _record_strength(b))),
        "elo": W_ELO * (a.elo - b.elo) / 400.0 * math.log(10.0),
        "style_matchup": _style_edge(a, style_a, b, style_b) - _style_edge(b, style_b, a, style_a),
    }


# --- Method and round distribution -------------------------------------------------------

def _method_weights(x: FighterProfile, y: FighterProfile, mean_def: float,
                    mean_cardio: float, rounds: int) -> list[float]:
    """Unnormalised [ko, sub, dec] weights for X winning against Y."""
    n = x.ko_wins + x.sub_wins + x.dec_wins + PRIOR_WINS
    shares = [(x.ko_wins + PRIOR_WINS * PRIOR_SHARES["ko"]) / n,
              (x.sub_wins + PRIOR_WINS * PRIOR_SHARES["sub"]) / n,
              (x.dec_wins + PRIOR_WINS * PRIOR_SHARES["dec"]) / n]

    striking_output = x.slpm * (1.0 - y.striking_defense) / (REF_STRIKE_RATE * (1.0 - REF_STRIKING_DEF))
    ko_mult = min(2.5, max(0.4, math.sqrt(striking_output * y.sapm / REF_STRIKE_RATE)))

    td_success = x.td_avg * x.td_accuracy * (1.0 - y.td_defense) / (REF_TD_AVG * REF_TD_ACC * (1.0 - REF_TD_DEF))
    sub_mult = min(3.0, max(0.2, math.sqrt(x.sub_avg / REF_SUB_AVG * (0.3 + 0.7 * td_success))))

    dec_mult = math.exp(DEC_DEFENSE_W * (mean_def - 0.575) + DEC_CARDIO_W * (mean_cardio - 0.7))
    if rounds == 5:
        dec_mult *= FIVE_ROUND_DEC_SCALE

    return [shares[0] * ko_mult, shares[1] * sub_mult, shares[2] * dec_mult]


def _round_profile(rounds: int, slope: float) -> list[float]:
    raw = [math.exp(slope * r) for r in range(rounds)]
    total = sum(raw)
    return [w / total for w in raw]


@dataclass
class MMAPrediction:
    fighter_a: str
    fighter_b: str
    style_a: str
    style_b: str
    win_prob_a: float
    win_prob_b: float
    method_probs: dict       # {"a": {"ko_tko", "submission", "decision"}, "b": {...}}, all six sum to 1
    expected_rounds: float   # expected fight length in rounds (finish counted at mid-round)
    finish_round_probs: dict  # {"1": p, ..., "decision": p}, sums to 1
    factors: dict            # per-factor contribution in logit space (positive favours A)

    def to_dict(self) -> dict:
        return asdict(self)


def predict_fight(a: FighterProfile, b: FighterProfile, rounds: int = 3) -> MMAPrediction:
    """Predict win probability, method of victory and finish timing for A vs B."""
    if rounds not in (3, 5):
        raise ValueError("rounds must be 3 or 5")

    style_a, style_b = classify_style(a), classify_style(b)
    factors = _factors(a, b, rounds, style_a, style_b)
    win_a = 1.0 / (1.0 + math.exp(-sum(factors.values())))
    win_b = 1.0 - win_a

    mean_def = (a.striking_defense + b.striking_defense + a.td_defense + b.td_defense) / 4.0
    mean_cardio = (a.cardio + b.cardio) / 2.0
    method_probs = {}
    for key, win, x, y in (("a", win_a, a, b), ("b", win_b, b, a)):
        w = _method_weights(x, y, mean_def, mean_cardio, rounds)
        total = sum(w)
        method_probs[key] = {m: win * wi / total for m, wi in zip(_METHODS, w)}

    ko_total = method_probs["a"]["ko_tko"] + method_probs["b"]["ko_tko"]
    sub_total = method_probs["a"]["submission"] + method_probs["b"]["submission"]
    dec_total = method_probs["a"]["decision"] + method_probs["b"]["decision"]

    fade = CARDIO_FADE * (0.7 - mean_cardio)  # > 0 for tired fighters: later finishes
    ko_profile = _round_profile(rounds, -KO_ROUND_DECAY + fade)
    sub_profile = _round_profile(rounds, SUB_ROUND_GROWTH + fade)
    finish_by_round = [ko_total * k + sub_total * s for k, s in zip(ko_profile, sub_profile)]

    finish_round_probs = {str(r + 1): p for r, p in enumerate(finish_by_round)}
    finish_round_probs["decision"] = dec_total
    expected_rounds = sum(p * (r + 0.5) for r, p in enumerate(finish_by_round)) + dec_total * rounds

    return MMAPrediction(
        fighter_a=a.name, fighter_b=b.name, style_a=style_a, style_b=style_b,
        win_prob_a=win_a, win_prob_b=win_b, method_probs=method_probs,
        expected_rounds=expected_rounds, finish_round_probs=finish_round_probs,
        factors=factors,
    )
