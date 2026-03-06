"""
Bayesian Bladder Urge Estimation
=================================
Sequential Monte Carlo (particle filter) framework for estimating
the probability of feeling the urge to urinate, given a history of
drink events and observed urge/void events.

Usage:
    model = BladderUrgeModel(n_particles=1000)
    model.process_event(DrinkEvent(time=0.0, volume=250.0, kind=DrinkKind.WATER))
    model.process_event(DrinkEvent(time=30.0, volume=120.0, kind=DrinkKind.COFFEE))
    prob = model.query(t_now=90.0, horizon=30.0)
    print(f"P(urge in next 30 min) = {prob:.3f}")
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums and Event types
# ---------------------------------------------------------------------------

class DrinkKind(Enum):
    WATER  = auto()
    COFFEE = auto()
    TEA    = auto()


@dataclass
class DrinkEvent:
    time:   float       # minutes since epoch
    volume: float       # ml
    kind:   DrinkKind


@dataclass
class UrgeEvent:
    time: float         # observed urge onset time


@dataclass
class VoidEvent:
    time: float         # bladder emptied at this time


# ---------------------------------------------------------------------------
# Particle (one hypothesis about model parameters)
# ---------------------------------------------------------------------------

@dataclass
class Params:
    lambda0:      float   # baseline hazard rate (events per minute)
    beta:         float   # sigmoid steepness
    theta:        float   # bladder-load threshold for urge (ml-equivalent)
    tau_absorb:   float   # absorption time constant (minutes)
    tau_excrete:  float   # excretion time constant (minutes)
    alpha_coffee: float   # diuretic multiplier for coffee (water=1)
    alpha_tea:    float   # diuretic multiplier for tea


@dataclass
class Particle:
    params: Params
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------

def _lognormal_sample(mean_log: float, sigma_log: float) -> float:
    """Sample from LogNormal(mean_log, sigma_log) using Box-Muller."""
    u1 = random.random()
    u2 = random.random()
    z  = math.sqrt(-2 * math.log(u1 + 1e-300)) * math.cos(2 * math.pi * u2)
    return math.exp(mean_log + sigma_log * z)


def sample_from_prior() -> Params:
    return Params(
        lambda0      = _lognormal_sample(math.log(0.005), 0.5),   # ~0.005 urges/min baseline
        beta         = _lognormal_sample(math.log(0.05),  0.5),   # sigmoid steepness
        theta        = _lognormal_sample(math.log(400.0), 0.4),   # ~400 ml threshold
        tau_absorb   = _lognormal_sample(math.log(20.0),  0.5),   # ~20 min absorption
        tau_excrete  = _lognormal_sample(math.log(120.0), 0.5),   # ~2 hr excretion
        alpha_coffee = _lognormal_sample(math.log(1.5),   0.4),   # coffee ~1.5x diuretic
        alpha_tea    = _lognormal_sample(math.log(1.2),   0.4),   # tea ~1.2x diuretic
    )


# ---------------------------------------------------------------------------
# Physiological kernel and bladder load
# ---------------------------------------------------------------------------

def absorption_kernel(tau: float, p: Params) -> float:
    """
    Normalised difference-of-exponentials kernel:

        phi(tau) = (exp(-tau/tau_excrete) - exp(-tau/tau_absorb))
                   / (tau_excrete - tau_absorb)

    The denominator ensures integral_0^inf phi(tau) dtau = 1, so the total
    bladder contribution of drink i is exactly volume_i * alpha_i
    ml-equivalents regardless of which particle's time constants are used.
    The time constants control only the temporal shape of the response.
    Returns 0 for tau <= 0.
    """
    if tau <= 0.0:
        return 0.0
    excrete = math.exp(-tau / p.tau_excrete)
    absorb  = math.exp(-tau / p.tau_absorb)
    return max((excrete - absorb) / (p.tau_excrete - p.tau_absorb), 0.0)


def type_scaling(kind: DrinkKind, p: Params) -> float:
    if kind == DrinkKind.WATER:  return 1.0
    if kind == DrinkKind.COFFEE: return p.alpha_coffee
    if kind == DrinkKind.TEA:    return p.alpha_tea
    return 1.0


def bladder_load(t: float, drink_history: List[DrinkEvent], p: Params) -> float:
    """
    Effective bladder load at time t in ml-equivalents.

    Each drink contributes volume_i * alpha_i ml-equivalents in total
    (integrated over all time), distributed in time by the normalised kernel.
    ml-equivalents are model-internal units expressing diuretic load relative
    to water; they are not physical ml of urine.
    """
    total = 0.0
    for drink in drink_history:
        tau   = t - drink.time
        alpha = type_scaling(drink.kind, p)
        total += drink.volume * alpha * absorption_kernel(tau, p)
    return max(total, 0.0)


# ---------------------------------------------------------------------------
# Hazard rate
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def hazard(t: float, drink_history: List[DrinkEvent], p: Params) -> float:
    """Instantaneous hazard rate lambda(t)."""
    B = bladder_load(t, drink_history, p)
    g = _sigmoid(p.beta * (B - p.theta))
    return p.lambda0 * g


# ---------------------------------------------------------------------------
# Survival integral (negative log-survival)
# ---------------------------------------------------------------------------

def log_survival(
    t_start: float,
    t_end:   float,
    drink_history: List[DrinkEvent],
    p: Params,
    n_steps: int = 50,
) -> float:
    """
    Approximates  -∫_{t_start}^{t_end} λ(t) dt  using midpoint rule.
    Returns a non-positive float (log probability of surviving the interval).
    """
    if t_end <= t_start:
        return 0.0
    dt    = (t_end - t_start) / n_steps
    total = 0.0
    for k in range(n_steps):
        t      = t_start + (k + 0.5) * dt
        total += hazard(t, drink_history, p)
    return -total * dt


# ---------------------------------------------------------------------------
# Particle maintenance
# ---------------------------------------------------------------------------

def normalize_weights(particles: List[Particle]) -> None:
    total = sum(p.weight for p in particles)
    if total <= 0.0:
        # Catastrophic degeneracy: reset to uniform
        for p in particles:
            p.weight = 1.0 / len(particles)
    else:
        for p in particles:
            p.weight /= total


def effective_sample_size(particles: List[Particle]) -> float:
    return 1.0 / sum(p.weight ** 2 for p in particles)


def systematic_resample(particles: List[Particle]) -> List[Particle]:
    """
    Systematic resampling. Returns a new list of particles with equal weights.
    """
    n   = len(particles)
    u0  = random.random() / n
    cumw = 0.0
    new_particles: List[Particle] = []
    j = 0
    for i in range(n):
        target = u0 + i / n
        while cumw + particles[j].weight < target and j < n - 1:
            cumw += particles[j].weight
            j    += 1
        new_p        = copy.deepcopy(particles[j])
        new_p.weight = 1.0 / n
        new_particles.append(new_p)
    return new_particles


def resample_if_needed(particles: List[Particle], threshold: float = 0.5) -> List[Particle]:
    n = len(particles)
    if effective_sample_size(particles) < threshold * n:
        return systematic_resample(particles)
    return particles


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BladderUrgeModel:
    """
    Sequential Monte Carlo model for bladder urge estimation.

    Event processing:
        model.process_event(DrinkEvent(...))
        model.process_event(UrgeEvent(...))
        model.process_event(VoidEvent(...))

    Query:
        prob = model.query(t_now=90.0, horizon=30.0)
    """

    def __init__(self, n_particles: int = 1000):
        self.particles: List[Particle] = [
            Particle(params=sample_from_prior(), weight=1.0 / n_particles)
            for _ in range(n_particles)
        ]
        self.drink_history:    List[DrinkEvent] = []
        self.last_void_time:   float = 0.0
        self.current_time:     float = 0.0

    # ------------------------------------------------------------------
    # Internal: advance time (account for silence = no urge observed)
    # ------------------------------------------------------------------

    def _advance_time(self, new_time: float) -> None:
        """
        Weight update for a quiet interval [current_time, new_time].
        Silence is evidence: particles predicting high hazard are penalized.
        """
        prev_time = self.current_time
        self.current_time = new_time

        if new_time <= prev_time:
            return

        for p in self.particles:
            log_surv  = log_survival(prev_time, new_time, self.drink_history, p.params)
            p.weight *= math.exp(log_surv)

        normalize_weights(self.particles)
        self.particles = resample_if_needed(self.particles)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_drink(self, event: DrinkEvent) -> None:
        """
        Drink events are fully observed — they reshape future hazard via
        bladder_load, but do not themselves update particle weights.
        """
        self.drink_history.append(event)

    def _on_urge(self, event: UrgeEvent) -> None:
        """
        Positive evidence: an urge occurred.
        Weight proportional to λ(t_urge) (hazard density at urge time).
        Note: _advance_time must be called BEFORE this to account for the
        silent period leading up to the urge.
        """
        for p in self.particles:
            h         = hazard(event.time, self.drink_history, p.params)
            p.weight *= max(h, 1e-300)

        normalize_weights(self.particles)
        self.particles = resample_if_needed(self.particles)

    def _on_void(self, event: VoidEvent) -> None:
        """
        Voiding is observed, not predicted. Reset drink history
        (bladder emptied) but do not update weights.
        """
        self.drink_history  = []
        self.last_void_time = event.time

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_event(self, event) -> None:
        """
        Route any event type. Automatically advances time to account for
        the quiet interval before each event.
        """
        # Always advance time first (silence is evidence)
        self._advance_time(event.time)

        if isinstance(event, DrinkEvent):
            self._on_drink(event)
        elif isinstance(event, UrgeEvent):
            self._on_urge(event)
        elif isinstance(event, VoidEvent):
            self._on_void(event)
        else:
            raise TypeError(f"Unknown event type: {type(event)}")

    def query(self, t_now: float, horizon: float) -> float:
        """
        Returns P(urge occurs in [t_now, t_now + horizon]),
        averaged over the particle posterior.

        Parameters
        ----------
        t_now   : current time in minutes
        horizon : look-ahead window in minutes
        """
        self._advance_time(t_now)
        t_end = t_now + horizon

        estimate = 0.0
        for p in self.particles:
            log_surv = log_survival(t_now, t_end, self.drink_history, p.params)
            p_urge   = 1.0 - math.exp(log_surv)
            estimate += p.weight * p_urge

        return estimate

    def posterior_summary(self) -> dict:
        """
        Weighted posterior mean and std of each parameter.
        Useful for inspecting what the model has learned.
        """
        param_names = [
            "lambda0", "beta", "theta",
            "tau_absorb", "tau_excrete",
            "alpha_coffee", "alpha_tea",
        ]
        summary = {}
        for name in param_names:
            vals    = [getattr(p.params, name) for p in self.particles]
            weights = [p.weight for p in self.particles]
            mean    = sum(w * v for w, v in zip(weights, vals))
            var     = sum(w * (v - mean) ** 2 for w, v in zip(weights, vals))
            summary[name] = {"mean": mean, "std": math.sqrt(var)}
        return summary


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    model = BladderUrgeModel(n_particles=2000)

    # --- Observed drink history ---
    model.process_event(DrinkEvent(time=0.0,   volume=250.0, kind=DrinkKind.WATER))
    model.process_event(DrinkEvent(time=30.0,  volume=120.0, kind=DrinkKind.COFFEE))
    model.process_event(DrinkEvent(time=60.0,  volume=200.0, kind=DrinkKind.WATER))

    # --- Observed urge at t=100 (training signal) ---
    model.process_event(UrgeEvent(time=100.0))

    # --- Void at t=105 ---
    model.process_event(VoidEvent(time=105.0))

    # --- More drinks after void ---
    model.process_event(DrinkEvent(time=110.0, volume=350.0, kind=DrinkKind.TEA))
    model.process_event(DrinkEvent(time=140.0, volume=200.0, kind=DrinkKind.WATER))

    # --- No urge observed through t=180 (silence is also evidence) ---
    # (Handled automatically by process_event's _advance_time call)

    # --- Query ---
    for horizon in [15, 30, 60]:
        prob = model.query(t_now=180.0, horizon=float(horizon))
        print(f"P(urge in next {horizon:2d} min | t=180) = {prob:.3f}")

    print()
    print("Posterior parameter summary:")
    for name, stats in model.posterior_summary().items():
        print(f"  {name:15s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}")
