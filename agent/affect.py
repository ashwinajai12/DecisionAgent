"""
QAC v1 -- Affective-agent layer. Two honestly-labelled kinds of agent.

The emotional state is one qubit:  |0> = approach/engage (trust),  |1> = avoid/withdraw (fear).
The decision is a measurement along Z:  p_approach = (1 + <Z>) / 2.

IMPORTANT HONESTY NOTE (verified by test_affect + ClassicalRotationAgent):
A single qubit's UNITARY evolution is exactly classical 3D rotation (SU(2) -> SO(3)).
So an agent that integrates cues by *rotating* the Bloch vector is CLASSICALLY
SIMULABLE, and any "order effect" it shows is the non-commutativity of ROTATIONS --
which is NOT quantum (a plain 3-vector Rodrigues model reproduces it to 1e-16).

The genuinely non-classical-*probability* signature lives in MEASUREMENT
non-commutativity: sequential projective appraisals (with collapse) whose order
effects satisfy the Wang-Busemeyer QQ equality (q == 0) that classical Markov/Bayesian
models cannot force. That -- not rotation -- is what ties to the validated survey data
(eval/run_comparison.py). Even then a single qubit is simulable; the claim is about
matching a data CONSTRAINT classical-probability models violate, never hardware speedup.

Classes:
  RotationAffectiveAgent   quantum-inspired GEOMETRY; cues = unitary rotations.
                           Classically simulable (see ClassicalRotationAgent). Useful as
                           a rich state representation, NOT as evidence of quantumness.
  ClassicalRotationAgent   plain 3-vector twin of the above; proves the qubit adds nothing
                           to a rotation-only agent.
  MeasurementAffectiveAgent appraisals = projective MEASUREMENTS; order effects satisfy the
                           QQ equality -- the signature validated against real data.
  ClassicalAffectiveAgent  scalar BELBIC baseline (E = sum(approach) - sum(avoid)).
  TwoChannelClassicalAgent 2-vector baseline (keeps both tendencies; order-blind).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.quantum_qubit import (                          # noqa: E402
    I2, SX, SY, SZ, bloch_to_rho, coherence, purity,
    Question, QuantumJudgmentModel, qq_residual,
)


# ---- primitives ------------------------------------------------------------
def rotation(angle: float, axis) -> np.ndarray:
    """Single-qubit unitary exp(-i angle/2 (n.sigma)), closed form (no scipy)."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    nsig = n[0] * SX + n[1] * SY + n[2] * SZ
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * nsig


def rodrigues(v, axis, theta):
    """Classical 3D rotation of vector v about `axis` by `theta` (no quantum objects)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    v = np.asarray(v, dtype=float)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))


def _expect(op, rho) -> float:
    return float(np.trace(op @ rho).real)


def bloch_vec(rho) -> tuple:
    return (_expect(SX, rho), _expect(SY, rho), _expect(SZ, rho))


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def bloch_distance(rho1, rho2) -> float:
    return float(np.linalg.norm(np.array(bloch_vec(rho1)) - np.array(bloch_vec(rho2))))


# ---- stimulus --------------------------------------------------------------
@dataclass
class Cue:
    """Appraisal signal. valence sign = approach(+)/avoid(-); |valence| = intensity.
    channel = appraisal-axis azimuth (radians)."""
    name: str
    valence: float
    channel: float = 0.0


@dataclass
class Scenario:
    name: str
    description: str
    cues: list = field(default_factory=list)

    def net_valence(self) -> float:
        return sum(c.valence for c in self.cues)


# ---- rotation-based agent (quantum-inspired GEOMETRY; classically simulable) ----
@dataclass
class RotationAffectiveAgent:
    kappa: float = 1.2
    init_bloch: tuple = (1.0, 0.0, 0.0)      # |+x>: aroused, undecided, coherent

    def _axis(self, channel):
        return (0.0, np.cos(channel), np.sin(channel))   # Y-Z plane so cues move Z (decision)

    def state_after(self, cues) -> np.ndarray:
        rho = bloch_to_rho(np.array(self.init_bloch, dtype=float))
        for c in cues:
            U = rotation(-self.kappa * c.valence, self._axis(c.channel))
            rho = U @ rho @ U.conj().T
        return rho

    def appraise(self, scenario: Scenario) -> dict:
        rho = self.state_after(scenario.cues)
        p_app = (1 + _expect(SZ, rho)) / 2
        return {"p_approach": p_app, "decision_entropy": binary_entropy(p_app),
                "coherence": coherence(rho), "purity": purity(rho),
                "bloch": bloch_vec(rho), "rho": rho}


@dataclass
class ClassicalRotationAgent:
    """Plain-3-vector twin of RotationAffectiveAgent. Same kappa/axes/angles, no qubit.
    Reproduces the quantum-inspired agent to ~1e-16 -> proves that agent is classical."""
    kappa: float = 1.2
    init: tuple = (1.0, 0.0, 0.0)

    def state_after(self, cues):
        r = np.array(self.init, dtype=float)
        for c in cues:
            axis = (0.0, np.cos(c.channel), np.sin(c.channel))
            r = rodrigues(r, axis, -self.kappa * c.valence)
        return r

    def appraise(self, scenario: Scenario) -> dict:
        r = self.state_after(scenario.cues)
        p_app = (1 + r[2]) / 2
        return {"p_approach": p_app, "decision_entropy": binary_entropy(p_app),
                "coherence": float(np.hypot(r[0], r[1])), "bloch": tuple(r)}


# ---- measurement-based agent (the genuine QQ signature) --------------------
@dataclass
class MeasurementAffectiveAgent:
    """
    Appraisals are projective MEASUREMENTS (collapse), not rotations. The agent holds
    an affective state and, when it appraises ("is it threatening?", "is it rewarding?"),
    the state collapses. Because the two appraisals are incompatible (non-commuting
    projectors), the ORDER of appraisal changes the outcome -- and those order effects
    satisfy the Wang-Busemeyer QQ equality (q == 0), the signature a classical Markov
    model cannot force and the one validated on real survey data.

    Decision: approach iff the (final) reward appraisal is 'yes'. p_approach = P(reward=y).
    """
    init_bloch: tuple = (0.3, 0.0, 0.6)
    threat_axis: tuple = (0.0, 0.0, 1.0)
    reward_axis: tuple = (float(np.sin(1.1)), 0.0, float(np.cos(1.1)))  # incompatible

    def _model(self) -> QuantumJudgmentModel:
        return QuantumJudgmentModel(
            init_bloch=np.array(self.init_bloch, dtype=float),
            questions={"threat": Question("threat", self.threat_axis),
                       "reward": Question("reward", self.reward_axis)},
        )

    def appraise(self, order=("threat", "reward")) -> dict:
        """Full joint over the two appraisals in the given order (keys: first,second)."""
        return self._model().predict(questions=list(order))

    def p_approach(self, order=("threat", "reward")) -> float:
        """Marginal P(reward = yes) under this appraisal order."""
        joint = self.appraise(order)
        r_index = 1 if order[0] == "threat" else 0     # position of 'reward' answer in keys
        return sum(v for k, v in joint.items() if k[r_index] == "y")

    def order_effect(self) -> float:
        return abs(self.p_approach(("threat", "reward")) - self.p_approach(("reward", "threat")))

    def qq(self) -> float:
        return qq_residual(self._model(), "threat", "reward")   # ~0 by construction


# ---- classical decision baselines -----------------------------------------
@dataclass
class ClassicalAffectiveAgent:
    """Scalar BELBIC: E = sum(approach) - sum(avoid). Conflates all balanced conflicts."""
    gain: float = 1.0

    def appraise(self, scenario: Scenario) -> dict:
        E = scenario.net_valence()
        p = 1.0 / (1.0 + np.exp(-self.gain * E))
        return {"E": E, "p_approach": p, "decision_entropy": binary_entropy(p)}


@dataclass
class TwoChannelClassicalAgent:
    """2-vector: keeps approach and avoid separately (distinguishes torn vs indifferent),
    but sums COMMUTE -> order-blind."""
    gain: float = 1.0

    def appraise(self, scenario: Scenario) -> dict:
        approach = sum(c.valence for c in scenario.cues if c.valence > 0)
        avoid = -sum(c.valence for c in scenario.cues if c.valence < 0)
        p = 1.0 / (1.0 + np.exp(-self.gain * (approach - avoid)))
        return {"approach": approach, "avoid": avoid, "conflict": min(approach, avoid),
                "p_approach": p, "decision_entropy": binary_entropy(p)}


# ---- scenarios -------------------------------------------------------------
def default_scenarios() -> dict:
    return {
        "angry_but_helping": Scenario(
            "angry_but_helping", "Angry tone (avoid) but asking for help (approach).",
            [Cue("angry_tone", -1.5, 0.0), Cue("asks_for_help", +1.5, 1.4)]),
        "risky_but_rewarding": Scenario(
            "risky_but_rewarding", "High risk (avoid) but high reward (approach).",
            [Cue("risk", -1.5, 0.15), Cue("reward", +1.5, 0.30)]),
        "trusted_but_aggressive": Scenario(
            "trusted_but_aggressive", "Trusted before (approach) but now aggressive (avoid).",
            [Cue("past_trust", +1.6, 0.0), Cue("now_aggressive", -1.6, 1.5)]),
        "neutral": Scenario("neutral", "No strong cues.", []),
        "confident_approach": Scenario(
            "confident_approach", "One strong approach cue.", [Cue("clear_good", +1.6, 0.0)]),
    }


if __name__ == "__main__":
    r, c = RotationAffectiveAgent(), ClassicalRotationAgent()
    for name, s in default_scenarios().items():
        ra, ca = r.appraise(s), c.appraise(s)
        print(f"[{name:22s}] rot p={ra['p_approach']:.4f}  classical-twin p={ca['p_approach']:.4f}  "
              f"|diff|={abs(ra['p_approach']-ca['p_approach']):.1e}")
    m = MeasurementAffectiveAgent()
    print(f"\nMeasurement agent: order effect |Δp_approach| = {m.order_effect():.4f}, "
          f"QQ residual q = {m.qq():.2e}  (q~0 is the validated signature)")
