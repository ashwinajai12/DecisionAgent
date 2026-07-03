"""
QAC v1 -- Classical belief-adjustment (Markov) baseline. (Agent E deliverable.)

The FAIR classical competitor to the quantum model, on the SHARED INTERFACE:
    predict(stimulus, questions: list[str]) -> full-joint dict
      2 questions -> {"yy","yn","ny","nn"} keyed (first-asked ans, second-asked ans)
      1 question  -> {"y","n"}

Why this is the right baseline (and not scalar BELBIC):
  - It ALSO produces question-order effects -- asking A first shifts the
    propensity for B (assimilation/anchoring). So "order effects" alone are NOT
    evidence of quantumness.
  - But it does NOT force the Wang-Busemeyer QQ equality: q is GENERICALLY != 0.
  - It needs MORE free parameters (6) than the quantum model (~3 effective) to
    generate those order effects.
That asymmetry -- order effects come free classically, but q==0 only comes from
non-commuting projective measurement -- is exactly what the comparison tests.

Parameterization (per question-pair {A,B}, names set at construction):
  a_first, b_first : P(ans=y) for each question when asked FIRST (or alone)
  w_AB_y, w_AB_n   : logit shift applied to B when A (asked first) answered y / n
  w_BA_y, w_BA_n   : logit shift applied to A when B (asked first) answered y / n
Set all w == 0 -> IndependentModel behaviour (no order effect, q==0 trivially).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float, eps: float = 1e-9) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


@dataclass
class ClassicalMarkovModel:
    """Belief-adjustment baseline. Same predict() contract as QuantumJudgmentModel."""
    a_first: float
    b_first: float
    w_AB_y: float = 0.0
    w_AB_n: float = 0.0
    w_BA_y: float = 0.0
    w_BA_n: float = 0.0
    names: tuple = ("A", "B")

    # ---- shared interface --------------------------------------------------
    def predict(self, stimulus=None, questions=None) -> dict:
        if questions is None:
            raise ValueError("questions (measurement order) required")
        order = list(questions)
        nameA, nameB = self.names

        if len(order) == 1:
            q = order[0]
            py = self.a_first if q == nameA else self.b_first
            return {"y": py, "n": 1.0 - py}

        if len(order) != 2:
            raise ValueError("baseline supports 1 or 2 questions")

        first, second = order
        if first == nameA:                      # order A then B
            pf = self.a_first
            pB_y = _sigmoid(_logit(self.b_first) + self.w_AB_y)   # A answered y
            pB_n = _sigmoid(_logit(self.b_first) + self.w_AB_n)   # A answered n
            yy = pf * pB_y
            yn = pf * (1.0 - pB_y)
            ny = (1.0 - pf) * pB_n
            nn = (1.0 - pf) * (1.0 - pB_n)
        elif first == nameB:                    # order B then A
            pf = self.b_first
            pA_y = _sigmoid(_logit(self.a_first) + self.w_BA_y)   # B answered y
            pA_n = _sigmoid(_logit(self.a_first) + self.w_BA_n)   # B answered n
            yy = pf * pA_y
            yn = pf * (1.0 - pA_y)
            ny = (1.0 - pf) * pA_n
            nn = (1.0 - pf) * (1.0 - pA_n)
        else:
            raise KeyError(f"unknown question {first!r}; model knows {self.names}")

        # keys are POSITIONAL: (first-asked answer, second-asked answer)
        return {"yy": yy, "yn": yn, "ny": ny, "nn": nn}

    # ---- fitting helpers ---------------------------------------------------
    @staticmethod
    def n_params() -> int:
        return 6

    @classmethod
    def from_params(cls, theta: np.ndarray, names=("A", "B")) -> "ClassicalMarkovModel":
        """Build from an UNCONSTRAINED 6-vector (base rates via logit space)."""
        t = np.asarray(theta, dtype=float)
        return cls(
            a_first=_sigmoid(t[0]), b_first=_sigmoid(t[1]),
            w_AB_y=t[2], w_AB_n=t[3], w_BA_y=t[4], w_BA_n=t[5],
            names=names,
        )


@dataclass
class IndependentModel:
    """Trivial classical control: two independent binary questions, NO order effect.

    Gets q == 0 -- but ONLY by having no order effect at all. The quantum model's
    interest is that it gets q == 0 WHILE still producing genuine order effects.
    """
    a_yes: float
    b_yes: float
    names: tuple = ("A", "B")

    def predict(self, stimulus=None, questions=None) -> dict:
        if questions is None:
            raise ValueError("questions (measurement order) required")
        order = list(questions)
        nameA, nameB = self.names
        rate = {nameA: self.a_yes, nameB: self.b_yes}
        if len(order) == 1:
            py = rate[order[0]]
            return {"y": py, "n": 1.0 - py}
        pf, ps = rate[order[0]], rate[order[1]]
        return {"yy": pf * ps, "yn": pf * (1 - ps),
                "ny": (1 - pf) * ps, "nn": (1 - pf) * (1 - ps)}


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.quantum_qubit import qq_residual as _quantum_qq  # noqa: F401

    # A Markov model with asymmetric assimilation -> real order effect AND q != 0.
    m = ClassicalMarkovModel(a_first=0.55, b_first=0.45,
                             w_AB_y=1.2, w_AB_n=-0.8, w_BA_y=0.3, w_BA_n=-0.2)
    ab = m.predict(questions=["A", "B"])
    ba = m.predict(questions=["B", "A"])
    # empirical q on the model's own joints (A_ans,B_ans convention):
    s_ab = ab["yn"] + ab["ny"]                 # (A=y,B=n)+(A=n,B=y)
    s_ba = ba["ny"] + ba["yn"]                 # BA positional -> (A=y,B=n)+(A=n,B=y)
    print("Markov  AB:", {k: round(v, 4) for k, v in ab.items()})
    print("Markov  BA:", {k: round(v, 4) for k, v in ba.items()})
    print("Markov  q =", round(s_ab - s_ba, 6), "  (classical: generically != 0)")

    ind = IndependentModel(a_yes=0.55, b_yes=0.45)
    iab, iba = ind.predict(questions=["A", "B"]), ind.predict(questions=["B", "A"])
    print("Indep   q =", round((iab["yn"] + iab["ny"]) - (iba["ny"] + iba["yn"]), 6),
          "  (no order effect -> q == 0 trivially)")
