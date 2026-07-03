"""
QAC v1 -- Evaluation metrics. (Agent E deliverable.)

The point the expert review flagged: quantum_qubit.qq_residual(MODEL, ...) is a
THEOREM CHECK on a model -- it is identically 0 and discriminates nothing against
data. Real evaluation needs (a) an EMPIRICAL q from observed arms and (b) a model
that does NOT force q==0 (the Markov baseline) to compare against. This module
supplies the empirical/data-facing metrics.

All joint dicts use the model's POSITIONAL keys ("yy","yn","ny","nn") =
(first-asked answer, second-asked answer); observed joints come from
OrderEffectArm.observed_joint(), which already matches that convention.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.quantum_qubit import qq_residual, interference_delta  # noqa: F401,E402


# ---- distribution distances ------------------------------------------------
def tvd(p: dict, q: dict) -> float:
    """Total variation distance over shared keys: 0.5 * sum|p_k - q_k| in [0,1]."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def multinomial_loglik(counts: dict, probs: dict, eps: float = 1e-12) -> float:
    """Sum_k count_k * log(prob_k). (Constant multinomial coefficient dropped.)"""
    return float(sum(c * np.log(max(probs.get(k, 0.0), eps)) for k, c in counts.items()))


# ---- empirical order-effect metric (data, not model) -----------------------
def empirical_qq(arm_ab, arm_ba) -> float:
    """
    Wang-Busemeyer q measured FROM DATA. Uses the arms' (A_ans,B_ans) proportions,
    which are order-agnostic (the CSV stores explicit answer columns):
        q = [p_AB(A=y,B=n) + p_AB(A=n,B=y)] - [p_BA(A=y,B=n) + p_BA(A=n,B=y)]
    Quantum prediction: q == 0. Any statistically reliable q != 0 across surveys
    is what a classical Markov model must strain to fit and quantum gets for free.
    """
    pab = arm_ab.proportions()   # keys ("y","y"),("y","n"),... = (A_ans, B_ans)
    pba = arm_ba.proportions()
    s_ab = pab[("y", "n")] + pab[("n", "y")]
    s_ba = pba[("y", "n")] + pba[("n", "y")]
    return float(s_ab - s_ba)


def arm_counts(arm) -> dict:
    """Observed COUNTS in positional keys (aligned with model.predict order)."""
    n = arm.N_arm or sum(arm.cells.values())
    return {k: pr * n for k, pr in arm.observed_joint().items()}


def arm_order(arm) -> list:
    """Measurement order for this arm as model question names."""
    return ["A", "B"] if arm.order == "AB" else ["B", "A"]


def fit_tvd(model, arms: dict) -> float:
    """Mean TVD between model prediction and observed joint, over the survey's arms."""
    vals = []
    for arm in arms.values():
        pred = model.predict(questions=arm_order(arm))
        vals.append(tvd(pred, arm.observed_joint()))
    return float(np.mean(vals)) if vals else float("nan")


def neg_loglik(model, arms: dict) -> float:
    """Total negative multinomial log-likelihood of the arms under the model."""
    ll = 0.0
    for arm in arms.values():
        pred = model.predict(questions=arm_order(arm))
        ll += multinomial_loglik(arm_counts(arm), pred)
    return float(-ll)


def model_qq(model) -> float:
    """Convenience: the model's own q (quantum -> ~0; Markov -> generally != 0)."""
    return qq_residual(model, "A", "B")
