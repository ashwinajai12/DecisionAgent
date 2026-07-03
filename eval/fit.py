"""
QAC v1 -- Fitting + held-out evaluation harness. (Agent E deliverable.)

No scipy in the environment, so this ships a small dependency-free Nelder-Mead
with multi-start. Two fitters, both minimizing negative multinomial log-likelihood
of a survey's arms:

  fit_quantum(arms) : 6 angles (pure init state 2 + two question axes 2+2).
                      The quantum model STRUCTURALLY forces q==0, so it cannot
                      "cheat" order effects into the fit -- that is the test.
  fit_markov(arms)  : 6 unconstrained params (2 base rates + 4 assimilation shifts).

Held-out protocol (fit_heldout): fit on ONE order-arm, score the OTHER arm's
log-likelihood. This is the informative comparison -- per-arm in-sample fit is
near-saturated (the review's parameter-counting caveat), so we report cross-order
generalization, where forcing q==0 actually costs or helps.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.quantum_qubit import QuantumJudgmentModel, Question           # noqa: E402
from models.classical_markov import ClassicalMarkovModel                  # noqa: E402
from eval.metrics import multinomial_loglik, arm_counts, arm_order, tvd   # noqa: E402


# ---- dependency-free optimizer --------------------------------------------
def nelder_mead(f, x0, step=0.4, max_iter=1200, tol=1e-10, no_improv_break=60,
                alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)
    simplex = [[x0, f(x0)]]
    for i in range(dim):
        x = np.array(x0, dtype=float)
        x[i] += step
        simplex.append([x, f(x)])
    prev_best = simplex[0][1]
    no_improv = 0
    for _ in range(max_iter):
        simplex.sort(key=lambda r: r[1])
        best = simplex[0][1]
        if best < prev_best - tol:
            prev_best, no_improv = best, 0
        else:
            no_improv += 1
        if no_improv >= no_improv_break:
            break
        centroid = np.mean([r[0] for r in simplex[:-1]], axis=0)
        worst = simplex[-1][0]
        xr = centroid + alpha * (centroid - worst)
        fr = f(xr)
        if simplex[0][1] <= fr < simplex[-2][1]:
            simplex[-1] = [xr, fr]
            continue
        if fr < simplex[0][1]:                                   # expand
            xe = centroid + gamma * (centroid - worst)
            fe = f(xe)
            simplex[-1] = [xe, fe] if fe < fr else [xr, fr]
            continue
        xc = centroid + rho * (worst - centroid)                 # contract
        fc = f(xc)
        if fc < simplex[-1][1]:
            simplex[-1] = [xc, fc]
            continue
        x1 = simplex[0][0]                                        # shrink
        simplex = [[x1, simplex[0][1]]] + [
            [x1 + sigma * (r[0] - x1), f(x1 + sigma * (r[0] - x1))] for r in simplex[1:]
        ]
    simplex.sort(key=lambda r: r[1])
    return simplex[0][0], simplex[0][1]


def _multistart(f, x0s):
    best_x, best_f = None, np.inf
    for x0 in x0s:
        x, fx = nelder_mead(f, x0)
        if fx < best_f:
            best_x, best_f = x, fx
    return best_x, best_f


# ---- model constructors from parameter vectors -----------------------------
def _axis(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def quantum_from_params(p):
    """6 angles -> QuantumJudgmentModel with a PURE init state (|r|==1, always valid)."""
    th0, ph0, thA, phA, thB, phB = p
    return QuantumJudgmentModel(
        init_bloch=_axis(th0, ph0),
        questions={"A": Question("A", _axis(thA, phA)),
                   "B": Question("B", _axis(thB, phB))},
    )


def _neg_loglik_model(model, arms):
    return -sum(multinomial_loglik(arm_counts(a), model.predict(questions=arm_order(a)))
                for a in arms.values())


# ---- fitters ---------------------------------------------------------------
def _seed_grid(rng, n, dim, scale):
    return [rng.uniform(-scale, scale, size=dim) for _ in range(n)]


def fit_quantum(arms, n_starts=8, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda p: _neg_loglik_model(quantum_from_params(p), arms)
    starts = [np.array([1.2, 0.3, 0.6, 0.0, 1.0, 0.4])]
    starts += [np.array([1.5, 0.0, 1.0, 0.0, 1.0, 0.0]) + rng.normal(0, 0.6, 6)
               for _ in range(n_starts - 1)]
    p, nll = _multistart(f, starts)
    return quantum_from_params(p), nll, p


def fit_markov(arms, n_starts=8, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda p: _neg_loglik_model(ClassicalMarkovModel.from_params(p), arms)
    starts = [np.zeros(6)]
    starts += [rng.normal(0, 1.0, 6) for _ in range(n_starts - 1)]
    p, nll = _multistart(f, starts)
    return ClassicalMarkovModel.from_params(p), nll, p


# ---- held-out cross-order evaluation ---------------------------------------
def fit_heldout(fit_fn, arms):
    """
    For each order-arm: fit on the OTHER arm, score this arm.
    Returns dict {held_out_order: {"neg_loglik":.., "tvd":..}} plus the mean.
    Requires both "AB" and "BA" arms present.
    """
    if not ({"AB", "BA"} <= set(arms)):
        return None
    out = {}
    for held in ("AB", "BA"):
        train = {k: v for k, v in arms.items() if k != held}
        model, _, _ = fit_fn(train)
        test_arm = arms[held]
        pred = model.predict(questions=arm_order(test_arm))
        out[held] = {
            "neg_loglik": -multinomial_loglik(arm_counts(test_arm), pred),
            "tvd": tvd(pred, test_arm.observed_joint()),
        }
    out["mean_neg_loglik"] = float(np.mean([out[o]["neg_loglik"] for o in ("AB", "BA")]))
    out["mean_tvd"] = float(np.mean([out[o]["tvd"] for o in ("AB", "BA")]))
    return out
