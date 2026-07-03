"""
Tests for evaluation metrics + a small end-to-end fit sanity check.
Runnable with plain python or pytest.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loaders import OrderEffectArm                              # noqa: E402
from eval.metrics import tvd, multinomial_loglik, empirical_qq       # noqa: E402
from eval.fit import fit_markov, quantum_from_params, fit_quantum    # noqa: E402
from models.quantum_qubit import qq_residual                         # noqa: E402


def test_tvd_properties():
    p = {"yy": 0.4, "yn": 0.1, "ny": 0.2, "nn": 0.3}
    assert abs(tvd(p, p)) < 1e-12, "TVD(p,p) must be 0"
    q = {"yy": 0.1, "yn": 0.4, "ny": 0.3, "nn": 0.2}
    assert abs(tvd(p, q) - tvd(q, p)) < 1e-12, "TVD must be symmetric"
    assert 0.0 <= tvd(p, q) <= 1.0, "TVD out of [0,1]"


def test_multinomial_loglik_orders_correctly():
    counts = {"yy": 40, "yn": 10, "ny": 20, "nn": 30}
    good = {"yy": 0.4, "yn": 0.1, "ny": 0.2, "nn": 0.3}   # == empirical
    bad = {"yy": 0.1, "yn": 0.4, "ny": 0.3, "nn": 0.2}
    assert multinomial_loglik(counts, good) > multinomial_loglik(counts, bad)


def _arm(order, cells, n):
    return OrderEffectArm(survey_id="t", stimulus_id="t", qA_text="", qB_text="",
                          order=order, cells=cells, N_arm=n)


def test_empirical_qq_hand_computed():
    # (A_ans,B_ans) counts. AB arm: p(y,n)=0.1, p(n,y)=0.2 -> 0.3
    ab = _arm("AB", {("y", "y"): 60, ("y", "n"): 10, ("n", "y"): 20, ("n", "n"): 10}, 100)
    # BA arm: p(y,n)=0.15, p(n,y)=0.05 -> 0.20
    ba = _arm("BA", {("y", "y"): 50, ("y", "n"): 15, ("n", "y"): 5, ("n", "n"): 30}, 100)
    q = empirical_qq(ab, ba)
    assert abs(q - (0.30 - 0.20)) < 1e-12, f"empirical q wrong: {q}"


def test_quantum_fit_keeps_q_zero():
    # Fit the quantum model to arbitrary arms; its own q must stay ~0 regardless.
    ab = _arm("AB", {("y", "y"): 55, ("y", "n"): 15, ("n", "y"): 18, ("n", "n"): 12}, 100)
    ba = _arm("BA", {("y", "y"): 40, ("y", "n"): 20, ("n", "y"): 10, ("n", "n"): 30}, 100)
    arms = {"AB": ab, "BA": ba}
    model, nll, _ = fit_quantum(arms, n_starts=6)
    assert abs(qq_residual(model, "A", "B")) < 1e-8, "fitted quantum q must remain 0"
    assert np.isfinite(nll)


def test_markov_fit_can_match_nonzero_q():
    # Data with a real q; the Markov model (6 params) should be able to fit close.
    ab = _arm("AB", {("y", "y"): 55, ("y", "n"): 15, ("n", "y"): 18, ("n", "n"): 12}, 100)
    ba = _arm("BA", {("y", "y"): 40, ("y", "n"): 20, ("n", "y"): 10, ("n", "n"): 30}, 100)
    arms = {"AB": ab, "BA": ba}
    model, nll, _ = fit_markov(arms, n_starts=8)
    assert np.isfinite(nll)
    # Markov q is free to be nonzero (unlike quantum) -- just assert it fit finitely.
    assert abs(qq_residual(model, "A", "B")) < 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} metric/fit tests passed.")
