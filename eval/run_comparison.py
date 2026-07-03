"""
QAC v1 -- End-to-end comparison harness. (Agent E deliverable.)

Loads order-effect survey arms, fits the quantum model and the classical Markov
baseline to each survey, and reports, per survey:
    q_data        empirical Wang-Busemeyer q (from observed cells)
    q_markov      the fitted Markov model's own q  (classical: generally != 0)
    q_quantum     the fitted quantum model's own q (structurally ~0)
    in-sample fit (neg log-lik / TVD) for each model
    HELD-OUT cross-order fit (fit one order, predict the other) -- the honest test

Run:  python qac/eval/run_comparison.py            # uses bundled data
      python qac/eval/run_comparison.py PATH.csv   # your own order-effect CSV

NOTE: the bundled CSV is SYNTHETIC (flagged DO_NOT_FIT). This run is a PIPELINE
SMOKE TEST, not a result. Point it at PNAS-2014-SI / Pew-derived cells for a real
comparison. The harness prints a banner if it detects synthetic sources.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loaders import load_order_effect                       # noqa: E402
from eval.metrics import empirical_qq, fit_tvd, neg_loglik, model_qq  # noqa: E402
from eval.fit import fit_quantum, fit_markov, fit_heldout         # noqa: E402


def _is_synthetic(surveys) -> bool:
    for arms in surveys.values():
        for arm in arms.values():
            if "SYNTHETIC" in (arm.source or "").upper() or "TEMPLATE" in arm.survey_id.upper():
                return True
    return False


def run(path=None):
    surveys = load_order_effect(path)
    synthetic = _is_synthetic(surveys)

    print("=" * 78)
    print("QAC order-effect comparison: quantum vs classical-Markov")
    if synthetic:
        print("!! SMOKE TEST ONLY -- data is SYNTHETIC (DO_NOT_FIT). Not a result. !!")
    print("=" * 78)

    rows = []
    for sid, arms in sorted(surveys.items()):
        if not ({"AB", "BA"} <= set(arms)):
            print(f"[skip] {sid}: needs both AB and BA arms")
            continue

        q_data = empirical_qq(arms["AB"], arms["BA"])
        qm, nll_m, _ = fit_markov(arms)
        qq, nll_q, _ = fit_quantum(arms)

        ho_m = fit_heldout(fit_markov, arms)
        ho_q = fit_heldout(fit_quantum, arms)

        row = {
            "survey": sid,
            "q_data": q_data,
            "q_markov": model_qq(qm),
            "q_quantum": model_qq(qq),
            "nll_markov": nll_m, "nll_quantum": nll_q,
            "tvd_markov": fit_tvd(qm, arms), "tvd_quantum": fit_tvd(qq, arms),
            "ho_nll_markov": ho_m["mean_neg_loglik"], "ho_nll_quantum": ho_q["mean_neg_loglik"],
            "ho_tvd_markov": ho_m["mean_tvd"], "ho_tvd_quantum": ho_q["mean_tvd"],
        }
        rows.append(row)

        print(f"\n--- {sid} ---")
        print(f"  q_data      = {q_data:+.4f}   (quantum predicts model q == 0)")
        print(f"  q_markov    = {row['q_markov']:+.4f}   q_quantum = {row['q_quantum']:+.2e}")
        print(f"  in-sample   NLL  markov={nll_m:8.2f}  quantum={nll_q:8.2f}")
        print(f"  in-sample   TVD  markov={row['tvd_markov']:.4f}   quantum={row['tvd_quantum']:.4f}")
        print(f"  HELD-OUT    NLL  markov={row['ho_nll_markov']:8.2f}  quantum={row['ho_nll_quantum']:8.2f}")
        print(f"  HELD-OUT    TVD  markov={row['ho_tvd_markov']:.4f}   quantum={row['ho_tvd_quantum']:.4f}")

    if rows:
        print("\n" + "=" * 78)
        print("SUMMARY (mean over surveys)")
        for key, label in [("ho_nll_markov", "held-out NLL  markov"),
                           ("ho_nll_quantum", "held-out NLL  quantum"),
                           ("ho_tvd_markov", "held-out TVD  markov"),
                           ("ho_tvd_quantum", "held-out TVD  quantum")]:
            print(f"  {label:24s} = {np.mean([r[key] for r in rows]):.4f}")
        print("  Interpretation: quantum forces q==0 with fewer params; if its held-out")
        print("  fit is COMPARABLE to Markov's despite that constraint, that is the win.")
        print("=" * 78)
    return rows


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
