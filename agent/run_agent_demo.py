"""
QAC v1 -- Affective-agent demonstration (honest version).

Three parts, from weakest to strongest claim:

  (A) A scalar emotional value destroys information: it conflates every balanced
      conflict (E=0, p=0.5). A richer STATE fixes this -- but note a 2-number
      classical vector already distinguishes 'torn' from 'indifferent', so richness
      alone is NOT a quantum result.

  (B) HONESTY CHECK. The rotation-based "quantum-inspired" agent is reproduced to
      ~1e-16 by a plain 3-vector classical agent. A single qubit's unitary evolution
      IS classical rotation, so the order effect it shows is NOT quantum.

  (C) THE GENUINE SIGNATURE. A measurement-based agent (appraisals = projective
      measurements with collapse) shows order effects that satisfy the Wang-Busemeyer
      QQ equality (q ~ 0) -- the constraint classical Markov models cannot force and
      the one validated on real survey data (eval/run_comparison.py).

Run:  python qac/agent/run_agent_demo.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.affect import (                                    # noqa: E402
    RotationAffectiveAgent, ClassicalRotationAgent, MeasurementAffectiveAgent,
    ClassicalAffectiveAgent, TwoChannelClassicalAgent, Scenario, default_scenarios,
)

BAL = ["angry_but_helping", "risky_but_rewarding", "trusted_but_aggressive"]


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    scen = default_scenarios()
    c1, c2 = ClassicalAffectiveAgent(), TwoChannelClassicalAgent()
    rot = RotationAffectiveAgent()

    section("(A) A scalar destroys information (all these have net E = 0)")
    print(f"{'scenario':24s} {'scalar E':>9s} {'2vec(app,avd)':>15s} {'rot p_approach':>15s}")
    for name in ["neutral"] + BAL:
        s = scen[name]
        print(f"{name:24s} {c1.appraise(s)['E']:>+9.2f} "
              f"{'(%.1f,%.1f)' % (c2.appraise(s)['approach'], c2.appraise(s)['avoid']):>15s} "
              f"{rot.appraise(s)['p_approach']:>15.3f}")
    print("  scalar: identical for all -> info destroyed. 2-vec: separates them (NOT quantum).")

    section("(B) HONESTY CHECK: the rotation agent is CLASSICAL (twin matches to ~1e-16)")
    ctwin = ClassicalRotationAgent()
    worst = 0.0
    print(f"{'scenario':24s} {'rot(qubit) p':>13s} {'classical p':>12s} {'|diff|':>9s}")
    for name in BAL:
        s = scen[name]
        pr, pc = rot.appraise(s)["p_approach"], ctwin.appraise(s)["p_approach"]
        worst = max(worst, abs(pr - pc))
        print(f"{name:24s} {pr:>13.6f} {pc:>12.6f} {abs(pr - pc):>9.1e}")
    print(f"  worst discrepancy = {worst:.1e}. The qubit adds NOTHING here; its 'order")
    print("  effect' is rotation non-commutativity, which is classical. Not a quantum win.")

    section("(C) THE GENUINE SIGNATURE: measurement agent -- order effect WITH QQ q~0")
    m = MeasurementAffectiveAgent()
    p_tr = m.p_approach(("threat", "reward"))
    p_rt = m.p_approach(("reward", "threat"))
    print(f"  appraise threat->reward : p_approach = {p_tr:.4f}")
    print(f"  appraise reward->threat : p_approach = {p_rt:.4f}")
    print(f"  order effect |Δp|       = {abs(p_tr - p_rt):.4f}   (appraisal order matters)")
    print(f"  QQ residual q           = {m.qq():.2e}   (== 0: the quantum-probability signature)")
    print("  A classical Markov appraisal model produces order effects too, but q != 0")
    print("  (see tests/test_baselines). Forcing q == 0 is the part validated on real data.")
    print("  NB: a single qubit is still classically SIMULABLE -- the claim is about matching")
    print("  a data constraint classical-PROBABILITY models violate, never hardware speedup.")


if __name__ == "__main__":
    main()
