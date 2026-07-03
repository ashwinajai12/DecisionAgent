"""
Tests for the affective-agent layer. Plain-python runnable or pytest.

Honest contracts:
  - agent states are valid density matrices; decisions are probabilities;
  - the scalar classical agent CONFLATES distinct balanced conflicts;
  - the rotation ("quantum-inspired") agent is CLASSICAL: a plain 3-vector twin
    reproduces it to ~1e-15 (so its order effect is NOT a quantum result);
  - the MEASUREMENT agent shows an order effect that satisfies the QQ equality
    (q ~ 0) -- the genuine, data-validated signature.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.affect import (                                    # noqa: E402
    RotationAffectiveAgent, ClassicalRotationAgent, MeasurementAffectiveAgent,
    ClassicalAffectiveAgent, Scenario, Cue, default_scenarios, bloch_distance, rotation,
)
from models.quantum_qubit import I2                            # noqa: E402

BAL = ["angry_but_helping", "risky_but_rewarding", "trusted_but_aggressive"]


def test_rotation_is_unitary():
    rng = np.random.default_rng(3)
    for _ in range(200):
        U = rotation(rng.uniform(-np.pi, np.pi), rng.normal(size=3))
        assert np.allclose(U @ U.conj().T, I2, atol=1e-12), "rotation not unitary"


def test_rotation_state_is_valid_density_matrix():
    r = RotationAffectiveAgent()
    for s in default_scenarios().values():
        rho = r.state_after(s.cues)
        assert abs(np.trace(rho).real - 1) < 1e-12
        assert np.allclose(rho, rho.conj().T, atol=1e-12)
        assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_decisions_are_probabilities():
    agents = [RotationAffectiveAgent(), ClassicalRotationAgent(), ClassicalAffectiveAgent()]
    for s in default_scenarios().values():
        for a in agents:
            p = a.appraise(s)["p_approach"]
            assert -1e-12 <= p <= 1 + 1e-12


def test_rotation_agent_is_classically_reproducible():
    """The key honesty fact: the qubit adds nothing to a rotation-only agent."""
    q, c = RotationAffectiveAgent(), ClassicalRotationAgent()
    worst = 0.0
    for s in default_scenarios().values():
        worst = max(worst, abs(q.appraise(s)["p_approach"] - c.appraise(s)["p_approach"]))
    assert worst < 1e-12, f"rotation agent should equal its classical twin; worst={worst:.2e}"


def test_scalar_conflates_state_separates():
    c1, r = ClassicalAffectiveAgent(), RotationAffectiveAgent()
    scen = default_scenarios()
    Es = [c1.appraise(scen[n])["E"] for n in BAL]
    assert max(abs(e) for e in Es) < 1e-9, "balanced scenarios should share E == 0"
    for i in range(len(BAL)):
        for j in range(i + 1, len(BAL)):
            d = bloch_distance(r.state_after(scen[BAL[i]].cues), r.state_after(scen[BAL[j]].cues))
            assert d > 0.1, f"state should separate {BAL[i]} vs {BAL[j]}"


def test_measurement_agent_satisfies_qq():
    m = MeasurementAffectiveAgent()
    assert abs(m.qq()) < 1e-9, f"measurement agent must satisfy QQ (q~0), got {m.qq()}"


def test_measurement_agent_has_order_effect():
    m = MeasurementAffectiveAgent()
    assert m.order_effect() > 1e-3, "incompatible appraisals should give an order effect"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} affect-agent tests passed.")
