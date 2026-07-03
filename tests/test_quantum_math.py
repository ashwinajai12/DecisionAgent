"""
Tests for the quantum core. Runnable two ways:
    python qac/tests/test_quantum_math.py     (no pytest needed)
    pytest qac/tests/test_quantum_math.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.quantum_qubit import (                       # noqa: E402
    bloch_to_rho, axis_projector, Question, QuantumJudgmentModel,
    qq_residual, interference_delta, von_neumann_entropy, purity, coherence,
)

RNG = np.random.default_rng(1234)


def _random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def test_bloch_to_rho_valid_density_matrix():
    for _ in range(500):
        r = _random_unit(RNG) * RNG.uniform(0, 1)          # inside unit ball
        rho = bloch_to_rho(r)
        assert abs(np.trace(rho).real - 1.0) < 1e-12, "trace != 1"
        assert np.allclose(rho, rho.conj().T, atol=1e-12), "not Hermitian"
        assert np.linalg.eigvalsh(rho).min() > -1e-12, "not PSD"


def test_bloch_to_rho_rejects_out_of_ball():
    raised = False
    try:
        bloch_to_rho(np.array([1.0, 1.0, 1.0]))            # |r| = sqrt(3) > 1
    except ValueError:
        raised = True
    assert raised, "bloch_to_rho must reject |r| > 1 (would be non-PSD)"


def test_axis_projector_is_projector():
    for _ in range(200):
        P = axis_projector(_random_unit(RNG))
        assert np.allclose(P @ P, P, atol=1e-12), "P^2 != P"
        assert abs(np.trace(P).real - 1.0) < 1e-12, "rank-1 projector trace != 1"


def _random_model(rng):
    return QuantumJudgmentModel(
        init_bloch=_random_unit(rng) * rng.uniform(0, 1),
        questions={"A": Question("A", _random_unit(rng)),
                   "B": Question("B", _random_unit(rng))},
    )


def test_joint_sums_to_one():
    for _ in range(300):
        m = _random_model(RNG)
        for order in (["A", "B"], ["B", "A"], ["A"], ["B"]):
            s = sum(m.predict(questions=order).values())
            assert abs(s - 1.0) < 1e-10, f"joint sums to {s} for order {order}"


def test_qq_equality_is_forced():
    worst = 0.0
    for _ in range(2000):
        q = abs(qq_residual(_random_model(RNG), "A", "B"))
        worst = max(worst, q)
    assert worst < 1e-9, f"quantum q should be ~0; worst |q| = {worst:.2e}"


def test_interference_zero_iff_commuting():
    # Commuting: A and B share an axis -> classical law of total probability holds.
    ax = _random_unit(RNG)
    m_comm = QuantumJudgmentModel(
        init_bloch=_random_unit(RNG) * 0.7,
        questions={"A": Question("A", ax), "B": Question("B", ax)},
    )
    assert abs(interference_delta(m_comm, "A", "B")) < 1e-10, "commuting -> Delta == 0"

    # Non-commuting tilted axes -> generically nonzero interference.
    m_nc = QuantumJudgmentModel(
        init_bloch=np.array([0.3, 0.0, 0.6]),
        questions={"A": Question("A", [0, 0, 1]),
                   "B": Question("B", [np.sin(1.1), 0, np.cos(1.1)])},
    )
    assert abs(interference_delta(m_nc, "A", "B")) > 1e-6, "non-commuting -> Delta != 0"


def test_scalar_metrics_ranges():
    for _ in range(200):
        rho = bloch_to_rho(_random_unit(RNG) * RNG.uniform(0, 1))
        S = von_neumann_entropy(rho)
        assert -1e-9 <= S <= 1.0 + 1e-9, f"entropy out of [0,1]: {S}"
        assert 0.5 - 1e-9 <= purity(rho) <= 1.0 + 1e-9, "purity out of [0.5,1]"
        assert coherence(rho) >= -1e-12, "coherence must be >= 0"
    # pure state along z: S=0, purity=1
    rho_pure = bloch_to_rho(np.array([0, 0, 1.0]))
    assert von_neumann_entropy(rho_pure) < 1e-9
    assert abs(purity(rho_pure) - 1.0) < 1e-12
    # maximally mixed: S=1, purity=0.5
    rho_mix = bloch_to_rho(np.zeros(3))
    assert abs(von_neumann_entropy(rho_mix) - 1.0) < 1e-9
    assert abs(purity(rho_mix) - 0.5) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} quantum-math tests passed.")
