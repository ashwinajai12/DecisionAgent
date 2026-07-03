"""
Tests for the classical baselines. Runnable with plain python or pytest.
Key contract: the Markov baseline PRODUCES order effects but does NOT force q==0,
while the Independent baseline gets q==0 only by having no order effect.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.classical_markov import ClassicalMarkovModel, IndependentModel  # noqa: E402
from models.quantum_qubit import qq_residual                                # noqa: E402

RNG = np.random.default_rng(7)


def test_markov_joint_sums_to_one():
    for _ in range(300):
        m = ClassicalMarkovModel(
            a_first=RNG.uniform(0.05, 0.95), b_first=RNG.uniform(0.05, 0.95),
            w_AB_y=RNG.normal(), w_AB_n=RNG.normal(),
            w_BA_y=RNG.normal(), w_BA_n=RNG.normal(),
        )
        for order in (["A", "B"], ["B", "A"], ["A"], ["B"]):
            s = sum(m.predict(questions=order).values())
            assert abs(s - 1.0) < 1e-12, f"markov joint sums to {s}"


def test_markov_probs_in_unit_interval():
    for _ in range(300):
        m = ClassicalMarkovModel(a_first=RNG.uniform(0.05, 0.95),
                                 b_first=RNG.uniform(0.05, 0.95),
                                 w_AB_y=RNG.normal() * 2, w_AB_n=RNG.normal() * 2,
                                 w_BA_y=RNG.normal() * 2, w_BA_n=RNG.normal() * 2)
        for order in (["A", "B"], ["B", "A"]):
            for v in m.predict(questions=order).values():
                assert -1e-12 <= v <= 1 + 1e-12


def test_markov_produces_order_effect():
    # Net-positive assimilation -> P(B=y) shifts between B-first and B-second.
    # (Symmetric +/- shifts would move the correlation but cancel in the marginal;
    # a genuine marginal order effect needs a net shift.)
    m = ClassicalMarkovModel(a_first=0.5, b_first=0.5, w_AB_y=1.6, w_AB_n=0.6)
    ab = m.predict(questions=["A", "B"])
    b_second = ab["yy"] + ab["ny"]                 # P(B=y) with B asked second
    b_first = m.predict(questions=["B"])["y"]      # P(B=y) with B asked first
    assert abs(b_second - b_first) > 1e-3, "Markov should show an order effect"


def test_markov_q_generically_nonzero():
    m = ClassicalMarkovModel(a_first=0.55, b_first=0.45,
                             w_AB_y=1.2, w_AB_n=-0.8, w_BA_y=0.3, w_BA_n=-0.2)
    q = qq_residual(m, "A", "B")
    assert abs(q) > 1e-3, f"classical Markov q should be != 0, got {q}"


def test_independent_q_is_zero_but_no_order_effect():
    m = IndependentModel(a_yes=0.6, b_yes=0.4)
    q = qq_residual(m, "A", "B")
    assert abs(q) < 1e-12, "independent model q must be 0"
    ab = m.predict(questions=["A", "B"])
    b_second = ab["yy"] + ab["ny"]
    b_first = m.predict(questions=["B"])["y"]
    assert abs(b_second - b_first) < 1e-12, "independent model must have NO order effect"


def test_interface_parity_with_quantum():
    m = ClassicalMarkovModel(a_first=0.5, b_first=0.5, w_AB_y=0.4)
    assert set(m.predict(questions=["A", "B"])) == {"yy", "yn", "ny", "nn"}
    assert set(m.predict(questions=["A"])) == {"y", "n"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} baseline tests passed.")
