"""
QAC v1 -- Single-qubit quantum-inspired affective/judgment model.

Agent C deliverable (Quantum-Inspired Model). Reference implementation of the
SHARED INTERFACE all downstream models must satisfy:

    predict(stimulus, questions) -> dict over joint outcomes of the measured questions

Everything here is CLASSICALLY SIMULATED (NumPy). No quantum hardware, no speedup.
The claim is REPRESENTATIONAL/BEHAVIORAL-FIT: non-commuting measurements force the
Wang-Busemeyer QQ equality (q == 0) with ZERO free order-effect parameters.

Basis convention (affective overlay -- see spec section 6):
    |0> = affective / amygdala pathway
    |1> = reflective / OFC pathway
A binary question is a projective measurement along a Bloch axis; "Yes" = + axis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# ---- Pauli basis -----------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def bloch_to_rho(r: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Density matrix from Bloch vector r=(x,y,z), |r|<=1. |r|<1 => mixed (ambivalence-as-uncertainty).

    Raises ValueError if |r| > 1: such an r yields a non-PSD matrix, i.e. NOT a
    valid density matrix (min eigenvalue < 0). Callers that optimize over Bloch
    space must stay inside the unit ball (e.g. parameterize by angles for pure
    states) rather than rely on silent clipping.
    """
    r = np.asarray(r, dtype=float)
    norm = float(np.linalg.norm(r))
    if norm > 1.0 + tol:
        raise ValueError(
            f"Bloch vector norm {norm:.6f} > 1: not a valid density matrix (would be non-PSD)"
        )
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def axis_projector(n: np.ndarray) -> np.ndarray:
    """Rank-1 'Yes' projector onto the +n axis: P_yes = (I + n.sigma)/2."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return 0.5 * (I2 + n[0] * SX + n[1] * SY + n[2] * SZ)


@dataclass
class Question:
    """A binary (Yes/No) question = complete projective measurement {P_yes, P_no=I-P_yes}."""
    name: str
    axis: np.ndarray  # Bloch axis of the "Yes" outcome

    def projectors(self):
        Py = axis_projector(self.axis)
        return {"y": Py, "n": I2 - Py}


@dataclass
class QuantumJudgmentModel:
    """
    Physical parameters (single qubit, pure init): 3 after SO(3) gauge fixing.
      - init_bloch: initial state (2 DOF if pure, 3 if mixed -- see spec 3/5)
      - questions: dict name -> Question (each axis = 2 DOF)
    None of these is an 'order-effect' parameter; the order effect emerges from
    [P_A, P_B] != 0, yet q==0 is forced identically (see test_qq_equality).
    """
    init_bloch: np.ndarray
    questions: dict = field(default_factory=dict)

    def rho0(self) -> np.ndarray:
        return bloch_to_rho(self.init_bloch)

    def predict(self, stimulus=None, questions=None) -> dict:
        """
        SHARED INTERFACE.
        questions: list of question names giving MEASUREMENT ORDER, e.g. ["A","B"], ["B","A"], ["B"].
        Returns the FULL JOINT over measured outcomes as a dict:
            2 questions -> {"yy":p,"yn":p,"ny":p,"nn":p} keyed in the *given order*
            1 question  -> {"y":p,"n":p}
        `stimulus` is accepted for interface symmetry; here init_bloch already encodes it
        (a real encoder would map stimulus -> init_bloch; see spec 1a).
        """
        if questions is None:
            raise ValueError("questions (measurement order) required")
        rho = self.rho0()
        return self._sequential(rho, list(questions))

    def _sequential(self, rho: np.ndarray, order: list) -> dict:
        if not order:
            return {"": 1.0}
        q = self.questions[order[0]]
        projs = q.projectors()
        out = {}
        for label, P in projs.items():
            p = float(np.trace(P @ rho).real)
            p = max(p, 0.0)
            if p < 1e-12:
                # zero-probability branch: children get 0 mass, keep keys well-formed
                sub = self._sequential(rho, order[1:])
                for k, v in sub.items():
                    out[label + k] = 0.0
                continue
            rho_post = (P @ rho @ P) / p                      # collapse + renormalize
            sub = self._sequential(rho_post, order[1:])
            for k, v in sub.items():
                out[label + k] = p * v
        return out


# ---- Falsifiable metrics (mirror Agent A's formal defs; keep in sync) -------
def qq_residual(model: QuantumJudgmentModel, A: str, B: str) -> float:
    """
    Wang-Busemeyer QQ residual. Cells keyed (A-answer, B-answer) regardless of order.
    q = [p_AB(y,n)+p_AB(n,y)] - [p_BA(y,n)+p_BA(n,y)]
    Quantum prediction: q == 0 exactly (proven: P_Ay P_Bn P_Ay + P_An P_By P_An
    - P_Bn P_Ay P_Bn - P_By P_An P_By == 0). Dimension-independent, any projectors.
    """
    ab = model.predict(questions=[A, B])   # keys are (A,B) order
    ba = model.predict(questions=[B, A])   # keys are (B,A) order -> reindex to (A,B)
    s_ab = ab["yn"] + ab["ny"]
    s_ba = ba["ny"] + ba["yn"]  # ba["ny"] = (B=n,A=y) = (A=y,B=n); ba["yn"] = (A=n,B=y)
    return s_ab - s_ba


def interference_delta(model: QuantumJudgmentModel, A: str, B: str) -> float:
    """
    Total-probability violation (disjunction/interference signature).
    Delta = p(B=y | B alone) - [ p(A=y) p(B=y|A=y) + p(A=n) p(B=y|A=n) ].
    Classical law of total probability => 0. Quantum => generally != 0.
    """
    b_alone = model.predict(questions=[B])["y"]
    ab = model.predict(questions=[A, B])
    marg = ab["yy"] + ab["ny"]  # sum over A of (A, B=y)
    return b_alone - marg


def von_neumann_entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho).real
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def coherence(rho: np.ndarray) -> float:
    """l1 off-diagonal coherence in the {|0>,|1>} pathway basis = |rho01| (twice, l1 norm)."""
    return float(2 * abs(rho[0, 1]))


if __name__ == "__main__":
    # Demo: two non-commuting questions -> real order effect in marginals, yet q==0.
    m = QuantumJudgmentModel(
        init_bloch=np.array([0.3, 0.0, 0.6]),
        questions={
            "A": Question("A", axis=[0, 0, 1]),      # along z
            "B": Question("B", axis=[np.sin(1.1), 0, np.cos(1.1)]),  # tilted -> non-commuting
        },
    )
    print("AB joint:", {k: round(v, 4) for k, v in m.predict(questions=["A", "B"]).items()})
    print("BA joint:", {k: round(v, 4) for k, v in m.predict(questions=["B", "A"]).items()})
    print("QQ residual q =", round(qq_residual(m, "A", "B"), 12))
    print("interference Delta =", round(interference_delta(m, "A", "B"), 6))
    print("S(rho)=", round(von_neumann_entropy(m.rho0()), 4),
          " purity=", round(purity(m.rho0()), 4),
          " coherence=", round(coherence(m.rho0()), 4))
