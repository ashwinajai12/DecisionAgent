# QAC — Quantum-inspired vs. classical models of question-order effects

A small, honest comparison study: does a single-qubit **projective-measurement**
model reproduce human **question-order effects** (the Wang–Busemeyer *QQ equality*)
that a fair **classical belief-adjustment (Markov)** baseline cannot get for free?

Everything is **classically simulated** (NumPy only). There is **no quantum speedup
or accuracy claim** — a 1-qubit density matrix is 3 real numbers. The claim is purely
*representational*: non-commuting measurements force the QQ equality (q = 0) with zero
free order-effect parameters, whereas the classical baseline needs parameters and
still does not force it.

## Layout
```
qac/
  models/
    quantum_qubit.py      single-qubit projective model + falsifiable metrics
    classical_markov.py   FAIR baseline: order effects, but q generically != 0
                          (+ IndependentModel control: q==0 only by having no order effect)
  agent/                  EMOTIONAL-AGENT layer (built on the validated core)
    affect.py             qubit as affect state (|0>=approach, |1>=avoid). Rotation agent
                          (+ classical 3-vector twin that matches it to 1e-16), a
                          MEASUREMENT agent (QQ signature), and scalar/2-vector baselines
    run_agent_demo.py     scalar conflation -> honesty check -> genuine QQ signature
  eval/
    metrics.py            empirical q from data, TVD, multinomial log-likelihood
    fit.py                dependency-free Nelder-Mead; fit_quantum / fit_markov / held-out
    run_comparison.py     end-to-end: load -> fit both -> head-to-head table
  data/
    loaders.py            order-effect + disjunction schemas, provenance registry
    *.csv                 SYNTHETIC templates (flagged DO_NOT_FIT) for smoke tests
  tests/                  24 tests, plain-python runnable (no pytest needed)
```

## Two layers, one mechanism
- **Validation layer** (`models/`, `eval/`): does the quantum encoding reproduce the
  Wang-Busemeyer QQ pattern in *real human data* that a fair classical model can't get
  for free? This is where the encoding earns credibility.
- **Agent layer** (`agent/`): the *same* qubit machinery as an affective agent that
  represents conflicting emotions (approach vs avoid) and resolves them under pressure.

Two honesty guardrails built into the agent layer (verified by tests):

1. **"Keeping both tendencies" is not quantum** — a 2-number classical vector
   (`TwoChannelClassicalAgent`) already separates *torn* from *indifferent*.
2. **A rotation-based qubit agent is not quantum either** — a single qubit's unitary
   evolution is classical 3D rotation (SU(2)→SO(3)). `ClassicalRotationAgent`, a plain
   3-vector, reproduces `RotationAffectiveAgent` to ~1e-16, order effect and all. So that
   order effect is rotation non-commutativity, **not** a quantum-probability result.

The genuine, data-tied signature is **measurement** non-commutativity:
`MeasurementAffectiveAgent` appraises via projective measurements (collapse), and its
order effects satisfy the QQ equality (q≈0) that classical Markov models violate — the
same constraint validated on real survey data. (A single qubit is still classically
*simulable*; the claim is about matching a data constraint classical-probability models
can't force, never hardware speedup.)

## Run
```bash
python qac/tests/run_all.py          # full test suite (24 tests)
python qac/agent/run_agent_demo.py   # emotional-agent demo (conflation + order effect)
python qac/eval/run_comparison.py    # end-to-end comparison (SMOKE on synthetic data)
python qac/models/quantum_qubit.py   # quantum demo (order effect + q==0)
python qac/models/classical_markov.py# baseline demo (order effect + q!=0)
```

## What the comparison shows
For each survey the harness reports empirical q, each fitted model's own q, in-sample
fit, and — the honest test — **held-out cross-order fit** (fit one order arm, predict
the other). Per-arm in-sample fit is near-saturated, so cross-order generalization is
where forcing q = 0 actually costs or helps.

The bundled data is synthetic (q ≈ −0.04), so the current run is a **pipeline smoke
test, not a result** — and on it the Markov model wins precisely because it can bend to
a nonzero q. The scientific question is decided only on **real data where q ≈ 0**.

## Status (honest)
- [x] Correct, verified quantum core (QQ equality forced to ≤1e-9; valid density matrices)
- [x] `bloch_to_rho` now rejects `|r| > 1` (was silently returning non-PSD matrices)
- [x] Fair classical Markov baseline (order effects without forced q=0) + independent control
- [x] Empirical q metric + model-vs-data fit (TVD, log-likelihood)
- [x] Fitting + held-out cross-order evaluation harness
- [x] Emotional-agent layer: rotation agent (+ classical twin proving it's classical),
      measurement agent with the QQ signature, scalar & 2-vector baselines
- [x] Self-falsifying honesty check: the rotation "order effect" is shown to be classical
- [x] 25 passing tests
- [ ] **Real data** — replace synthetic CSVs with PNAS-2014-SI / Pew-derived cells
      (order effects) and OSF `gu58m` cells (disjunction). See `data/loaders.SOURCES`.
- [ ] **Quantum disjunction model** — the interference (`interference_delta`) side has
      data-facing metrics only; no quantum model yet (Pothos–Busemeyer superposition
      "unknown" + unitary action map).
- [ ] Statistical significance across many surveys; stimulus→state encoder (or drop the
      unused `stimulus` argument from the interface).
```
```
