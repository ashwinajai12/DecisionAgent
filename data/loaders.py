"""
QAC v1 -- Data loaders and canonical parsing schemas (Agent D deliverable).

Two DISTINCT observed-data shapes feed the shared model interface
    predict(stimulus, questions:list[str]) -> full-joint dict
      2 questions -> {"yy","yn","ny","nn"} keyed in the GIVEN measurement order
      1 question  -> {"y","n"}

  (1) ORDER-EFFECT data (PRIMARY: Wang/Solloway/Shiffrin/Busemeyer PNAS 2014;
      the 26-Pew approval x satisfaction set from the CogSci-2013 companion).
      Between-subjects split sample: each survey has TWO independent arms
      (order AB, order BA); the 2x2 joint is observed WITHIN an arm only.
      Feeds metric: qq_residual (q, forced ==0 by the quantum model).

  (2) DISJUNCTION data (SECONDARY: Tversky-Shafir 1992; 2020 JBDM replication,
      OSF gu58m; Broekaert-Busemeyer-Pothos 2020 two-stage gamble).
      Three conditions {win, loss, unknown}; a single binary action. NOT a
      question-order design -- feeds metric: interference_delta (total-prob Delta).

Nothing here is committed with restricted microdata. CSVs hold DERIVED aggregate
cell counts + a regeneration script provenance string. See spec in the handoff.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent

# ===========================================================================
# (1) ORDER-EFFECT SCHEMA
# ===========================================================================
# Canonical tidy CSV: qac/data/order_effect_cells.csv
# ONE ROW PER (survey_id, order, A_ans, B_ans) => 4 rows per arm, 8 per survey.
# Use EXPLICIT A_ans/B_ans columns (NOT positional yy/yn) so the file is
# order-agnostic and self-documenting; the loader does the positional mapping.
ORDER_EFFECT_COLUMNS = [
    "survey_id",   # str, stable key, e.g. "pew_approve_satisfy_2007_01"
    "stimulus_id", # str, the question-pair identity, e.g. "approve_x_satisfy"
    "qA_text",     # str, verbatim wording of question A
    "qB_text",     # str, verbatim wording of question B
    "order",       # str in {"AB","BA"}: which question was asked FIRST in this arm
    "A_ans",       # str in {"y","n"}: answer to question A
    "B_ans",       # str in {"y","n"}: answer to question B
    "count",       # int: respondents in this arm giving (A_ans,B_ans)
    "N_arm",       # int: total respondents in this arm (sum of its 4 cells)
    "source",      # str provenance, e.g. "PNAS2014_SI_Table_Sx" / "Moore2002_POQ_Table1"
    "date",        # str ISO or field date; "" if unknown
    "notes",       # str, free
]


@dataclass
class OrderEffectArm:
    """One split-sample arm: a 2x2 joint over (A_ans, B_ans) as raw counts."""
    survey_id: str
    stimulus_id: str
    qA_text: str
    qB_text: str
    order: str                 # "AB" or "BA"
    cells: dict                # {("y","y"):int, ("y","n"):int, ("n","y"):int, ("n","n"):int}
    N_arm: int
    source: str = ""
    date: str = ""

    def proportions(self) -> dict:
        n = self.N_arm or sum(self.cells.values())
        return {k: (v / n if n else 0.0) for k, v in self.cells.items()}

    def observed_joint(self) -> dict:
        """
        Observed distribution in the MODEL's positional keys, using THIS arm's
        asked order. Positional label = (first-asked answer, second-asked answer).
          order AB: key "yn" means A=y, B=n
          order BA: key "yn" means B=y, A=n   (i.e. A_ans=n, B_ans=y)
        This mirrors quantum_qubit.predict(questions=[first,second]). The A<->B
        transpose for BA is exactly what qq_residual()'s ba[...] reindexing undoes,
        so pass model.predict(questions=[first,second]) against this dict directly.
        """
        p = self.proportions()
        first, second = ("A", "B") if self.order == "AB" else ("B", "A")
        out = {}
        for (a, b), pr in p.items():
            fa = a if first == "A" else b
            sb = b if second == "B" else a
            out[fa + sb] = out.get(fa + sb, 0.0) + pr
        return out


def load_order_effect(path: str | Path = None) -> dict:
    """Return {survey_id: {"AB": OrderEffectArm, "BA": OrderEffectArm}}."""
    path = Path(path) if path else DATA_DIR / "order_effect_cells.csv"
    surveys: dict = {}
    meta: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid, order = row["survey_id"], row["order"]
            key = (sid, order)
            arm = meta.setdefault(key, {
                "survey_id": sid, "stimulus_id": row["stimulus_id"],
                "qA_text": row["qA_text"], "qB_text": row["qB_text"],
                "order": order, "cells": {}, "N_arm": int(row["N_arm"]),
                "source": row.get("source", ""), "date": row.get("date", ""),
            })
            arm["cells"][(row["A_ans"], row["B_ans"])] = int(row["count"])
    for (sid, order), a in meta.items():
        surveys.setdefault(sid, {})[order] = OrderEffectArm(**a)
    return surveys


# ===========================================================================
# (2) DISJUNCTION SCHEMA
# ===========================================================================
# Canonical tidy CSV: qac/data/disjunction_cells.csv
# ONE ROW PER (study, paradigm, condition, action).
DISJUNCTION_COLUMNS = [
    "study",      # str, e.g. "TverskyShafir1992_gamble" | "Repl2020_JBDM_choiceRisk"
    "paradigm",   # str in {"two_stage_gamble","paying_to_know"} (a.k.a. choice_under_risk / vacation)
    "condition",  # str in {"win","loss","unknown"}  (unknown == disjunctive/"don't know" arm)
    "action",     # str in {"gamble","not"} (or {"buy","not"} for paying_to_know)
    "count",      # int
    "N_cond",     # int total in this condition
    "source",     # str provenance
    "design",     # str in {"between","within"}
    "notes",
]


@dataclass
class DisjunctionCell:
    study: str
    paradigm: str
    condition: str      # win | loss | unknown
    p_gamble: float     # P(action=gamble | condition)
    N_cond: int
    design: str = "between"
    source: str = ""


def load_disjunction(path: str | Path = None) -> dict:
    """Return {(study,paradigm): {condition: DisjunctionCell}}."""
    path = Path(path) if path else DATA_DIR / "disjunction_cells.csv"
    agg: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["study"], row["paradigm"], row["condition"])
            a = agg.setdefault(key, {"gamble": 0, "not": 0,
                                     "N": int(row["N_cond"]),
                                     "design": row.get("design", "between"),
                                     "source": row.get("source", "")})
            a[row["action"] if row["action"] in ("gamble", "not") else "gamble"] += int(row["count"])
    out: dict = {}
    for (study, paradigm, cond), a in agg.items():
        n = a["N"] or (a["gamble"] + a["not"])
        out.setdefault((study, paradigm), {})[cond] = DisjunctionCell(
            study=study, paradigm=paradigm, condition=cond,
            p_gamble=(a["gamble"] / n if n else 0.0), N_cond=n,
            design=a["design"], source=a["source"])
    return out


def disjunction_delta(cells: dict, p_win: float = 0.5) -> float:
    """
    Total-probability (disjunction) violation for one study/paradigm.
      Delta = P(gamble | unknown)
              - [ p_win*P(gamble|win) + (1-p_win)*P(gamble|loss) ]
    NOTE for Agent E: the mixture weights here are the OBJECTIVE first-stage odds
    (0.5/0.5 in Tversky-Shafir), NOT model A-measurement marginals. The reference
    interference_delta() in quantum_qubit.py uses P(A=y|A-alone) as weights, which
    do NOT exist in this between-subjects design. Use THIS function for the data;
    have the quantum/classical models emit a comparable Delta via a matched mapping.
    """
    return cells["unknown"].p_gamble - (
        p_win * cells["win"].p_gamble + (1 - p_win) * cells["loss"].p_gamble)


# ---- provenance registry (verified 2026-07-02; see handoff for citations) ---
SOURCES = {
    "order_effect": {
        "primary_paper": "Wang, Solloway, Shiffrin, Busemeyer (2014) PNAS 111(26):9431-9436, 10.1073/pnas.1407756111",
        "n_surveys": "70 national surveys + 2 lab experiments = 72 studies (abstracts vary; SI table row count is ground truth)",
        "recommended_v1_set": "26 Pew approval x satisfaction surveys (Wang/Solloway/Busemeyer CogSci-2013 companion), N in [815,3006], M=1644 SD=422.24",
        "worked_examples": "Moore (2002) POQ 66(1):80-91 Gallux Clinton/Gore, White/Black, Rose/Jackson (PNAS Table 1)",
        "cell_availability": "MOST surveys report only MARGINALS publicly; complete 2x2 joints were reconstructed by authors and live in the PNAS SI (pnas.201407756SI.pdf) or must be rebuilt from Pew microdata.",
        "microdata": "Pew Research Center per-wave datasets (free account + T&C, redistribution-restricted); Gallup via Roper iPoll (subscription).",
    },
    "disjunction": {
        "original": "Tversky & Shafir (1992) Psychol Sci 3(5):305-309",
        "replication": "Revisiting Tversky & Shafir's (1992) Disjunction Effect, J. Behavioral Decision Making (2020), N=890 MTurk, between+within",
        "osf": "https://osf.io/gu58m/ ; API root https://api.osf.io/v2/nodes/gu58m/files/osfstorage/ ; folders: 'Choice under risk', 'Paying to know', 'Raw datasets'",
        "quantum_modeling_ref": "Broekaert, Busemeyer, Pothos (2020) two-stage gamble, heuristic/Markov/quantum-like comparison",
    },
}
