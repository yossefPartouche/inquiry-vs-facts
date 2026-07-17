"""
src/stats.py — paired analysis for the substitution-curve experiment.

UNIT OF ANALYSIS = THE PROBLEM.

Every statistic here is computed on a *balanced paired set*: exactly one
graded row per (problem_id, condition) for a single model. All bootstrap
replicates resample PROBLEMS (clusters), never rows, and a single index
draw is shared across all conditions -- which is what preserves the paired
design inside every replicate.

Never pool models. Analyse each model separately.

Condition labels are owned by src/schema.py. If you add a condition, add it
THERE, not here. The ordered ladder (C,B1,B2,B3,B4) is imported from schema,
so the headline plot can never grow a phantom rung. The ablation labels
(A_corrupt, A_swap) are analysed by the SAME functions but are never placed
on the ladder -- see report(..., anchor_conditions=ABLATION_CONDITIONS).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.schema import (
    CONDITIONS, MODELS,
    LADDER_CONDITIONS, INQUIRY_CONDITION, ABLATION_CONDITIONS,
    parse_ok_from_status,
)

Z95 = 1.959963985
LADDER = tuple(LADDER_CONDITIONS)   # ordered substitution curve; owned by schema.py


# --------------------------------------------------------------------------
# 1. Loading and filtering
# --------------------------------------------------------------------------

def load_rows(paths: str | Path | Iterable[str | Path]) -> list[dict]:
    """Load result rows from one or more .jsonl files into a flat list."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{lineno}: malformed JSON line") from e
    return rows


def select(rows: Sequence[dict], model: str | None = None,
           conditions: Sequence[str] | None = None) -> list[dict]:
    """Filter rows by model and/or condition. Pure filter -- no side effects."""
    out = list(rows)
    if model is not None:
        out = [r for r in out if r["model"] == model]
    if conditions is not None:
        keep = set(conditions)
        out = [r for r in out if r["condition"] in keep]
    return out


def models_present(rows: Sequence[dict]) -> list[str]:
    return sorted({r["model"] for r in rows})


def _parse_ok(row: dict) -> int:
    """parse_ok as 0/1.

    Prefer the derived parse_ok field. If it is absent (a half-migrated row),
    fall back to deriving it from box_extraction_status via the SAME schema
    function the grader uses -- not a private copy of the rule, and not the
    wrong field name. Absent both, treat as a parse failure (0).
    """
    v = row.get("parse_ok")
    if v is None:
        status = row.get("box_extraction_status")
        v = parse_ok_from_status(status) if status is not None else False
    return int(bool(v))


# --------------------------------------------------------------------------
# 2. The precondition check -- refuses to analyse an unbalanced set
# --------------------------------------------------------------------------

def complete_problem_ids(rows: Sequence[dict], model: str,
                         conditions: Sequence[str]) -> list[str]:
    """problem_ids that have exactly one GRADED row for every condition.

    Use this to *deliberately* restrict a half-failed run -- an explicit,
    logged decision, never a silent one.
    """
    conditions = tuple(conditions)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in select(rows, model=model, conditions=conditions):
        if r.get("correct") is None:
            continue  # ungraded rows do not count as present
        counts[(r["problem_id"], r["condition"])] += 1
    pids = {pid for pid, _ in counts}
    return sorted(
        pid for pid in pids
        if all(counts.get((pid, c), 0) == 1 for c in conditions)
    )


def check_balanced(rows: Sequence[dict], model: str,
                   conditions: Sequence[str],
                   anchor_conditions: Sequence[str] | None = None) -> list[str]:
    """Assert exactly one graded row per (problem_id, condition) for `model`,
    over the set of problems DEFINED BY the anchor conditions.

    anchor_conditions: which labels define membership in this analysis.
      - None (default): every listed condition is an anchor. For the ladder
        this is all of C..B4,A, so every pilot problem qualifies.
      - For the ABLATION pass: ("A_corrupt", "A_swap"). Condition A runs on
        ALL problems, so it must NOT be an anchor -- otherwise the balance
        check would demand the ablation labels on never-ablated problems and
        raise on a perfectly valid run. Membership = problems carrying an
        ablation label; A is then required on that subset but not outside it.

    RAISES ValueError on any missing cell, duplicate, ungraded row, or unknown
    label WITHIN the membership set. Returns the sorted member problem_ids.

    Raises rather than warns on purpose: silently analysing an unbalanced set
    breaks the paired design and quietly corrupts every number in the paper.
    """
    conditions = tuple(conditions)
    anchors = conditions if anchor_conditions is None else tuple(anchor_conditions)

    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"Unknown condition label(s) {unknown}. Add them to CONDITIONS in "
            "src/schema.py -- the schema is the single source of truth. Do not "
            "special-case labels in stats.py."
        )
    bad_anchor = [c for c in anchors if c not in conditions]
    if bad_anchor:
        raise ValueError(f"anchor_conditions {bad_anchor} are not in conditions "
                         f"{conditions}.")
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. MODELS = {sorted(MODELS)}")

    sub = select(rows, model=model, conditions=conditions)
    if not sub:
        raise ValueError(f"No rows for model={model!r}, conditions={conditions}.")

    # Membership: a problem is IN this analysis iff it carries >=1 row under ANY
    # anchor condition. Presence (not gradedness) so ungraded anchor rows are
    # still caught below rather than silently excluding the problem.
    member = {r["problem_id"] for r in sub if r["condition"] in anchors}
    if not member:
        raise ValueError(
            f"No problem carries any anchor condition {anchors} for "
            f"model={model!r}. Analysing the ablation? Are the ablation rows in?"
        )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    ungraded: list[tuple[str, str]] = []
    for r in sub:
        pid = r["problem_id"]
        if pid not in member:
            continue  # outside this analysis (e.g. A on a non-ablated problem)
        counts[(pid, r["condition"])] += 1
        if r.get("correct") is None:
            ungraded.append((pid, r["condition"]))

    pids = sorted(member)
    missing = [(pid, c) for pid in pids for c in conditions
               if counts.get((pid, c), 0) == 0]
    dupes = [(pid, c, counts[(pid, c)]) for pid in pids for c in conditions
             if counts.get((pid, c), 0) > 1]

    faults = []
    if missing:
        faults.append(f"{len(missing)} MISSING cell(s), e.g. {missing[:5]}")
    if dupes:
        faults.append(f"{len(dupes)} DUPLICATE cell(s), e.g. {dupes[:5]}")
    if ungraded:
        faults.append(f"{len(ungraded)} UNGRADED row(s) (correct=None), "
                      f"e.g. {ungraded[:5]}")

    if faults:
        hint = ""
        if (anchor_conditions is None
                and (set(conditions) & set(ABLATION_CONDITIONS))
                and INQUIRY_CONDITION in conditions):
            hint = (" HINT: you mixed ablation labels with 'A' but left "
                    "anchor_conditions=None. 'A' runs on every problem, so the "
                    "ablation subset looks unbalanced. Pass "
                    "anchor_conditions=ABLATION_CONDITIONS.")
        n_ok = len(complete_problem_ids(rows, model, conditions))
        raise ValueError(
            f"UNBALANCED PAIRED SET for model={model!r}: {len(pids)} member "
            f"problem(s) x {len(conditions)} condition(s). " + " | ".join(faults) +
            f"  Refusing to compute. ({n_ok} problem(s) are fully complete; to "
            "proceed on those, filter with complete_problem_ids() explicitly and "
            "say so in the paper.)" + hint
        )
    return pids


# --------------------------------------------------------------------------
# 3. The paired set
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedSet:
    """A balanced problem x condition matrix for ONE model.

    correct[cond][i]  == 1 iff problem_ids[i] was answered correctly
    parse_ok[cond][i] == 1 iff the grader could extract an answer
    """
    model: str
    problem_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    correct: dict[str, np.ndarray]
    parse_ok: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(self.problem_ids)

    def field(self, name: str) -> dict[str, np.ndarray]:
        return {"correct": self.correct, "parse_ok": self.parse_ok}[name]


def to_paired(rows: Sequence[dict], model: str,
              conditions: Sequence[str],
              anchor_conditions: Sequence[str] | None = None) -> PairedSet:
    """Validate (see check_balanced) and pivot rows into a PairedSet.

    For the ablation pass, call with anchor_conditions=("A_corrupt","A_swap")
    so condition A on non-ablated problems is excluded rather than demanded.
    """
    conditions = tuple(conditions)
    pids = check_balanced(rows, model, conditions, anchor_conditions)
    pos = {pid: i for i, pid in enumerate(pids)}
    n = len(pids)

    correct = {c: np.zeros(n, dtype=np.int8) for c in conditions}
    parse = {c: np.zeros(n, dtype=np.int8) for c in conditions}
    for r in select(rows, model=model, conditions=conditions):
        i = pos.get(r["problem_id"])
        if i is None:
            continue  # member set only (e.g. A on a non-ablated problem)
        correct[r["condition"]][i] = int(bool(r["correct"]))
        parse[r["condition"]][i] = _parse_ok(r)

    return PairedSet(model=model, problem_ids=tuple(pids),
                     conditions=conditions, correct=correct, parse_ok=parse)


# --------------------------------------------------------------------------
# 4. Bootstrap (cluster = problem; shared draws = paired)
# --------------------------------------------------------------------------

def bootstrap_replicates(ps: PairedSet, field: str = "correct",
                         n_boot: int = 10_000, seed: int = 0
                         ) -> dict[str, np.ndarray]:
    """Cluster-bootstrap over PROBLEMS.

    Returns {condition: array of shape (n_boot,)} of replicate rates.

    THE KEY LINE is the single `idx` draw: the SAME resampled problems are
    used for every condition in a given replicate. That is what makes
    reps[c1] - reps[c2] a *paired* difference, and it is why differences
    must be taken from these arrays rather than bootstrapped separately.
    """
    if ps.n == 0:
        raise ValueError("Empty PairedSet.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ps.n, size=(n_boot, ps.n))       # <-- shared draws
    data = ps.field(field)
    return {c: data[c][idx].mean(axis=1) for c in ps.conditions}


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. Sane at the boundary, where the percentile
    bootstrap degenerates to a zero-width interval (k==0 or k==n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# 5. Statistics
# --------------------------------------------------------------------------

def rate_table(ps: PairedSet, reps: dict[str, np.ndarray],
               field: str = "correct") -> dict[str, dict]:
    """Per-condition rate + bootstrap CI (+ Wilson CI as a boundary fallback).

    field="correct"  -> the accuracy table (primary metric; parse failures
                        are already recorded as correct=False upstream).
    field="parse_ok" -> the parse-rate table.
    """
    data = ps.field(field)
    out = {}
    for c in ps.conditions:
        k, n = int(data[c].sum()), ps.n
        lo, hi = percentile_ci(reps[c])
        degen = (k == 0 or k == n)
        out[c] = dict(n=n, k=k, rate=k / n, lo=lo, hi=hi,
                      wilson=wilson_ci(k, n), degenerate=degen)
    return out


def paired_diff(ps: PairedSet, reps: dict[str, np.ndarray],
                c1: str, c2: str) -> dict:
    """Paired difference c1 - c2 with a bootstrap CI on the DIFFERENCE.

    Point estimate = mean of the per-problem differences (identical to the
    difference of means, but computed the paired way for clarity).
    The CI comes from reps[c1] - reps[c2], i.e. the same resampled problems
    on both sides -- this is what retains the covariance term and gives the
    paired design its power.
    """
    d = ps.correct[c1].astype(np.int16) - ps.correct[c2].astype(np.int16)
    lo, hi = percentile_ci(reps[c1] - reps[c2])
    return dict(c1=c1, c2=c2, diff=float(d.mean()), lo=lo, hi=hi,
                excludes_zero=bool(lo > 0 or hi < 0),
                n_discordant=int((d != 0).sum()))


def exact_binom_two_sided(b: int, n: int) -> float:
    """Two-sided exact binomial p-value for b successes in n trials, p=0.5."""
    if n == 0:
        return 1.0
    k = min(b, n - b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def mcnemar(ps: PairedSet, c1: str, c2: str) -> dict:
    """The 2x2 paired table + exact McNemar p-value.

    only_c1 / only_c2 are the DISCORDANT pairs -- the only cells that carry
    information about the difference. Large counts in BOTH directions mean
    churn: the conditions solve different problems, and a small net delta is
    hiding a big disagreement.
    """
    a1 = ps.correct[c1].astype(bool)
    a2 = ps.correct[c2].astype(bool)
    b = int((a1 & ~a2).sum())
    c = int((~a1 & a2).sum())
    return dict(c1=c1, c2=c2,
                only_c1=b, only_c2=c,
                both=int((a1 & a2).sum()),
                neither=int((~a1 & ~a2).sum()),
                n_discordant=b + c,
                p_exact=exact_binom_two_sided(b, b + c))


def recovery_fraction(ps: PairedSet, reps: dict[str, np.ndarray],
                      target: str = "A", floor: str = "C",
                      ceiling: str = "B4") -> dict:
    """(acc[target] - acc[floor]) / (acc[ceiling] - acc[floor]), with a CI.

    "Transferred inquiry recovers X% of the gain available from full
    hand-holding." B4 is a CEILING NORMALIZER, never a fair competitor.

    Returns value=None if the ceiling is not reliably above the floor, in
    which case the ratio is meaningless and must not be reported.
    """
    num = float(ps.correct[target].mean() - ps.correct[floor].mean())
    den = float(ps.correct[ceiling].mean() - ps.correct[floor].mean())
    rn = reps[target] - reps[floor]
    rd = reps[ceiling] - reps[floor]
    usable = rd > 1e-9
    if den <= 0 or usable.mean() < 0.99:
        return dict(value=None, lo=None, hi=None,
                    note=f"{ceiling} is not reliably above {floor}; "
                         "recovery fraction is undefined -- do not report it.")
    lo, hi = percentile_ci(rn[usable] / rd[usable])
    return dict(value=num / den, lo=lo, hi=hi, note=None)


def locate_on_ladder(ps: PairedSet, reps: dict[str, np.ndarray],
                     target: str = "A",
                     ladder: Sequence[str] = LADDER) -> dict:
    """WHERE DOES A LAND ON THE C -> B1 -> B2 -> B3 -> B4 CURVE?

    Walks the EXPLICIT ordered `ladder` (schema's LADDER_CONDITIONS by
    default), filtered to what is present -- it never iterates the condition
    set, so ablation labels sitting in the PairedSet can never become rungs.

    For each rung, the paired difference (target - rung):
      CI entirely > 0  -> target is ABOVE that rung
      CI entirely < 0  -> target is BELOW that rung
      CI contains 0    -> INDISTINGUISHABLE at this sample size
                          (absence of evidence, NOT evidence of equivalence
                          -- quote the CI width, do not claim equality)
    """
    rungs = [r for r in ladder if r in ps.conditions]
    rows = []
    for r in rungs:
        d = paired_diff(ps, reps, target, r)
        rel = "above" if d["lo"] > 0 else "below" if d["hi"] < 0 else "tie"
        rows.append({**d, "rung": r, "relation": rel})
    return dict(
        target=target,
        rows=rows,
        above=[x["rung"] for x in rows if x["relation"] == "above"],
        tie=[x["rung"] for x in rows if x["relation"] == "tie"],
        below=[x["rung"] for x in rows if x["relation"] == "below"],
    )


# --------------------------------------------------------------------------
# 6. Sensitivity analysis: filter + the SAME functions (no parallel path)
# --------------------------------------------------------------------------

def drop_problems_with_parse_failure(rows: Sequence[dict], model: str,
                                     conditions: Sequence[str]
                                     ) -> tuple[list[dict], list[str]]:
    """Complete-case filter at the PROBLEM level. Returns (kept_rows, dropped_ids).

    NOTE, and this matters: you cannot drop unparseable *rows*. That would
    punch holes in the problem x condition matrix and check_balanced would
    (correctly) refuse the result. The paired analogue of "exclude
    unparseable" is: drop the whole PROBLEM if ANY condition failed to parse
    on it. The set stays balanced, every condition keeps being measured on
    the same problems, and you can feed the output straight back into
    to_paired().

    This filter is NOT neutral -- it preferentially removes problems that the
    weaker/wordier conditions choked on, which flatters them. That is exactly
    why the PRIMARY metric counts parse failures as incorrect, and this is
    only a robustness check.
    """
    sub = select(rows, model=model, conditions=conditions)
    bad = {r["problem_id"] for r in sub if not _parse_ok(r)}
    kept = [r for r in sub if r["problem_id"] not in bad]
    return kept, sorted(bad)


# --------------------------------------------------------------------------
# 7. Report
# --------------------------------------------------------------------------

def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def report(rows: Sequence[dict], model: str,
           conditions: Sequence[str] = LADDER + ("A",),
           target: str = "A", ladder: Sequence[str] = LADDER,
           comparisons: Sequence[tuple[str, str]] | None = None,
           anchor_conditions: Sequence[str] | None = None,
           n_boot: int = 10_000, seed: int = 0) -> PairedSet:
    """Print the full analysis for one model. Returns the PairedSet.

    Ladder pass (default): conditions = C..B4,A, ladder = C..B4.
    Ablation pass: conditions = ("A",)+ABLATION_CONDITIONS, ladder = (),
                   comparisons = [("A","A_corrupt"), ("A","A_swap")],
                   anchor_conditions = ABLATION_CONDITIONS.
    """
    ps = to_paired(rows, model, conditions, anchor_conditions)
    reps = bootstrap_replicates(ps, "correct", n_boot, seed)
    preps = bootstrap_replicates(ps, "parse_ok", n_boot, seed)

    print(f"\n=== {model} | n = {ps.n} problems | {n_boot} bootstrap replicates "
          f"(resampling PROBLEMS) ===")

    print("\n-- Accuracy (primary: parse failures count as INCORRECT) --")
    acc = rate_table(ps, reps, "correct")
    print(f"{'cond':>6} {'acc%':>6} {'95% CI':>16} {'k/n':>10}")
    for c in conditions:
        a = acc[c]
        flag = "  <-- DEGENERATE (bootstrap CI is zero-width; " \
               f"Wilson: [{_pct(a['wilson'][0])},{_pct(a['wilson'][1])}])" \
               if a["degenerate"] else ""
        print(f"{c:>6} {_pct(a['rate'])} "
              f"[{_pct(a['lo'])},{_pct(a['hi'])}] {a['k']:>4}/{a['n']:<4}{flag}")

    print("\n-- parse_ok rate --")
    pk = rate_table(ps, preps, "parse_ok")
    for c in conditions:
        p = pk[c]
        print(f"{c:>6} {_pct(p['rate'])} [{_pct(p['lo'])},{_pct(p['hi'])}]")

    if comparisons is None:
        comparisons = [(target, r) for r in ladder if r in ps.conditions]
    print("\n-- Paired differences (CI on the DIFFERENCE; * = excludes 0) --")
    for c1, c2 in comparisons:
        d = paired_diff(ps, reps, c1, c2)
        m = mcnemar(ps, c1, c2)
        star = "*" if d["excludes_zero"] else " "
        print(f"{c1:>4} - {c2:<4} {_pct(d['diff']):>7} pp "
              f"[{_pct(d['lo'])},{_pct(d['hi'])}] {star}   "
              f"McNemar: {c1}-only={m['only_c1']:<3} {c2}-only={m['only_c2']:<3} "
              f"both={m['both']:<3} neither={m['neither']:<3} "
              f"p={m['p_exact']:.4f}")

    if ladder and target in ps.conditions and set(ladder) <= set(ps.conditions):
        loc = locate_on_ladder(ps, reps, target, ladder)
        print(f"\n-- Ladder placement for {target} --")
        print(f"   ABOVE            : {loc['above'] or '-'}")
        print(f"   INDISTINGUISHABLE: {loc['tie'] or '-'}")
        print(f"   BELOW            : {loc['below'] or '-'}")
        rec = recovery_fraction(ps, reps, target, ladder[0], ladder[-1])
        if rec["value"] is None:
            print(f"   recovery: {rec['note']}")
        else:
            print(f"   recovery of ({ladder[-1]} - {ladder[0]}) gain: "
                  f"{_pct(rec['value'])}% [{_pct(rec['lo'])},{_pct(rec['hi'])}]")
    return ps


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python -m src.stats results/*.jsonl")
    all_rows = load_rows(paths)
    for m in models_present(all_rows):
        conds = tuple(LADDER) + (INQUIRY_CONDITION,)
        report(all_rows, m, conds)

        kept, dropped = drop_problems_with_parse_failure(all_rows, m, conds)
        print(f"\n### SENSITIVITY: dropping {len(dropped)} problem(s) with a "
              f"parse failure in ANY condition ###")
        if kept:
            report(kept, m, conds)

        # Ablation pass, if the labels are present for this model. Same
        # machinery, membership pinned to the ablated subset, no ladder.
        ab_conds = (INQUIRY_CONDITION,) + tuple(ABLATION_CONDITIONS)
        if any(r["condition"] in ABLATION_CONDITIONS
               for r in select(all_rows, model=m)):
            print(f"\n### ABLATION (novelty) for {m} ###")
            report(all_rows, m, ab_conds,
                   comparisons=[(INQUIRY_CONDITION, c) for c in ABLATION_CONDITIONS],
                   ladder=(), anchor_conditions=ABLATION_CONDITIONS)