"""Synthetic-data tests: every expected value is hand-computed."""
import numpy as np
import pytest

from src.stats import (
    to_paired, check_balanced, bootstrap_replicates, rate_table,
    paired_diff, mcnemar, locate_on_ladder, recovery_fraction,
    drop_problems_with_parse_failure, wilson_ci, exact_binom_two_sided,
)

M = "gemma4-e4b"
LADDER6 = ("C", "B1", "B2", "B3", "B4", "A")
AB = ("A", "A_corrupt", "A_swap")           # ablation set (A + doctored chains)


def row(pid, cond, correct, parse_ok=True, model=M):
    # box_extraction_status uses REAL schema.BOX_STATUSES values so these
    # fixtures match production rows. parse_ok is set explicitly, so the
    # stats fallback path isn't exercised here -- it's covered by real data.
    return dict(problem_id=pid, condition=cond, model=model,
                correct=correct, parse_ok=parse_ok,
                box_extraction_status="ok" if parse_ok else "none_found")


# ---------------------------------------------------------------- 1. GUARD

def test_precondition_raises_on_missing_condition():
    """p02 is missing B3 -> the whole analysis MUST refuse to run."""
    rows = []
    for pid in ["p00", "p01", "p02"]:
        for c in LADDER6:
            if pid == "p02" and c == "B3":
                continue                      # <-- the hole
            rows.append(row(pid, c, True))
    with pytest.raises(ValueError, match="UNBALANCED"):
        to_paired(rows, M, LADDER6)


def test_precondition_raises_on_duplicate():
    rows = [row(p, c, True) for p in ["p00", "p01"] for c in LADDER6]
    rows.append(row("p00", "A", False))       # duplicate cell
    with pytest.raises(ValueError, match="DUPLICATE"):
        to_paired(rows, M, LADDER6)


def test_precondition_raises_on_ungraded():
    rows = [row(p, c, True) for p in ["p00", "p01"] for c in LADDER6]
    rows[3]["correct"] = None                 # not yet graded
    with pytest.raises(ValueError, match="UNGRADED"):
        to_paired(rows, M, LADDER6)


def test_balanced_set_passes():
    rows = [row(p, c, True) for p in ["p00", "p01"] for c in LADDER6]
    assert check_balanced(rows, M, LADDER6) == ["p00", "p01"]


# ------------------------------------------------------------ 2. ACCURACY

def test_accuracy_is_exactly_what_i_put_in():
    """C right on 3 of 10 -> 0.30, exactly. No cleverness."""
    rows = []
    for i in range(10):
        rows.append(row(f"p{i:02d}", "C", i < 3))
        rows.append(row(f"p{i:02d}", "A", True))
    ps = to_paired(rows, M, ("C", "A"))
    reps = bootstrap_replicates(ps, n_boot=2000, seed=1)
    tbl = rate_table(ps, reps)
    assert tbl["C"]["rate"] == pytest.approx(0.30)
    assert tbl["C"]["k"] == 3 and tbl["C"]["n"] == 10
    assert tbl["C"]["lo"] < 0.30 < tbl["C"]["hi"]


# ---------------------------------------- 3. THE ONE THAT PROVES PAIRING

def test_overlapping_marginal_cis_but_significant_paired_difference():
    """40 problems. B3 right on 24 (60%). A right on 32 (80%), a strict
    superset of B3. Hand-computed:
        marginal 95% CIs OVERLAP  (~[0.45,0.75] vs ~[0.68,0.92])
        paired diff = +0.20, CI ~[0.08,0.32], EXCLUDES ZERO
        discordant: 8 in A's favour, 0 against -> McNemar p ~ 0.008
    If the paired machinery is broken, this test fails.
    """
    rows = []
    for i in range(40):
        rows.append(row(f"p{i:02d}", "B3", i < 24))
        rows.append(row(f"p{i:02d}", "A",  i < 32))
    ps = to_paired(rows, M, ("B3", "A"))
    reps = bootstrap_replicates(ps, n_boot=10_000, seed=7)
    tbl = rate_table(ps, reps)

    # (a) the marginal CIs overlap -- eyeballing them would say "no difference"
    assert tbl["A"]["lo"] < tbl["B3"]["hi"], "expected the marginal CIs to overlap"

    # (b) the paired difference is unambiguous
    d = paired_diff(ps, reps, "A", "B3")
    assert d["diff"] == pytest.approx(0.20)
    assert d["lo"] > 0, "paired CI must exclude zero -- pairing is broken"
    assert d["excludes_zero"]

    # (c) and the exact test agrees
    m = mcnemar(ps, "A", "B3")
    assert (m["only_c1"], m["only_c2"], m["both"], m["neither"]) == (8, 0, 24, 8)
    assert m["p_exact"] < 0.01


def test_unpaired_would_be_wider_than_paired():
    """Sanity: shuffling one condition's problem order (destroying the
    pairing while preserving both marginals) INFLATES the difference CI.
    This is the concrete cost of resampling rows instead of problems."""
    rows = []
    for i in range(40):
        rows.append(row(f"p{i:02d}", "B3", i < 24))
        rows.append(row(f"p{i:02d}", "A",  i < 32))
    ps = to_paired(rows, M, ("B3", "A"))
    reps = bootstrap_replicates(ps, n_boot=10_000, seed=7)
    paired = reps["A"] - reps["B3"]

    rng = np.random.default_rng(0)                    # break the pairing:
    unpaired = reps["A"] - rng.permutation(reps["B3"])  # independent draws
    assert unpaired.std() > paired.std()


# ------------------------------------------------------------- 4. EDGE CASES

def test_degenerate_all_correct_and_all_wrong():
    """The percentile bootstrap DEGENERATES at the boundary: a condition that
    is 100% (or 0%) correct yields a zero-width CI. That is the method
    behaving as defined, not a bug -- but it is NOT a defensible inference.
    Report the Wilson interval instead, and flag it."""
    rows = []
    for i in range(20):
        rows.append(row(f"p{i:02d}", "B4", True))    # all correct
        rows.append(row(f"p{i:02d}", "C", False))    # all wrong
    ps = to_paired(rows, M, ("C", "B4"))
    reps = bootstrap_replicates(ps, n_boot=2000, seed=3)
    tbl = rate_table(ps, reps)

    assert tbl["B4"]["lo"] == tbl["B4"]["hi"] == 1.0
    assert tbl["C"]["lo"] == tbl["C"]["hi"] == 0.0
    assert tbl["B4"]["degenerate"] and tbl["C"]["degenerate"]

    # Wilson stays sane and does not claim certainty
    assert tbl["B4"]["wilson"][0] < 1.0
    assert tbl["C"]["wilson"][1] > 0.0

    # and the difference degenerates too -- +100pp, zero width
    d = paired_diff(ps, reps, "B4", "C")
    assert (d["diff"], d["lo"], d["hi"]) == (1.0, 1.0, 1.0)


def test_no_discordant_pairs():
    """Identical conditions: zero discordant pairs -> diff CI is [0,0] and
    McNemar p = 1.0. Degenerate, but correct and not a crash."""
    rows = []
    for i in range(20):
        rows.append(row(f"p{i:02d}", "A", i < 10))
        rows.append(row(f"p{i:02d}", "B1", i < 10))
    ps = to_paired(rows, M, ("A", "B1"))
    reps = bootstrap_replicates(ps, n_boot=1000, seed=5)
    d = paired_diff(ps, reps, "A", "B1")
    assert (d["diff"], d["lo"], d["hi"]) == (0.0, 0.0, 0.0)
    assert not d["excludes_zero"]
    assert mcnemar(ps, "A", "B1")["p_exact"] == 1.0


def test_bootstrap_is_deterministic_given_seed():
    rows = [row(f"p{i:02d}", c, i % 3 == 0) for i in range(15) for c in ("C", "A")]
    ps = to_paired(rows, M, ("C", "A"))
    a = bootstrap_replicates(ps, n_boot=500, seed=42)["A"]
    b = bootstrap_replicates(ps, n_boot=500, seed=42)["A"]
    assert np.array_equal(a, b)


# ------------------------------------------------------- 5. MCNEMAR / LADDER

def test_mcnemar_counts_and_churn():
    """Same net delta, opposite stories -- this is the whole point of McNemar."""
    rows = []
    #  p00-p04: A right, B3 wrong (5)   p05-p07: B3 right, A wrong (3)
    #  p08-p11: both right (4)          p12-p14: both wrong (3)
    for i in range(5):
        rows += [row(f"p{i:02d}", "A", True), row(f"p{i:02d}", "B3", False)]
    for i in range(5, 8):
        rows += [row(f"p{i:02d}", "A", False), row(f"p{i:02d}", "B3", True)]
    for i in range(8, 12):
        rows += [row(f"p{i:02d}", "A", True), row(f"p{i:02d}", "B3", True)]
    for i in range(12, 15):
        rows += [row(f"p{i:02d}", "A", False), row(f"p{i:02d}", "B3", False)]
    ps = to_paired(rows, M, ("A", "B3"))
    m = mcnemar(ps, "A", "B3")
    assert (m["only_c1"], m["only_c2"], m["both"], m["neither"]) == (5, 3, 4, 3)
    assert m["n_discordant"] == 8
    # net is only +2/15, but 8 of 15 problems DISAGREE -> churn, not dominance


def test_ladder_placement():
    """Monotone ladder, A engineered to sit between B2 and B3."""
    rows = []
    hits = {"C": 4, "B1": 12, "B2": 24, "B3": 40, "B4": 56, "A": 30}
    for i in range(80):
        for c, k in hits.items():
            rows.append(row(f"p{i:02d}", c, i < k))
    ps = to_paired(rows, M, tuple(hits))
    reps = bootstrap_replicates(ps, n_boot=10_000, seed=11)
    loc = locate_on_ladder(ps, reps, "A")
    assert "C" in loc["above"] and "B1" in loc["above"] and "B2" in loc["above"]
    assert "B3" in loc["below"] and "B4" in loc["below"]

    rec = recovery_fraction(ps, reps, "A", "C", "B4")
    assert rec["value"] == pytest.approx((30 - 4) / (56 - 4), abs=1e-9)


# --------------------------------------------------------- 6. SENSITIVITY

def test_sensitivity_filter_keeps_the_set_balanced():
    """p03 has ONE unparseable row (under A). The complete-case filter must
    remove ALL SIX of p03's rows, leaving a balanced set the same functions
    can consume unchanged."""
    rows = []
    for i in range(10):
        for c in LADDER6:
            bad = (i == 3 and c == "A")
            rows.append(row(f"p{i:02d}", c, False if bad else True,
                            parse_ok=not bad))
    kept, dropped = drop_problems_with_parse_failure(rows, M, LADDER6)
    assert dropped == ["p03"]
    assert len(kept) == 9 * 6
    ps = to_paired(kept, M, LADDER6)          # must NOT raise
    assert ps.n == 9 and "p03" not in ps.problem_ids


# ----------------------------------------------------------------- 7. MATHS

def test_exact_binomial():
    assert exact_binom_two_sided(0, 0) == 1.0
    assert exact_binom_two_sided(5, 10) == pytest.approx(1.0)
    assert exact_binom_two_sided(8, 8) == pytest.approx(2 / 256)


def test_wilson_never_degenerate():
    lo, hi = wilson_ci(20, 20)
    assert 0.0 < lo < 1.0 and hi == pytest.approx(1.0)

# --------------------------------------------- 8. ABLATION SUBSET / ISOLATION

def test_ablation_subset_does_not_false_raise():
    """6 main conditions on all 10 problems; only p00..p03 are ablated.
    The ablation analysis must cover exactly those 4 and NOT demand
    A_corrupt/A_swap on the never-ablated problems."""
    rows = []
    for i in range(10):
        for c in LADDER6:                       # C,B1,B2,B3,B4,A on everyone
            rows.append(row(f"p{i:02d}", c, i % 2 == 0))
    for i in range(4):                          # ablate only p00..p03
        rows.append(row(f"p{i:02d}", "A_corrupt", False))
        rows.append(row(f"p{i:02d}", "A_swap", True))

    # ladder analysis: all 10, ablation labels absent from the PairedSet
    ps_ladder = to_paired(rows, M, LADDER6)
    assert ps_ladder.n == 10
    assert "A_corrupt" not in ps_ladder.conditions

    # ablation analysis: exactly the 4 ablated problems, no false raise
    ps_ab = to_paired(rows, M, AB, anchor_conditions=("A_corrupt", "A_swap"))
    assert ps_ab.n == 4
    assert set(ps_ab.problem_ids) == {"p00", "p01", "p02", "p03"}


def test_ablation_half_failed_still_raises():
    """p02 has A_corrupt but is MISSING A_swap -> a genuine half-failed
    ablation run -> MUST still raise (this is the failure we DO want caught)."""
    rows = []
    for i in range(10):
        for c in LADDER6:
            rows.append(row(f"p{i:02d}", c, True))
    for i in range(4):
        rows.append(row(f"p{i:02d}", "A_corrupt", True))
        if i != 2:                              # p02's swap run "failed"
            rows.append(row(f"p{i:02d}", "A_swap", True))
    with pytest.raises(ValueError, match="UNBALANCED"):
        to_paired(rows, M, AB, anchor_conditions=("A_corrupt", "A_swap"))


def test_mixing_ablation_with_A_and_no_anchor_gives_hint():
    """The easy mistake: ablation labels + 'A' but anchor_conditions left None.
    'A' runs on every problem, so the subset looks unbalanced -> must raise
    AND print the hint telling you to pass anchor_conditions."""
    rows = []
    for i in range(10):
        for c in LADDER6:
            rows.append(row(f"p{i:02d}", c, True))
    for i in range(4):
        rows.append(row(f"p{i:02d}", "A_corrupt", True))
        rows.append(row(f"p{i:02d}", "A_swap", True))
    with pytest.raises(ValueError, match="anchor_conditions=ABLATION_CONDITIONS"):
        to_paired(rows, M, AB)                  # anchor_conditions=None -> trap


def test_locate_on_ladder_ignores_ablation_labels():
    """Even if ablation labels are (wrongly) inside a PairedSet, the ladder
    placement walks the explicit LADDER and cannot grow phantom rungs."""
    hits = {"C": 4, "B1": 12, "B2": 24, "B3": 40, "B4": 56,
            "A": 30, "A_corrupt": 10, "A_swap": 50}
    rows = [row(f"p{i:02d}", c, i < k)
            for i in range(80) for c, k in hits.items()]
    ps = to_paired(rows, M, tuple(hits))        # includes ablation labels
    reps = bootstrap_replicates(ps, n_boot=3000, seed=2)
    loc = locate_on_ladder(ps, reps, "A")       # default ladder = LADDER
    seen = set(loc["above"]) | set(loc["tie"]) | set(loc["below"])
    assert seen <= {"C", "B1", "B2", "B3", "B4"}
    assert "A_corrupt" not in seen and "A_swap" not in seen


def test_ablation_uses_same_diff_machinery():
    """The ablation is just paired_diff/mcnemar on different labels. Corrupting
    the chain drops accuracy; the same difference-CI machinery detects it."""
    rows = []
    for i in range(30):
        rows.append(row(f"p{i:02d}", "A", True))            # A solves all 30
        rows.append(row(f"p{i:02d}", "A_corrupt", i < 12))  # corrupt: 12/30
        rows.append(row(f"p{i:02d}", "A_swap", i < 27))     # swap: 27/30
    ps = to_paired(rows, M, AB, anchor_conditions=("A_corrupt", "A_swap"))
    reps = bootstrap_replicates(ps, n_boot=10_000, seed=4)
    d = paired_diff(ps, reps, "A", "A_corrupt")
    assert d["diff"] == pytest.approx(18 / 30)
    assert d["lo"] > 0 and d["excludes_zero"]              # chain is load-bearing
    m = mcnemar(ps, "A", "A_corrupt")
    assert (m["only_c1"], m["only_c2"]) == (18, 0)