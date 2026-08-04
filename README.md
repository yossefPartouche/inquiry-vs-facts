# Does Self-Questioning Transfer Across Domains, and Is It Causally Used?

Testing whether a self-questioning procedure, elicited via demonstration
on GSM8K, transfers to raise accuracy on MATH Number Theory and whether
it is causally used by the model or merely narrated alongside an
independently-derived answer.

## Project summary

Six conditions are compared on a frozen pool of 110 MATH Number Theory
(Levels 1-2) problems, on two models (Gemma 4 E4B, Qwen3-1.7B):

- **C** — zero-shot
- **A** — self-questioning (elicited via out-of-domain GSM8K demonstrations)
- **B1-B4** — a graduated ladder of injected in-domain facts, from a single bare fact (B1) to a full worked procedure with the answer
  withheld (B4, the ceiling)

The headline question is not "does A beat B?" but *where on the ladder does A fall*. On Gemma, condition A is statistically indistinguishable
from B1 and B4, and significantly beats C, B2, and B3 recovering 86.4% of the total achievable gain.

Three causal ablations then test whether A's self-questioning chain is actually used: corrupting one value, corrupting the whole chain, and swapping in an irrelevant chain entirely. All three show large, statistically significant accuracy drops, with a dose-response pattern (swap < corrupt-one-step < corrupt-everything) that rules out a generic "any disruption hurts" explanation.

Full results and discussion: see `paper/` (final report) and `analysis/headline_report.txt` (raw numbers, source of truth for every statistic in the paper).

## Repository structure

- `src/` — core pipeline: schema, grader, generation backend, prompt construction, statistics
- `prompts/` — condition templates (C, A, B1-B4) and the fact library
- `data/` — problem sets, frozen filtered pool, per-problem B3/B4 content, ablation content
- `results/` — all generated model outputs (pilot, headline, three
  ablations), graded
- `scripts/` — reusable utilities (grading, verification, backfilling) and  `scripts/archive/` for one-off diagnostic scripts kept for the audit trail
- `analysis/` — `headline_report.txt`, the canonical, append-only record of every statistic reported in the paper
- `docs/` — design notes, including the Week 1 grader-validation process (`grader_validation_notes.md`)
- `tests/` — pytest suite for schema, grader, and stats

## Reproducing the headline result

```bash
PYTHONPATH=. python -m src.runner          # generate all six conditions
PYTHONPATH=. python -m src.grading_pipeline results/gen_number_theory_headline.jsonl
PYTHONPATH=. python -c "
from src.stats import load_rows, report
rows = load_rows('results/gen_number_theory_headline.jsonl')
for model in ['gemma4-e4b', 'qwen3-1.7b']:
    report(rows, model=model)
"
```

## Reproducing the ablations

```bash
PYTHONPATH=. python -m scripts.run_ablation_corrupt_last
PYTHONPATH=. python -m scripts.run_ablation_corrupt_all
PYTHONPATH=. python -m scripts.run_ablation_swap
```

Each writes to its own `results/gen_number_theory_ablation*.jsonl`, graded via the same `src.grading_pipeline` module.

## Tests

```bash
python -m pytest tests/ -v
```
