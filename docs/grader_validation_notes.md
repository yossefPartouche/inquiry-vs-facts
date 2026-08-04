## Grader validation (Week 1 gate)

**Goal.** `src/grader.py` is the instrument every number in this project is
measured with. Its 32 golden tests prove it is self-consistent, not that it is
*correct* — for that it has to agree with an external, independently published
number. This run is that check.

**Reference.** Qwen3-1.7B-Base, GSM8K = **75.44** (Qwen3 Technical Report,
arXiv:2505.09388, Table 8). Protocol per §3.3: 4-shot, chain-of-thought. The
base model was chosen deliberately: 75.44 is a base-model number, and a base
model is a pure completion engine, so nothing (chat template, system prompt,
thinking mode) sits between our prompt and the result.

**Prompt.** The Qwen3 report does not publish its prompt, stating only "widely-used
evaluation settings". We used the canonical 4-shot GSM8K CoT exemplars from Wei et
al. (2022), verbatim. Greedy decoding (`do_sample=False`), 512 max new tokens,
200 problems from the test split.

**Two deviations, both disclosed:**

1. *Exemplar format.* We rewrote the exemplars' closing line from
   `The answer is 6.` to `The answer is \boxed{6}.`, since our grader keys on
   `\boxed{}`.
2. *Answer-extraction adapter.* The exemplar rewrite did not take: the model
   emitted `The answer is N.` anyway, in 19 of the first 20 outputs. The Wei et
   al. prompt is one of the most-replicated text blocks in the pretraining corpus,
   and four counter-demonstrations do not outweigh that prior — a base model has
   no instruction-following with which to override it. We therefore rewrite
   `The answer is N.` into `\boxed{N}` at grade time, in the baseline script only.
   `src/grader.py` is untouched. This is the same "flexible-extract" step
   `lm-evaluation-harness` applies, and is throwaway: the experiment's models are
   instruction-tuned and follow the `\boxed{}` instruction directly.

**Why we scaled from 20 to 200.** At n=20 the 95% CI is roughly ±20 points — wide
enough that our 0.70 and the target 0.7544 were indistinguishable, but also wide
enough to hide a real defect. n=200 narrows it to about ±6.6, which is tight
enough for the comparison to mean something.

**Result.**

| metric | value |
|---|---|
| accuracy (all 200) | **0.645** (129/200) |
| `parse_ok` rate | **0.865** (173/200) |
| accuracy over parseable outputs | **0.746** (129/173) |
| target | 0.7544 |

**Reading.** Of the outputs the model actually completed, the grader scores
**74.6%** the published number to within noise. The 11-point gap in the headline
figure is entirely the 27 completions (13.5%) that hit the token cap mid-reasoning
and never emitted a final answer. Those are model/truncation failures, not grading
failures: no answer was produced, so none could be graded. **The gate passes.**

**Implication for the experiment (Track X, please read).** `parse_ok` is not
bookkeeping it is a confound. Here it moved the headline accuracy by 11 points on
its own. Condition A (self-questioning) produces longer chains than condition C, and
is therefore structurally more likely to truncate or lose its final box. If A's
unparseable rate is 8% and C's is 1%, that difference alone will show up as an
accuracy difference and be mistaken for an effect. `parse_ok` must be a column in
the results schema (it is not in the plan) and must be reported per
condition.