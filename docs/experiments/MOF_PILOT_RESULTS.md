# MOF screening pilot — results

**Phase:** `pilot`. **Protocol digest:** `9f5940ed36ffbedf3b4ce3a6f43a4d65ee07435df1a4727f15e6c9620430b0f8`.
**Model:** `cc-switch-deep-seek` / `deepseek-v4-pro`, extended reasoning `off` for every call.
**Ran:** 2026-10-12 18:26 → 2026-10-13 05:42 local. **Blocks:** 4 of 4 complete
(2 split seeds × 2 repeats). **Arm-runs:** 12 of 12 `COMMITTED`.
**Held-out data:** sealed for the whole phase — 4,089 files off disk, verified absent,
restored byte-identical. **Failed service requests:** 0.

## The answer to the question the experiment asks

> Does Dream-RSI produce a better screening program than a frozen fixed search policy,
> under the same model, data, tools and allowed budget?

**No evidence that it does.**

| quantity | value |
| --- | --- |
| **A − B**, sealed-pool `S` | **−0.0115** |
| exploratory 95% interval (cluster bootstrap over splits) | **[−0.0334, +0.0104]** |
| splits favouring A | 1 of 2 |
| pre-registered reading | **insufficient evidence** |

The design's bar for "worth further validation" is a mean of at least **+0.05 `S`**, the
same direction in at least 4 of 5 splits, and an interval that does not cross zero. The
observed mean is negative and small, the interval crosses zero, and the two splits
disagree in sign. This is the "interval crosses zero ⇒ report insufficient evidence"
branch, not a claim that Dream-RSI is harmful.

## Paired runs

`S` is the normalised node score; `Q` is the raw mean recommended uptake in mol/kg, so the
difference can be read as a physical quantity and not only as a ratio. `Q*` for the sealed
pool is 9.054 mol/kg.

| split | repeat | A `S` | A `Q` | B `S` | B `Q` | C `S` | C `Q` | A−B `S` | A−B `Q` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | 0.8408 | 7.613 | 0.8443 | 7.644 | 0.6948 | 6.291 | −0.0035 | −0.032 |
| 101 | 2 | 0.7827 | 7.086 | 0.8460 | 7.659 | 0.7735 | 7.004 | −0.0633 | −0.573 |
| 211 | 1 | 0.8560 | 7.750 | 0.8360 | 7.569 | 0.7455 | 6.749 | +0.0200 | +0.181 |
| 211 | 2 | 0.8605 | 7.791 | 0.8597 | 7.784 | 0.8251 | 7.471 | +0.0008 | +0.008 |
| **mean** | | **0.8350** | **7.560** | **0.8465** | **7.664** | **0.7597** | **6.879** | **−0.0115** | **−0.104** |

Read as material performance: A's top ten average **7.56 mol/kg** against B's **7.66** and
C's 6.88, out of a pool whose true best ten average 9.05. The gap between A and B is about a
tenth of a mole per kilogram — under the noise of which materials happen to land in a
ten-material shortlist, which is the same statement the interval makes.

The per-checkpoint curve and the full table are in `report.json` and
`fig3_paired_final_comparison.csv`.

## Development against the sealed pool

| arm | runs | mean dev `S` | mean test `S` | gap | failures |
| --- | --- | --- | --- | --- | --- |
| A Dream-RSI | 4 | 0.8223 | 0.8350 | −0.013 | 0 |
| B frozen initial policy | 4 | 0.8057 | 0.8465 | −0.041 | 0 |
| C frozen serial policy | 4 | 0.8157 | 0.7597 | **+0.056** | 0 |

C is the interesting one. It had the *best* development score in three of the four blocks
and the worst sealed-pool score in all four. C is the arm that spends the most tokens
(2.7–3.3 M per block against 0.8–1.0 M for A) by running 128 sequential greedy rewrites,
each inheriting the whole branch history, and that deep greedy path is exactly the shape
that overfits a 131–161 material development pool. A and B, which search more broadly,
have a small *negative* gap. The design predicted this failure mode and the sealed pool is
what revealed it.

## Screening references on the same sealed pool

| reference | `S`, split 101 | `S`, split 211 | `Q` (mol/kg), split 101 / 211 |
| --- | --- | --- | --- |
| random ordering (100 draws) | 0.4009 | 0.4009 | 3.63 / 3.63 |
| fixed ridge, `alpha=1.0` | 0.7595 | 0.7758 | 6.88 / 7.02 |
| fixed random forest | 0.9022 | 0.8515 | 8.17 / 7.71 |

The sealed pool is the same 197 materials for every split, so a reference's `S` moves
between splits only through its training set; random ordering does not train, so its two
rows are identical by construction.

A and B land at 0.78–0.86: clearly above the ridge baseline they started from, and around
the random forest or slightly below it on split 101. These are *material-screening*
references with no model calls at all — they say the task is neither trivial nor saturated,
and they are explicitly **not** Pi comparisons with a matched budget.

The accessible-pore-volume reference named in the design **could not be computed**: the
pinned CRAFTED 2.0.1 archive carries the RAC descriptor family only, and the design forbids
taking descriptors from outside the pinned version. Recorded as a deviation, not as a
missing run.

## Cost

| | |
| --- | --- |
| service requests | **1,091** (refused before sending: 0) |
| tokens charged | **17,414,177** (input 13,643,823 · output 3,352,178 · cache read 418,176 · reasoning 0) |
| requests with no reported usage | **0** |
| money | **unknown** — the provider reports no per-request cost, so no currency figure is claimed |

Against the design's ceiling of 5,000,000 tokens per method per block, the highest single
arm spent 3.32 M (method C on one block); A and B stayed between 0.54 M and 0.99 M. The
pilot's overall cap was 3 × 4 × 5 M = 60 M and it used 17.4 M.

## Figures

All five are in `docs/experiments/pilot/` as PNG, SVG and the CSV behind every series.

| figure | what it shows |
| --- | --- |
| `fig1_quality_budget_staircase` | development best-so-far `S` against cumulative tokens, one line per paired run, steps in the order candidates settled |
| `fig2_recursive_dynamics` | sealed-pool `S` and `Q` at each budget checkpoint, one panel per split seed |
| `fig3_paired_final_comparison` | final sealed-pool `S` per arm per paired run, A–B lines, the bootstrap interval and the 0.05 bar |
| `fig4_recommendation_explanation` | the CO₂ uptake of each arm's top ten against the fixed references |
| `fig5_internal_mechanism` | Eq(1) `V` and tree best-so-far `S` by outer round — mechanism only |

`fig5` is captioned as an internal policy signal. A rising `V` is not evidence that the
materials recommended to a user got better, and is not used as one here: A's `V` improved
over the incumbent in the selection rounds while its sealed-pool score did not.

## What went wrong, and what was fixed

Three defects were found by running the thing, and all three are fixed and covered:

1. **A candidate defeated its own deadline.** The evaluator enforced the 120 s ceiling with
   `communicate(timeout=...)` over pipes. A frozen candidate that spawns worker processes
   hands those pipes to its grandchildren, the read never reaches EOF, and the deadline
   stops bounding anything — one candidate ran for 56 minutes of CPU with the unseal pass
   waiting on it. The child's streams now go to files and an independent watchdog kills the
   process tree on its own clock.
2. **The trace dropped `usage` exactly where it mattered.** Records longer than the trace
   cap were replaced wholesale, so the longest and most expensive replies were the ones
   recorded as "usage not reported". Truncation now keeps `role`, `stopReason` and `usage`.
3. **Thinking deltas counted as deliverable.** The output ceiling killed the first live
   probe 32 KiB into its reasoning with no text emitted. Only text deltas count now.

None of these affected the pilot's integrity: all twelve arm-runs committed, no request
failed, and every request reported its usage.

## Limits

- **Two splits, two repeats each.** Four paired runs. The interval is a cluster bootstrap
  over *two* clusters; it is exploratory in the weakest sense and should not be read as a
  confidence statement about any population.
- Both splits come from one material library and share one sealed pool, so the two splits
  are not independent samples of anything.
- The design's full pilot is four blocks, which is what ran, but the analysis it prescribes
  (five split seeds) needs the formal phase's ten blocks.
- The provider reports no per-request money, so the cost is tokens and no currency figure.
- Extended reasoning was fixed `off` across all arms. That is identical across arms and so
  does not affect the comparison, but it does mean the absolute scores describe this model
  configuration, not the model at its default effort.
- The sealed test pool is now open for split seeds 101 and 211, and the formal phase would
  reuse it. Running the formal phase after this one means the experimenter knows two of its
  five splits' test outcomes in advance. Nothing here was used to tune anything, but it is a
  real limitation of reusing the pool and belongs in the record.

## Formal phase: measured, not run

The pilot's job included measuring whether the formal phase is affordable, and it is not,
on this machine, in this session:

- per pilot block (2 outer worlds): 2.5 h, of which method C alone is ~1.8 h
- the formal phase is 8 outer worlds × 12 rounds for A and B, and 96 serial rounds for C,
  which scales a block to roughly **5.7 h**
- ten blocks ⇒ roughly **55–60 h of wall clock**, and by the same scaling roughly
  **150–200 M tokens**, inside the 900 M ceiling but far outside any single working session

The design anticipated exactly this and requires the protocol to be re-frozen before the
formal phase rather than started and quietly changed. The honest options are to run it over
several days as a background job, or to revise the protocol first — most plausibly by
cutting method C's serial depth, which dominates the wall clock and produced the worst
sealed-pool results. **No re-frozen formal protocol has been written, and no formal block
has been started.**
