# MOF screening experiment — how to run it, and what each artifact proves

The design is in `MOF_SCREENING_DESIGN.md`. This file is the operational half:
the commands, the order, and the file each claim comes from.

## Commands, in order

```bash
# 1. Implementation checks. Makes no model call and states no result.
dream-rsi mof-acceptance --phase pilot

# 2. One phase, block by block, with the held-out data off disk throughout.
dream-rsi mof-run --phase pilot

# 3. Open the sealed test pool once, for every frozen candidate.
dream-rsi mof-unseal --phase pilot

# 4. Figures, tables, summary.md.
dream-rsi mof-report --phase pilot
```

`mof-run` accepts `--only-split-seed`, `--only-repeat` and `--block-limit` for a
partial phase; every arm of a selected block still runs, and the report shows
how many blocks were complete. `--no-isolation` exists only for an offline
rehearsal and must not be used for a reported run.

## What each artifact is evidence of

| Artifact | What it proves |
| --- | --- |
| `.runtime/mof_experiment/acceptance.json` | the implementation checks ran, and their result |
| `<phase>/vault-manifest.jsonl` | the held-out files were sealed, verified absent, and restored byte-identical, with timestamps for the window |
| `runs/<run_id>/cost-ledger.jsonl` | one row per service request: role, model, duration, input/output/cache/reasoning tokens, whether usage was reported, cost when the service supplied one |
| `runs/<run_id>/block-summary.json` | the block's findings in settle order, the frozen candidate at each checkpoint, and the ledger totals |
| `<phase>/<phase>-blocks.jsonl` | the same records, one line per arm-run, as the phase index |
| `unsealed/unsealed.json` | every frozen candidate's sealed-pool S, Q, Q*, hit count and row-order diagnostic |
| `unsealed/references.json` | random / ridge / random-forest reference scores on the same pool |
| `unsealed/report.json` | the paired table, the bootstrap interval, the pre-registered verdict, and the figure paths |
| `unsealed/summary.md` | the reader-facing summary |
| `unsealed/figures/*.png`, `*.svg`, `*.csv` | the five figures and the data behind each series |

## The order that matters

The vault is sealed before the first block and released after the last. The
development labels are read into the runner's memory first, because afterwards
there is nothing on disk to read. `verify_absent` opens every protected path and
records the answer; if anything is still readable the phase aborts rather than
starting a search with the answers in reach.

Selection uses the development score only. The sealed pool is opened once, in
`mof-unseal`, after every candidate is frozen, and nothing measured there is fed
back into anything. Grading the tree afterwards to find a better node would
destroy the meaning of the test number, and does not happen.

## Reading the cost ledger

`usage_reported` is the honesty switch. When it is `false`, the service returned
no usage for that call and the row's `tokens.total` is `null`; the call was
charged its reservation, which is an upper bound built from the bytes actually
sent plus the reply ceiling. It is never recorded as zero.

`cost.known` is `false` on this provider: it reports no per-request money. The
report therefore gives tokens and says the currency figure is unknown rather
than estimating one from a price table.

## Resuming

A block interrupted before it sealed is resumed in place: run the same command
again. The token ceiling is seeded from the ledger the run already wrote and the
session ceiling from its persisted nodes and revisions, so a restart cannot hand
an arm a budget it has already spent. A resume under a different protocol digest
is refused.

If the process dies while the vault is closed, the keeper restores the files as
soon as the parent's pipe closes -- including after a hard kill, which is tested.
If a keeper is itself killed, the sealed files are lost from disk; they are
regenerable from the pinned Zenodo archive plus `scripts/prepare_crafted_mof_splits.py`,
and the vault manifest records exactly which paths were taken.

## The reduced A/B phase (`ab`)

`ab` is the frozen two-arm phase from §10 of the design: splits 101 and 211, one
repeat, A and B only. It runs the design's primary comparison and nothing else.
Its digest is `a07196279bf61523…`; it carries its own run ids (`mof-ab-…`), its
own output directory (`.runtime/mof_experiment/experiment/ab/`), its own pause
marker (`ab.pause`) and its own unseal/report pass, so none of its records can be
confused with `pilot`, `formal` or `formal2`.

```bash
# 1. Implementation checks. Makes no model call and states no result.
dream-rsi mof-acceptance --phase ab

# 2. Both paired blocks, under the vault.
dream-rsi mof-run --phase ab

# 3. Open the sealed pool once, for the frozen candidates of this phase only.
dream-rsi mof-unseal --phase ab

# 4. Figures, tables, summary.md.
dream-rsi mof-report --phase ab
```

**Before step 1, copy the existing acceptance record aside.** `mof-acceptance`
writes `.runtime/mof_experiment/acceptance.json`, which is a single shared file
and currently holds the `formal2` run at 2026-10-14T10:02:35. Running it for `ab`
overwrites that record; the digest it held is also recorded in `formal2`'s block
records and phase index, but the check list is not.

`ab` stops at a block boundary if `.runtime/mof_experiment/experiment/ab/ab.pause`
exists. Two blocks, so it is a short phase -- but the marker is the only way to
stop between them without killing a process that holds a sealed vault.

Its report is **descriptive by design** (§10.3): it gives the two paired
differences, their mean, an exploratory interval and the cost, and applies no
pre-registered verdict. A `descriptive_only` reading is not a failed test.

### Single approved failure (2026-10-14)

The user authorized continuing `ab` with exactly one preserved discovery
failure in `mof-ab-s101-r1-B`: request 60, call id
`229d0d5d413b4ec988dfcaf79b1d029c`, `PI_OUTPUT_LIMIT`. The request produced output
before it was aborted. Its ledger records zero tokens; its billed usage is
unknown. The original ledger and failed node remain unchanged.

`tools/mof_gate.py` matches the request's identity fingerprint
`330566cf47932d61160173329075d5ebe50ea250b0710432466545df56f17dc9` and the exact
phase, block and arm. It reports the failure and its approval separately;
different requests, altered identity fields, duplicates and other phases are
not covered. Removing the known failure also fails its gate.

For budget checks, its original 23,312-token reservation is carried separately
from recorded tokens. This is a conservative allowance, not measured or billed
usage. The gate adds it once to the arm and phase budget checks, using the
latest record per unique run across both blocks. Reports must disclose the
unknown usage and that recorded-token checkpoint positions were not rewritten.

All other completion checks remain required. Complete each arm under its vault,
verify restoration and its ledger before starting the next arm, then gate both
paired blocks before final freezing, test-pool scoring and reporting. Preserve
the completed initial B without rerunning it. The ignored continuation script
is `.runtime/mof_experiment/codex-cc-control/continue-ab.py`.

The implementation's replay horizon is `K2 = 6` in
`src/dream_rsi/mof/experiment.py`, including the completed initial B and the
historical phases. The design's tables stated 12 for formal/ab (8 for pilot);
those planned values were not implemented. Continuing uses the existing value
6 for every arm. This discrepancy must be stated in the final report; it is
not evidence that the original parameter table was satisfied.
