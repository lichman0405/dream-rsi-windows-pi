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
