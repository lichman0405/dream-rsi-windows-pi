"""Gate between a phase's paired blocks.

Run after a block finishes. Exits 0 only if every condition the work order names
holds; anything else exits non-zero and the caller pauses the phase rather than
letting it continue on a broken foundation.

Usage::

    python tools/mof_gate.py <phase> [<block_id>]

``<block_id>`` defaults to the phase's first block.

Conditions, all read from the records rather than from the console:

* every arm the phase registers is ``COMMITTED``;
* each arm finished the full eight outer worlds, all sealed and scoreable;
* no policy-development revision was cut off by the context ceiling;
* no call was refused by the token ceiling, and no request failed;
* every registered arm has a readable cost ledger recording no refusal, no
  failed call and no call settled without a usage report;
* no arm left a reserved call unsettled -- asserted only once that arm has
  completed, since a reservation in flight while a runner is alive is normal;
* every arm stayed inside its own ceiling and the phase inside its total;
* the vault reported itself sealed and verified absent.

The arm set comes from ``PHASES[phase].methods``. The first version of this
check, written for the three-arm ``formal2`` phase, hard-coded ``A``/``B``/``C``
and would have failed a two-arm phase for the arm it deliberately does not
register -- reporting a design choice as a defect. Everything else is unchanged.

The ledger sweep reads every registered arm's ``cost-ledger.jsonl`` instead of
arm ``A``'s alone; the aggregate check above reads the block record, so a failed
request in any other arm was invisible here. Arm ``A`` keeps the revision-level
checks it alone can have, because ``B`` runs no policy developer.

Not settled here: a call can settle with ``charged: 0`` and an all-zero token
record while still having produced output, so the billed spend is unknown rather
than zero. This gate does not read such a row as a free call; neither does it
fail on the row itself, since ``PI_OUTPUT_LIMIT`` already fails as a failed call.
The test pool is not opened here and no score is computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The auditor sits next to this script. Running the gate from another working
# directory must not change which module is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream_rsi.mof.protocol import PHASES          # noqa: E402
from dream_rsi.storage.sqlite_registry import Registry  # noqa: E402
from mof_ledger_audit import audit_ledger          # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "formal2"

spec = PHASES[PHASE]
ARMS = sorted(spec.methods)
BLOCK = sys.argv[2] if len(sys.argv) > 2 else f"{PHASE}-s{spec.split_seeds[0]}-r{spec.repeats[0]}"
OUT = Path(".runtime/mof_experiment/experiment") / PHASE

problems: list[str] = []
notes: list[str] = []


def check(ok: bool, message: str) -> None:
    (notes if ok else problems).append(message)


# ------------------------------------------------------------------ block records
index = OUT / f"{PHASE}-blocks.jsonl"
records = {}
if index.is_file():
    for line in index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["plan"].get("block_id") == BLOCK:
                records[row["plan"]["method"]] = row

check(set(records) == set(ARMS),
      f"{len(records)} of {len(ARMS)} registered arms present for {BLOCK}: {sorted(records)}")
committed = {method: False for method in ARMS}
for method in ARMS:
    row = records.get(method)
    if row is None:
        continue
    check(row["status"] == "COMMITTED", f"{method} status {row['status']}")
    committed[method] = row["status"] == "COMMITTED"

# Every arm must have run its full length: the phase's worlds, sealed, scoreable.
# ``arm_complete`` keeps those facts as a value, so the reservation check further
# down can ask whether the arm has stopped. A reservation that outlives its arm
# is a hole in the ceiling accounting; one still open while the arm runs is just
# the newest call.
arm_complete: dict[str, bool] = {method: False for method in ARMS}
registry = Registry(".runtime/registry.sqlite3")
try:
    # block_id is "<phase>-s<seed>-r<repeat>"; the run id is "mof-<phase>-s<seed>-r<rep>-<arm>"
    prefix = BLOCK.split("-", 1)[1]
    run_ids = {m: f"mof-{PHASE}-{prefix}-{m}" for m in ARMS}
    for method, run_id in run_ids.items():
        worlds = registry.list_worlds(run_id)
        sealed = [w for w in worlds if w.state.value == "SEALED_SCOREABLE"]
        check(len(worlds) == spec.worlds,
              f"{method} has {len(worlds)} of {spec.worlds} outer worlds")
        check(len(sealed) == len(worlds),
              f"{method}: {len(sealed)} of {len(worlds)} worlds are scoreable")
        arm_complete[method] = (committed[method]
                                and len(worlds) == spec.worlds
                                and len(sealed) == len(worlds))
    # A-specific, and it stays that way: only A registers a policy developer.
    if "A" in run_ids:
        revisions = registry.list_revisions(run_ids["A"])
        context_cut = [r for r in revisions if r.status == "CONTEXT_LIMIT"]
        check(not context_cut,
              f"A had no revision cut by the context ceiling ({len(revisions)} revisions)")
        errors = [r for r in revisions if r.status in ("DEVELOPER_ERROR", "DEVELOPER_ERROR")]
        check(not errors, f"A had no developer error ({len(errors)})")
finally:
    registry.close()

# ------------------------------------------------------------------ ledgers
total_charged = 0
for method, row in sorted(records.items()):
    ledger = row.get("ledger") or {}
    charged = int(ledger.get("charged_total") or 0)
    total_charged += charged
    check(charged <= spec.token_budget,
          f"{method} spent {charged:,} of its {spec.token_budget:,} ceiling")
    check(int(ledger.get("refusals") or 0) == 0,
          f"{method} had no call refused by the token ceiling")
    check(int(ledger.get("requests_without_reported_usage") or 0) == 0,
          f"{method} reported usage on every request")
    failed = [r for r in row.get("findings", []) if r.get("error_type") == "TOKEN_BUDGET_EXHAUSTED"]
    check(not failed, f"{method} has no budget-exhausted finding")

check(total_charged <= spec.total_token_cap,
      f"phase committed {total_charged:,} of {spec.total_token_cap:,}")

# The persistent ledgers, read arm by arm rather than in aggregate and rather
# than for A alone. ``total_charged`` above is the block record's summary of the
# same calls; this opens the files the runner actually wrote.
for method in ARMS:
    ledger_path = Path(".runtime/runs") / run_ids[method] / "cost-ledger.jsonl"
    if not ledger_path.is_file():
        check(False, f"{method} has no cost ledger at {ledger_path}")
        continue
    audit = audit_ledger(ledger_path)
    check(audit.refused == 0, f"{method} ledger refusals {audit.refused}")
    check(audit.failed == 0, f"{method} ledger failed calls {audit.failed}")
    check(audit.unreported == 0,
          f"{method} ledger calls settled without a usage report {audit.unreported}")
    check(audit.unreadable == 0, f"{method} ledger unreadable rows {audit.unreadable}")

    if audit.outstanding and arm_complete[method]:
        check(False, f"{method} left {audit.outstanding} reservation(s) unsettled after "
                     f"completing: {', '.join(audit.outstanding_call_ids[:3])}")
    elif audit.outstanding:
        notes.append(f"{method} has {audit.outstanding} reservation(s) in flight "
                     f"(arm not complete; not a gate condition)")

    # Diagnostic only. A cut-off call leaves this shape in the ledger, and the
    # spend it actually incurred is unknown rather than zero. Not a gate
    # condition: a failed call already fails above.
    if audit.zero_charge:
        notes.append(f"{method} ledger has {audit.zero_charge} call(s) charged nothing "
                     f"against a live reservation (billed spend unknown)")

# ------------------------------------------------------------------ isolation
manifest = OUT / "vault-manifest.jsonl"
events = []
if manifest.is_file():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
kinds = [e["event"] for e in events]
check("vault.sealed" in kinds, "the vault was sealed for this phase")
absent = next((e for e in events if e["event"] == "vault.verified_absent"), None)
check(absent is not None and absent.get("ok") is True,
      "every protected path was verified absent")

# ------------------------------------------------------------------ verdict
print(f"gate for {BLOCK} [{PHASE}, arms {'/'.join(ARMS)}]: "
      f"{len(notes)} passed, {len(problems)} failed")
for message in notes:
    print(f"  ok   {message}")
for message in problems:
    print(f"  FAIL {message}")
print(f"phase tokens committed so far: {total_charged:,}")
sys.exit(1 if problems else 0)
