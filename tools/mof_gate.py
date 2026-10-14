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
* no call was refused by the token ceiling, and no request failed -- counted
  over every registered arm's ledger, and not reduced by the single approval
  named below, which stays counted and is reported separately;
* every registered arm has a readable cost ledger recording no refusal, no
  unapproved failed call and no call settled without a usage report;
* no arm left a reserved call unsettled -- asserted only once that arm has
  completed, since a reservation in flight while a runner is alive is normal;
* every arm stayed inside its own ceiling and the phase inside its total, each
  comparing recorded spend plus the separate conservative reserve;
* the vault reported itself sealed and verified absent.

The arm set comes from ``PHASES[phase].methods``. The first version of this
check, written for the three-arm ``formal2`` phase, hard-coded ``A``/``B``/``C``
and would have failed a two-arm phase for the arm it deliberately does not
register -- reporting a design choice as a defect. Everything else is unchanged.

The ledger sweep reads every registered arm's ``cost-ledger.jsonl`` instead of
arm ``A``'s alone; the aggregate check reads the block record, so a failed
request in any other arm was invisible here. Arm ``A`` keeps the revision-level
checks it alone can have, because ``B`` runs no policy developer.

One failure has been approved by the human user, for one run of phase ``ab``.
It is supplied as named constants plus an explicit scope check -- there is no
registry and no exception file is read at run time. Matching is on a SHA-256 of
the identity fields the ledger row itself carries, so another run, phase, method,
seed, repeat, call id, sequence, error type or token report does not match, a
second identical row is not covered, and removing or altering the approved row
fails the gate instead of passing it. Its ``charged: 0`` is not a measurement:
the spend it actually incurred is unknown, so its reservation upper bound is
carried as a separate conservative reserve against both ceilings and is never
counted as recorded spend. The test pool is not opened here and no score is
computed.
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

from mof_ledger_audit import ApprovedFailure, audit_ledger  # noqa: E402

from dream_rsi.mof.protocol import PHASES  # noqa: E402
from dream_rsi.storage.sqlite_registry import Registry  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "formal2"

spec = PHASES[PHASE]
ARMS = sorted(spec.methods)
BLOCK = sys.argv[2] if len(sys.argv) > 2 else f"{PHASE}-s{spec.split_seeds[0]}-r{spec.repeats[0]}"
OUT = Path(".runtime/mof_experiment/experiment") / PHASE
RUNS = Path(".runtime/runs")

#: The one approved failure: its run, and the fingerprint of the identity
#: recorded in approved-exception.json. This is the only waiver in this gate;
#: every other row stays blocking, and no other phase, block or arm reaches it.
AB_APPROVED_RUN_ID = "mof-ab-s101-r1-B"
AB_APPROVED_FAILURE_SHA256 = \
    "330566cf47932d61160173329075d5ebe50ea250b0710432466545df56f17dc9"

problems: list[str] = []
notes: list[str] = []
waivers: list[str] = []


def check(ok: bool, message: str) -> None:
    (notes if ok else problems).append(message)


def run_id_for(block_id: str, method: str) -> str:
    """block_id is "<phase>-s<seed>-r<repeat>"; run id is "mof-<phase>-s<seed>-r<rep>-<arm>"."""
    return f"mof-{PHASE}-{block_id.split('-', 1)[1]}-{method}"


def approved_failure_for(method: str) -> ApprovedFailure | None:
    """The recorded approval, and only inside the phase, block and arm it names."""
    if PHASE != "ab" or BLOCK != "ab-s101-r1" or method != "B":
        return None
    return ApprovedFailure(sha256=AB_APPROVED_FAILURE_SHA256, run_id=AB_APPROVED_RUN_ID)


# ------------------------------------------------------------------ block records
index = OUT / f"{PHASE}-blocks.jsonl"
records = {}
phase_latest = {}
if index.is_file():
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        plan = row["plan"]
        # The index is append-only and a resumed run appends a further record
        # under the same run id, so the last record written for a run is the one
        # that counts towards the phase total. Every run the phase has recorded
        # is kept, not just the arms of this block.
        if plan.get("block_id") and plan.get("method"):
            phase_latest[run_id_for(plan["block_id"], plan["method"])] = row
        if plan.get("block_id") == BLOCK:
            records[plan["method"]] = row

check(set(records) == set(ARMS),
      f"{len(records)} of {len(ARMS)} registered arms present for {BLOCK}: {sorted(records)}")
committed = dict.fromkeys(ARMS, False)
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
arm_complete: dict[str, bool] = dict.fromkeys(ARMS, False)
registry = Registry(".runtime/registry.sqlite3")
try:
    run_ids = {m: run_id_for(BLOCK, m) for m in ARMS}
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

# ------------------------------------------------------- per-arm cost ledgers
# Read arm by arm rather than in aggregate and rather than for A alone. The
# summaries below come from the block record; this opens the files the runner
# actually wrote. ``reserve_by_method`` carries each arm's approved reservation
# forward to the ceiling comparisons, separately from anything recorded.
reserve_by_method: dict[str, int] = dict.fromkeys(ARMS, 0)
for method in ARMS:
    ledger_path = RUNS / run_ids[method] / "cost-ledger.jsonl"
    if not ledger_path.is_file():
        check(False, f"{method} has no cost ledger at {ledger_path}")
        continue
    approval = approved_failure_for(method)
    audit = audit_ledger(ledger_path, approved=approval)
    reserve_by_method[method] = audit.conservative_unknown_reserve

    if approval is not None:
        # The approved failure must still be there, unchanged: removing or
        # mutating the row fails the gate rather than passing it.
        check(audit.approved_failed == 1,
              f"{method} ledger carries exactly one approved failure in "
              f"{approval.run_id} ({audit.approved_failed})")

    # An approval never reduces the failure count: an approved failure is still
    # counted as failed, and is reported separately in the WAIVED lines.
    check(audit.failed == audit.approved_failed,
          f"{method} ledger failed calls {audit.failed} (approved {audit.approved_failed})")
    check(audit.refused == 0, f"{method} ledger refusals {audit.refused}")
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
    # spend it actually incurred is unknown rather than zero.
    if audit.zero_charge:
        notes.append(f"{method} ledger has {audit.zero_charge} call(s) charged nothing "
                     f"against a live reservation (billed spend unknown)")

    if audit.approved_failed:
        waivers.append(
            f"{method} {audit.approved_failed} approved failed call(s) of "
            f"{audit.failed}; conservative reserve "
            f"{audit.conservative_unknown_reserve:,} tokens, not measured and not billed")

# ------------------------------------------------------------------ arm budgets
for method, row in sorted(records.items()):
    ledger = row.get("ledger") or {}
    charged = int(ledger.get("charged_total") or 0)
    reserve = reserve_by_method.get(method, 0)
    check(charged + reserve <= spec.token_budget,
          f"{method} spent {charged:,} recorded + {reserve:,} conservative reserve "
          f"of its {spec.token_budget:,} ceiling")
    check(int(ledger.get("refusals") or 0) == 0,
          f"{method} had no call refused by the token ceiling")
    check(int(ledger.get("requests_without_reported_usage") or 0) == 0,
          f"{method} reported usage on every request")
    findings = [r for r in row.get("findings", [])
                if r.get("error_type") == "TOKEN_BUDGET_EXHAUSTED"]
    check(not findings, f"{method} has no budget-exhausted finding")

# ------------------------------------------------------------------ phase budget
# The phase total is not one block's sum: it is every run the phase index has
# recorded, latest record per run id, so a resumed run is not counted twice.
# The approved reserve is a property of one recorded failure, added here once
# from whichever ledger holds it -- which may be another block's.
phase_recorded = sum(int((r.get("ledger") or {}).get("charged_total") or 0)
                     for r in phase_latest.values())
phase_reserve = sum(reserve_by_method.values())
for run_id in sorted(phase_latest):
    if run_id in run_ids.values() or run_id != AB_APPROVED_RUN_ID:
        continue          # already audited just above, or not the approved run
    ledger_path = RUNS / run_id / "cost-ledger.jsonl"
    if ledger_path.is_file():
        approval = ApprovedFailure(sha256=AB_APPROVED_FAILURE_SHA256, run_id=run_id)
        phase_reserve += audit_ledger(ledger_path, approved=approval).conservative_unknown_reserve
check(phase_recorded + phase_reserve <= spec.total_token_cap,
      f"phase committed {phase_recorded:,} recorded + {phase_reserve:,} conservative "
      f"reserve of {spec.total_token_cap:,}")

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
for message in waivers:
    print(f"  WAIVED {message}")
for message in problems:
    print(f"  FAIL {message}")
print(f"phase tokens recorded: {phase_recorded:,}; separate conservative reserve: "
      f"{phase_reserve:,} (approved, not billed)")
sys.exit(1 if problems else 0)
