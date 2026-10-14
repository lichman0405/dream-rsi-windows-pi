"""Synthetic controller flow: replace every model, gate and scoring entrypoint."""
from types import SimpleNamespace

import pytest
from tools import mof_ab_continue as controller


@pytest.mark.parametrize("case", ["skip_completed_b", "arm_defect", "pair_gate_failure"])
def test_continuation_skips_b_and_stops_on_new_failure(tmp_path, monkeypatch, case):
    for name, filename in (("PHASE_JSON", "ab-phase.json"), ("PAUSE_MARKER", "ab.pause"),
                           ("STATUS_PATH", "controller-status.json")):
        monkeypatch.setattr(controller, name, tmp_path / filename)
    monkeypatch.setattr(controller, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(controller, "RUNS", tmp_path / "runs")
    controller.RUNS.mkdir()
    monkeypatch.setattr(controller, "STATE", {
        "phase": "ab", "protocol_digest": controller.PHASE_DIGEST,
        "initial_b_start": controller.INITIAL_B_START, "completed_arms": [],
        "isolation_cycles": [], "gate_results": {},
    })
    isolation = {"sealed": {"files": 1}, "verified_absent": {"ok": True},
                 "released": {"files": 1}, "verified_restored": {"ok": True}}
    controller.write_json(controller.PHASE_JSON, {"isolation": isolation})
    rows = {controller.APPROVED_RUN_ID: {
        "run_id": controller.APPROVED_RUN_ID, "status": "COMMITTED",
        "protocol_digest": controller.PHASE_DIGEST,
        "ledger": {"charged_total": 1000}, "implementation_revision": 2,
    }}
    calls = []
    monkeypatch.setattr(controller, "latest_records", lambda: rows)
    monkeypatch.setattr(controller, "preflight", lambda plans: ([], rows, {"isolation": isolation}))
    monkeypatch.setattr(controller, "build_protocol", lambda: SimpleNamespace(digest=controller.PHASE_DIGEST))
    monkeypatch.setattr(controller, "elapses_past_budget", lambda: False)
    monkeypatch.setattr(controller, "ledger_report", lambda rid, row: ([], {
        "exists": True, "conservative_unknown_reserve": 23312 if rid == controller.APPROVED_RUN_ID else 0,
    }))

    def arm_report(rid, method):
        if case == "arm_defect" and rid.endswith("s101-r1-A") and rid in rows:
            return ["synthetic new failure"], {"status": "COMMITTED"}
        return [], {"status": "COMMITTED", "worlds": 8, "sealed": 8, "revisions": 0, "sources": []}

    def run_arm(plan, protocol):
        rid = controller.run_id_for(plan.block_id, plan.method)
        calls.append(rid)
        rows[rid] = {"run_id": rid, "status": "COMMITTED", "protocol_digest": controller.PHASE_DIGEST,
                     "ledger": {"charged_total": 1000}, "implementation_revision": 2}
        result = {"isolation": isolation, "blocks": [rows[rid]]}
        controller.write_json(controller.PHASE_JSON, result)
        return result

    monkeypatch.setattr(controller, "arm_report", arm_report)
    monkeypatch.setattr(controller, "run_one_arm", run_arm)
    monkeypatch.setattr(controller, "run_gate", lambda block: (case != "pair_gate_failure", "synthetic gate"))
    monkeypatch.setattr(controller, "finalise", lambda plans: 0)
    code = controller._execute_locked()
    expected = ["mof-ab-s101-r1-A", "mof-ab-s211-r1-A", "mof-ab-s211-r1-B"]
    if case != "skip_completed_b":
        expected = expected[:1]
    assert calls == expected
    assert code == (0 if case == "skip_completed_b" else 2)
    assert controller.APPROVED_RUN_ID not in calls
