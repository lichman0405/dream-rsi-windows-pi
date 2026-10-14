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


# ------------------------------------------------------- interrupted-arm resume


def _partial_plans():
    from dream_rsi.mof.protocol import block_plans
    return [plan for plan in block_plans("ab")
            if plan.block_id == "ab-s101-r1" and plan.method == "A"]


def _partial_env(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "PAUSE_MARKER", tmp_path / "ab.pause")
    monkeypatch.setattr(controller, "PHASE_JSON", tmp_path / "ab-phase.json")
    monkeypatch.setattr(controller, "BLOCKS_INDEX", tmp_path / "ab-blocks.jsonl")
    monkeypatch.setattr(controller, "RESUME_APPROVALS", tmp_path / "ab.resume-approvals.json")
    monkeypatch.setattr(controller, "RUNS", tmp_path / "runs")
    controller.RUNS.mkdir()
    run_root = controller.RUNS / "mof-ab-s101-r1-A"
    run_root.mkdir()
    (run_root / "cost-ledger.jsonl").write_text("", encoding="utf-8")
    controller.write_json(controller.PHASE_JSON, {"isolation": {
        "sealed": {"files": 1}, "verified_absent": {"ok": True},
        "released": {"files": 1}, "verified_restored": {"ok": True}}})


def test_partial_arm_needs_an_approval_that_matches_its_state(tmp_path, monkeypatch):
    """Silence is not permission: no approval blocks, and a moved state unblocks nothing."""
    _partial_env(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, "partial_fingerprint", lambda run_id: "fp-1")
    plans = _partial_plans()

    problems, _, rows = controller.phase_preconditions(plans)
    assert rows[0]["state"] == "partial"
    assert any("partially run; not resumed" in problem for problem in problems)
    assert not rows[0].get("resume_approved")

    controller.write_json(controller.RESUME_APPROVALS, {"approvals": [
        {"run_id": "mof-ab-s101-r1-A", "partial_fingerprint": "fp-stale"}]})
    problems, _, rows = controller.phase_preconditions(plans)
    assert any("differs from the approved resume fingerprint" in problem for problem in problems)

    controller.write_json(controller.RESUME_APPROVALS, {"approvals": [
        {"run_id": "mof-ab-s101-r1-A", "partial_fingerprint": "fp-1",
         "authority": "test"}]})
    problems, _, rows = controller.phase_preconditions(plans)
    assert problems == []
    assert rows[0]["resume_approved"] is True


def test_partial_state_facts_summarize_without_judging(monkeypatch):
    def fake_column(query, run_id):
        if "from worlds" in query:
            return [("w0", 0, "SEALED_SCOREABLE"), ("w1", 1, "SEALED_SCOREABLE"),
                    ("w1-r1", 1, "SEALED_SCOREABLE"), ("w2", 2, "BUILDING")]
        if "from revisions" in query:
            return []

        return [("POLICY_SELECTED",)]

    monkeypatch.setattr(controller, "registry_column", fake_column)
    facts = controller.partial_state_facts("mof-ab-s101-r1-A")
    assert facts["readable"] is True
    assert facts["run_state"] == "POLICY_SELECTED"
    assert [world["state"] for world in facts["worlds"]] == [
        "SEALED_SCOREABLE", "SEALED_SCOREABLE", "SEALED_SCOREABLE", "BUILDING"]


# ----------------------------------------------------------- world-row accounting


def test_world_report_counts_rounds_and_discloses_a_quarantine(monkeypatch):
    rounds = [("w0", 0, "SEALED_SCOREABLE"), ("w3", 3, "QUARANTINED"),
              ("w3-r1", 3, "SEALED_SCOREABLE")]
    rounds += [(f"w{t}", t, "SEALED_SCOREABLE") for t in (1, 2, 4, 5, 6, 7)]

    def fake_column(query, run_id):
        return rounds if "from worlds" in query else []

    monkeypatch.setattr(controller, "registry_column", fake_column)
    problems, facts = controller.world_report("mof-ab-s101-r1-A", "A")
    assert problems == []
    assert facts["sealed"] == 8
    assert facts["rounds"] == 8
    assert facts["quarantined"] == 1
    assert facts["quarantine_detail"] == [{"world_id": "w3", "outer_t": 3}]


def test_world_report_fails_a_missing_or_unfinished_round(monkeypatch):
    rounds = [(f"w{t}", t, "SEALED_SCOREABLE") for t in range(7)]
    rounds.append(("w7", 7, "BUILDING"))

    def fake_column(query, run_id):
        return rounds if "from worlds" in query else []

    monkeypatch.setattr(controller, "registry_column", fake_column)
    problems, facts = controller.world_report("mof-ab-s101-r1-A", "A")
    assert facts["sealed"] == 7
    assert any("7 of 8 outer rounds sealed scoreable" in problem for problem in problems)
    assert any("BUILDING at round 7" in problem for problem in problems)


# --------------------------------------------------------------------- the lock


def test_a_dead_controllers_lock_is_archived_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "LOCK_PATH", tmp_path / "ab.controller.lock")
    monkeypatch.setattr(controller, "LOCK_ARCHIVE", tmp_path / "ab.controller.lock.stale")
    monkeypatch.setattr(controller, "lock_owner_alive", lambda data: False)
    monkeypatch.setattr(controller, "STATE", {"phase": "ab"})
    controller.LOCK_PATH.write_text(
        '{"token": "dead", "pid": 999999, "at": "2026-10-14T16:31:00+08:00"}\n', encoding="utf-8")

    token = controller.acquire_lock()
    assert token is not None
    archived = list(tmp_path.glob("ab.controller.lock.stale-*"))
    archived = [path for path in archived if path.suffix != ".json"]
    assert len(archived) == 1
    assert "dead" in archived[0].read_text(encoding="utf-8")
    assert controller.STATE["stale_locks"][0]["lock"]["pid"] == 999999

    controller.release_lock(token)
    assert not controller.LOCK_PATH.exists()


def test_a_live_controllers_lock_is_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "LOCK_PATH", tmp_path / "ab.controller.lock")
    monkeypatch.setattr(controller, "LOCK_ARCHIVE", tmp_path / "ab.controller.lock.stale")
    monkeypatch.setattr(controller, "lock_owner_alive", lambda data: True)
    controller.LOCK_PATH.write_text('{"token": "live", "pid": 1, "at": "now"}\n', encoding="utf-8")

    assert controller.acquire_lock() is None
    assert "live" in controller.LOCK_PATH.read_text(encoding="utf-8")
