"""Read-only replay, reference validation, and numerical attribution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_learning import AuditStatus, audit_run
from agent_learning.config import LearnerConfig
from agent_learning.learners import ReinforceLearner
from agent_learning.policy import SoftmaxPolicy
from agent_learning.storage import CosmosStore, InMemoryStore, LocalFileStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    ConsumedInput,
    Episode,
    Reward,
    RewardSource,
    TrainingRun,
    TrainingStatus,
)


def _trained(
    store: LocalFileStore | InMemoryStore, *, entropy: float = 0.02,
    limit: float = 3.0, nonuniform: bool = True, empty: bool = False,
) -> TrainingRun:
    policy = SoftmaxPolicy.from_actions(
        [Action(id="cached"), Action(id="live")], agent_id="agent", task_id="routing",
        max_logit_abs=limit,
        initial_logits={"cached": 0.1, "live": -0.1} if nonuniform else None,
    )
    before = policy.snapshot()
    store.store_policy(before)
    if not empty:
        for index, (action, value) in enumerate([("cached", 0.8), ("live", -0.4)]):
            ep = Episode(
                id=f"ep-{index}", agent_id="agent", task_id="routing",
                action_id=action, action_logprob=math.log(0.5),
                policy_id=before.id, policy_version=before.version,
                intent_summary="Route a synthetic request", expected_outcome="Find a handler",
                execution_status="completed", result_summary="Synthetic response",
                created_at=f"2026-09-16T00:00:0{index}+00:00",
            )
            store.store_episode(ep)
            store.store_reward(Reward(
                id=f"reward-{index}", episode_id=ep.id, agent_id="agent",
                source=RewardSource.AGGREGATE, value=value,
                created_at="2026-09-16T00:01:00+00:00",
            ))
    return LearningRunner(
        store=store, policy=policy, metrics=[],
        learner_config=LearnerConfig(
            learning_rate=0.4, baseline_decay=0.8, entropy_bonus=entropy,
            importance_clip=2.0, max_logit_abs=99.0, min_train_episodes=0,
        ),
    ).run_offline_batch("agent", task_id="routing", score_missing=False)


@pytest.fixture
def trained(tmp_path: Path) -> tuple[LocalFileStore, TrainingRun]:
    store = LocalFileStore(tmp_path / "store")
    return store, _trained(store)


def _fingerprint(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def test_fresh_process_replay_after_rescore_is_read_only(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path,
) -> None:
    store, run = trained
    store.store_reward(Reward(
        id="later-rescore", episode_id="ep-0", agent_id="agent",
        source=RewardSource.AGGREGATE, value=-1.0, created_at="2099-01-01T00:00:00+00:00",
    ))
    root = tmp_path / "store"
    before = _fingerprint(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AGENT_LEARNING_LR"] = "invalid-current-config"
    env["AGENT_LEARNING_SCORE_TIER"] = "llm"
    env["AGENT_LEARNING_STORE_BACKEND"] = "cosmos"
    script = (
        "import json,sys; from agent_learning import audit_run,LocalFileStore; "
        "print(json.dumps(audit_run(LocalFileStore(sys.argv[1]),'agent',sys.argv[2]).to_dict(),allow_nan=False))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(root), run.id],
        env=env, text=True, capture_output=True, check=True,
    )
    report = json.loads(result.stdout)

    assert report["status"] == "verified"
    assert report["input_policy"]["id"] == run.policy_id
    assert report["output_policy"]["id"] == run.output_policy_id
    assert [c["reward_id"] for c in report["contributions"]] == ["reward-1", "reward-0"]
    assert report["summary"]["consumed_episodes"] == 2
    assert report["differences"] == []
    assert _fingerprint(root) == before


def test_memory_objects_are_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryStore()
    run = _trained(store)
    original = copy.deepcopy(store.__dict__)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Audit must not write, sample, or score.")

    for method in ("store_run", "store_policy", "store_episode", "store_reward", "store_metric_results"):
        monkeypatch.setattr(store, method, unexpected)
    monkeypatch.setattr(SoftmaxPolicy, "choose", unexpected)
    monkeypatch.setattr(LearningRunner, "run_offline_batch", unexpected)
    monkeypatch.setattr(LearningRunner, "score_and_record", unexpected)

    assert audit_run(store, "agent", run.id).status == AuditStatus.VERIFIED
    assert {key: store.__dict__[key] for key in original} == original


def test_episode_terms_match_hand_calculated_update(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    run = _trained(store, entropy=0.0, nonuniform=False)
    report = audit_run(store, "agent", run.id)
    by_episode = {c["episode_id"]: c for c in report.contributions}

    assert report.status == AuditStatus.VERIFIED
    assert by_episode["ep-0"]["reward_deltas"] == pytest.approx({"cached": 0.08, "live": -0.08})
    assert by_episode["ep-1"]["reward_deltas"] == pytest.approx({"cached": 0.04, "live": -0.04})
    assert report.summary["proposed_deltas"] == pytest.approx({"cached": 0.12, "live": -0.12})
    assert report.summary["entropy_deltas"] == pytest.approx({"cached": 0.0, "live": 0.0})
    assert report.output_policy is not None
    assert report.output_policy["baseline"] == pytest.approx(0.04)


def test_clipping_and_entropy_reconcile_separately(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    run = _trained(store, limit=0.12)
    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.VERIFIED
    assert any(abs(value) > 0 for value in report.summary["entropy_deltas"].values())
    assert any(abs(value) > 0 for value in report.summary["clipping_adjustments"].values())
    for aid, applied in report.summary["recorded_applied_deltas"].items():
        reward_terms = sum(item["reward_deltas"][aid] for item in report.contributions)
        assert reward_terms + report.summary["entropy_deltas"][aid] == pytest.approx(
            report.summary["proposed_deltas"][aid],
        )
        assert reward_terms + report.summary["entropy_deltas"][aid] + report.summary[
            "clipping_adjustments"
        ][aid] == pytest.approx(applied)


def test_supported_noop_is_verified(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    run = _trained(store, empty=True)
    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.VERIFIED
    assert report.summary["no_update"]
    assert report.contributions == []
    assert report.input_policy == report.output_policy


@pytest.mark.parametrize("status", [TrainingStatus.FAILED, TrainingStatus.RUNNING, TrainingStatus.CANCELLED])
def test_non_successful_runs_are_not_verified(
    trained: tuple[LocalFileStore, TrainingRun], status: TrainingStatus,
) -> None:
    store, run = trained
    run.status = status
    store.store_run(run)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.INSUFFICIENT_LINEAGE
    assert report.reason_code == "unsupported_run"


@pytest.mark.parametrize("change", ["legacy", "custom", "future-version", "full-authority", "contextual"])
def test_unsupported_records_fail_closed(trained: tuple[LocalFileStore, TrainingRun], change: str) -> None:
    store, run = trained
    if change == "legacy":
        run.metadata.clear()
    elif change == "custom":
        run.algorithm = "CustomLearner"
    elif change == "future-version":
        run.metadata["lineage_version"] = 2
    else:
        snapshot = store.get_policy(run.policy_id, "agent")
        assert snapshot is not None
        if change == "full-authority":
            snapshot.metadata["decision_authority"] = "full"
        else:
            snapshot.metadata["policy_kind"] = "contextual_softmax"
        store.store_policy(snapshot)
    store.store_run(run)

    assert audit_run(store, "agent", run.id).status == AuditStatus.INSUFFICIENT_LINEAGE


@pytest.mark.parametrize("missing", ["receipt", "output", "learner-config", "policy-config"])
def test_missing_lineage_is_not_guessed(trained: tuple[LocalFileStore, TrainingRun], missing: str) -> None:
    store, run = trained
    if missing == "receipt":
        run.consumed_inputs = None
    elif missing == "output":
        run.output_policy_id = None
    elif missing == "learner-config":
        del run.hyperparameters["learner"]["baseline_decay"]
    else:
        del run.hyperparameters["policy"]
    store.store_run(run)

    assert audit_run(store, "agent", run.id).status == AuditStatus.INSUFFICIENT_LINEAGE


@pytest.mark.parametrize("missing", ["run", "input", "output", "episode", "reward"])
def test_missing_exact_references_are_reported(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, missing: str,
) -> None:
    store, run = trained
    if missing == "run":
        report = audit_run(store, "agent", "absent-run")
    else:
        assert run.consumed_inputs
        if missing == "input":
            run.policy_id = "absent-snapshot"
        elif missing == "output":
            run.output_policy_id = "absent-snapshot"
        elif missing == "episode":
            # Only this known test-created file is removed.
            (tmp_path / "store" / "episodes" / "agent" / "ep-1.json").unlink()
        else:
            run.consumed_inputs[0] = ConsumedInput("ep-1", "absent-reward")
        store.store_run(run)
        report = audit_run(store, "agent", run.id)
    assert report.status == AuditStatus.MISSING_RECORD
    assert "absent" in report.message or "ep-1" in report.message


@pytest.mark.parametrize(
    "change",
    ["receipt-order", "receipt-count", "wrong-task", "unknown-action", "reward-source", "reward-value",
     "invalid-logprob", "changed-output", "changed-summary", "invalid-config", "changed-input"],
)
def test_inconsistent_records_produce_mismatch(
    trained: tuple[LocalFileStore, TrainingRun], change: str,
) -> None:
    store, run = trained
    assert run.consumed_inputs
    if change == "receipt-order":
        run.consumed_inputs.reverse()
    elif change == "receipt-count":
        run.metrics["episodes_used"] = 0
    elif change in {"wrong-task", "unknown-action", "invalid-logprob"}:
        episode = store.get_episode("ep-0", "agent")
        assert episode is not None
        if change == "wrong-task":
            episode.task_id = "different"
        elif change == "unknown-action":
            episode.action_id = "outside"
        else:
            episode.action_logprob = float("nan")
        store.store_episode(episode)
    elif change in {"reward-source", "reward-value"}:
        [reward] = store.get_rewards_for_episode("ep-0", "agent")
        if change == "reward-source":
            reward.source = RewardSource.METRIC
        else:
            reward.value = -0.9
        store.store_reward(reward)
    elif change in {"changed-output", "changed-input"}:
        identifier = run.output_policy_id if change == "changed-output" else run.policy_id
        assert identifier is not None
        snapshot = store.get_policy(identifier, "agent")
        assert snapshot is not None
        snapshot.logits["cached"] += 0.1
        store.store_policy(snapshot)
    elif change == "changed-summary":
        run.metrics["logit_deltas"]["live"] = 99.0
    else:
        run.hyperparameters["learner"]["learning_rate"] = float("inf")
    store.store_run(run)
    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code != "reconciled"
    json.dumps(report.to_dict(), allow_nan=False)


def test_missing_propensity_is_reported_as_unknown(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    # The uniform behavior probability equals the current probability, so
    # replacing its logprob with None also leaves the historical update unchanged.
    run = _trained(store, nonuniform=False)
    episode = store.get_episode("ep-0", "agent")
    assert episode is not None
    episode.action_logprob = None
    episode.policy_id = "older-behavior-snapshot"
    store.store_episode(episode)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.VERIFIED
    item = next(c for c in report.contributions if c["episode_id"] == "ep-0")
    assert item["behavior_probability"] is None
    assert item["importance_fallback"]
    assert item["importance_weight"] == 1
    assert any("Missing behavior probability" in warning for warning in report.warnings)


def test_cloud_backend_rejected_before_any_io(monkeypatch: pytest.MonkeyPatch) -> None:
    store = CosmosStore()

    def unexpected(*args: Any) -> None:
        pytest.fail("Cloud I/O is out of scope.")

    monkeypatch.setattr(store, "get_run", unexpected)
    report = audit_run(store, "agent", "run")

    assert report.status == AuditStatus.INSUFFICIENT_LINEAGE
    assert report.reason_code == "unsupported_store"


def test_invalid_record_shape_is_not_verified(trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path) -> None:
    store, run = trained
    path = tmp_path / "store" / "runs" / "agent" / f"{run.id}.json"
    payload = json.loads(path.read_text())
    payload["consumed_inputs"] = False
    path.write_text(json.dumps(payload))

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"


def test_corrupt_local_json_reports_unreadable_record(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "runs" / "agent" / f"{run.id}.json"
    path.write_text("{not-json")

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISSING_RECORD
    assert "unreadable" in report.message
    assert "failed to read" in caplog.text


def test_unexpected_storage_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryStore()

    def unavailable(*args: Any) -> None:
        raise OSError("storage failure")

    monkeypatch.setattr(store, "get_run", unavailable)
    with pytest.raises(OSError, match="storage failure"):
        audit_run(store, "agent", "run")


def test_repeated_receipt_entries_keep_order_and_multiplicity(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    run = _trained(store)
    before = store.get_policy(run.policy_id, "agent")
    assert before is not None and run.consumed_inputs
    receipt = run.consumed_inputs + [run.consumed_inputs[0]]
    episodes = [store.get_episode(item.episode_id, "agent") for item in receipt]
    assert all(episode is not None for episode in episodes)
    loaded_episodes = [episode for episode in episodes if episode is not None]
    rewards = [
        r for item in receipt for r in store.get_rewards_for_episode(item.episode_id, "agent")
        if r.id == item.reward_id
    ]
    policy = SoftmaxPolicy.from_snapshot(before, max_logit_abs=3.0)
    config = LearnerConfig(
        **run.hyperparameters["learner"], max_logit_abs=3.0, min_train_episodes=0,
    )
    result = ReinforceLearner(config).update(policy, loaded_episodes, rewards)
    after = policy.snapshot()
    store.store_policy(after)
    run.output_policy_id = after.id
    run.consumed_inputs = receipt
    run.episode_ids.append(receipt[0].episode_id)
    run.metrics.update({
        "episodes_used": result.episodes_used,
        "mean_reward": result.mean_reward,
        "baseline_before": result.baseline_before,
        "baseline_after": result.baseline_after,
        "logit_deltas": result.logit_deltas,
        "extra": result.extra,
    })
    store.store_run(run)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.VERIFIED
    assert [item["episode_id"] for item in report.contributions] == ["ep-1", "ep-0", "ep-1"]


def test_ambiguous_selected_rewards_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryStore()
    run = _trained(store)
    get_rewards = store.get_rewards_for_episode

    def duplicated(episode_id: str, agent_id: str) -> list[Reward]:
        records = get_rewards(episode_id, agent_id)
        return records + records

    monkeypatch.setattr(store, "get_rewards_for_episode", duplicated)
    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "ambiguous_reward"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.1, True])
def test_invalid_aggregate_values_fail_closed(value: float | bool) -> None:
    store = InMemoryStore()
    run = _trained(store)
    [reward] = store.get_rewards_for_episode("ep-0", "agent")
    reward.value = value
    store.store_reward(reward)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_reward"
    json.dumps(report.to_dict(), allow_nan=False)


def test_nonfinite_reconstruction_is_reported_not_serialized(trained: tuple[LocalFileStore, TrainingRun]) -> None:
    store, run = trained
    run.hyperparameters["learner"]["learning_rate"] = 1e308
    run.hyperparameters["learner"]["entropy_bonus"] = 1e308
    store.store_run(run)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "nonfinite_update"
    json.dumps(report.to_dict(), allow_nan=False)


def test_partial_summary_is_not_verified(trained: tuple[LocalFileStore, TrainingRun]) -> None:
    store, run = trained
    del run.metrics["baseline_before"]
    store.store_run(run)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert any(d["field"] == "metrics.baseline_before" for d in report.differences)


def test_snapshot_difference_overflow_remains_json_serializable(
    trained: tuple[LocalFileStore, TrainingRun],
) -> None:
    store, run = trained
    assert run.output_policy_id is not None
    before = store.get_policy(run.policy_id, "agent")
    after = store.get_policy(run.output_policy_id, "agent")
    assert before is not None and after is not None
    before.logits["cached"] = 1e308
    after.logits["cached"] = -1e308
    store.store_policy(before)
    store.store_policy(after)

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "nonfinite_difference"
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 1.9), ("version", "1"), ("version", True),
        ("updates_applied", True), ("episodes_seen", 2.9),
        ("baseline", "0.0"), ("baseline", False),
        ("logits", {"cached": "0.1", "live": "-0.1"}),
        ("version", float("inf")), ("episodes_seen", float("nan")),
    ],
)
def test_raw_snapshot_numbers_are_validated_before_coercion(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, side: str, field: str, value: Any,
) -> None:
    store, run = trained
    identifier = run.policy_id if side == "input" else run.output_policy_id
    path = tmp_path / "store" / "policies" / "agent" / f"{identifier}.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    before = _fingerprint(tmp_path / "store")

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert field in report.message
    json.dumps(report.to_dict(), allow_nan=False)
    assert _fingerprint(tmp_path / "store") == before


@pytest.mark.parametrize(
    "field",
    ["id", "agent_id", "task_id", "baseline", "version", "episodes_seen", "updates_applied",
     "actions", "logits", "metadata"],
)
def test_raw_snapshot_required_fields_are_not_defaulted(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, field: str,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "policies" / "agent" / f"{run.policy_id}.json"
    payload = json.loads(path.read_text())
    del payload[field]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert field in report.message


@pytest.mark.parametrize("mutation", ["missing", "string", "boolean", "nonfinite"])
def test_selected_reward_value_is_not_defaulted_or_coerced(tmp_path: Path, mutation: str) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    store.store_policy(policy.snapshot())
    episode = Episode(
        id="ep", action_id="a", intent_summary="route", expected_outcome="handler",
        execution_status="completed", result_summary="fixture",
    )
    store.store_episode(episode)
    value = 0.0 if mutation == "missing" else 1.0
    store.store_reward(Reward(id="r", episode_id=episode.id, source=RewardSource.AGGREGATE, value=value))
    run = LearningRunner(store=store, policy=policy, metrics=[]).run_offline_batch("default", score_missing=False)
    path = tmp_path / "rewards" / "default" / "ep.json"
    payload = json.loads(path.read_text())
    if mutation == "missing":
        del payload[0]["value"]
    else:
        payload[0]["value"] = {"string": "1.0", "boolean": True, "nonfinite": float("inf")}[mutation]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "default", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert "value" in report.message
    if mutation != "nonfinite":
        # Ordinary SDK loading retains its legacy defaults/conversions.
        assert store.get_rewards_for_episode("ep", "default")[0].value == value


def test_ordinary_snapshot_loading_remains_backward_compatible(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    directory = tmp_path / "policies" / "default"
    directory.mkdir(parents=True)
    (directory / "old.json").write_text(json.dumps({
        "id": "old", "version": 1.9, "updates_applied": True,
        "actions": [{"id": "a"}], "logits": {"a": "0.0"},
    }))

    loaded = store.get_policy("old", "default")

    assert loaded is not None
    assert loaded.version == loaded.updates_applied == 1
    assert loaded.baseline == 0.0
    assert loaded.logits == {"a": 0.0}


@pytest.mark.parametrize("field", ["agent_id", "task_id", "episode_ids"])
def test_raw_noop_run_cannot_fill_missing_defaults(tmp_path: Path, field: str) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions([Action(id="a")])
    store.store_policy(policy.snapshot())
    run = LearningRunner(store=store, policy=policy, metrics=[]).run_offline_batch("default", score_missing=False)
    path = tmp_path / "runs" / "default" / f"{run.id}.json"
    payload = json.loads(path.read_text())
    del payload[field]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "default", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"


def test_numeric_decode_overflow_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryStore()

    def overflowing(*args: Any) -> None:
        raise OverflowError("counter cannot be decoded")

    monkeypatch.setattr(store, "get_run", overflowing)

    report = audit_run(store, "agent", "run")

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    json.dumps(report.to_dict(), allow_nan=False)


def test_cancellation_attribution_reports_residual_without_failing_replay(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    store.store_policy(policy.snapshot())
    # Store order is newest first: preserve this consumption order for the
    # cancellation-sensitive floating-point calculation from the review.
    for index, value in enumerate([1.0, 2e-13, -1.0]):
        ep = Episode(
            id=f"ep-{index}", action_id="a", action_logprob=-10.0,
            intent_summary="route", expected_outcome="handler",
            execution_status="completed", result_summary="fixture",
            created_at=f"2026-09-16T00:00:0{3 - index}+00:00",
        )
        store.store_episode(ep)
        store.store_reward(Reward(
            id=f"r-{index}", episode_id=ep.id, source=RewardSource.AGGREGATE, value=value,
        ))
    run = LearningRunner(
        store=store, policy=policy, metrics=[],
        learner_config=LearnerConfig(
            learning_rate=1e6, baseline_decay=0.9, entropy_bonus=0.0,
            importance_clip=5.0, max_logit_abs=10.0, min_train_episodes=0,
        ),
    ).run_offline_batch("default", score_missing=False)

    report = audit_run(store, "default", run.id)

    assert report.status == AuditStatus.VERIFIED
    assert report.differences == []
    residuals = report.summary["attribution_rounding_residuals"]
    assert abs(residuals["a"]) > 1e-12
    assert any("rounding" in warning.lower() for warning in report.warnings)
    for aid, proposed in report.summary["proposed_deltas"].items():
        summed = math.fsum(item["reward_deltas"][aid] for item in report.contributions)
        assert summed == report.summary["attribution_reward_totals"][aid]
        assert summed + residuals[aid] + report.summary["entropy_deltas"][aid] == pytest.approx(
            proposed, rel=1e-9, abs=1e-12,
        )
    json.dumps(report.to_dict(), allow_nan=False)
    # A real persisted-state mismatch must still fail under the original tolerances.
    assert run.output_policy_id is not None
    after = store.get_policy(run.output_policy_id, "default")
    assert after is not None
    after.logits["a"] += 1e-6
    store.store_policy(after)
    mismatched = audit_run(store, "default", run.id)
    assert mismatched.status == AuditStatus.MISMATCH
    assert any(d["field"] == "output.logits.a" for d in mismatched.differences)


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [("version", "1e400"), ("updates_applied", "true"), ("episodes_seen", "2.9")],
)
def test_operator_json_classifies_invalid_raw_counters(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, field: str, raw_value: str,
) -> None:
    _, run = trained
    root = tmp_path / "store"
    path = root / "policies" / "agent" / f"{run.output_policy_id}.json"
    payload = json.loads(path.read_text())
    payload[field] = "raw-counter-placeholder"
    path.write_text(json.dumps(payload).replace('"raw-counter-placeholder"', raw_value))
    before = _fingerprint(root)
    operator = Path(__file__).resolve().parents[1] / "examples" / "audit_run.py"

    result = subprocess.run(
        [sys.executable, "-B", str(operator), "--store-dir", str(root),
         "--agent-id", "agent", "--run-id", run.id, "--format", "json"],
        text=True, capture_output=True, check=False,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "mismatch"
    assert report["reason_code"] == "invalid_record"
    assert field in report["message"]
    assert _fingerprint(root) == before


@pytest.mark.parametrize("field", ["agent_id", "task_id", "action_logprob"])
def test_raw_episode_fields_cannot_be_defaulted(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, field: str,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "episodes" / "agent" / "ep-0.json"
    payload = json.loads(path.read_text())
    del payload[field]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert field in report.message


@pytest.mark.parametrize("field", ["agent_id", "source", "created_at"])
def test_selected_reward_required_fields_cannot_be_defaulted(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, field: str,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "rewards" / "agent" / "ep-0.json"
    payload = json.loads(path.read_text())
    del payload[0][field]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert field in report.message


@pytest.mark.parametrize("field", ["description", "parameters"])
def test_raw_action_definitions_cannot_be_defaulted(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path, field: str,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "policies" / "agent" / f"{run.policy_id}.json"
    payload = json.loads(path.read_text())
    del payload["actions"][0][field]
    path.write_text(json.dumps(payload))

    report = audit_run(store, "agent", run.id)

    assert report.status == AuditStatus.MISMATCH
    assert report.reason_code == "invalid_record"
    assert field in report.message


def test_local_audit_validates_and_decodes_the_same_raw_read(
    trained: tuple[LocalFileStore, TrainingRun], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, run = trained
    read_json = store._read_json
    reads: list[Path] = []

    def read_once(path: Path) -> Any:
        assert path not in reads, "Audit must decode its validated read, not reopen the record."
        reads.append(path)
        return read_json(path)

    def legacy_decoder(*args: Any) -> None:
        pytest.fail("Local audit must not invoke permissive store decoders.")

    monkeypatch.setattr(store, "_read_json", read_once)
    for method in ("get_run", "get_policy", "get_episode", "get_rewards_for_episode"):
        monkeypatch.setattr(store, method, legacy_decoder)

    assert audit_run(store, "agent", run.id).status == AuditStatus.VERIFIED
    assert len(reads) == 7


def test_raw_validation_ignores_unselected_rescore_value(
    trained: tuple[LocalFileStore, TrainingRun], tmp_path: Path,
) -> None:
    store, run = trained
    path = tmp_path / "store" / "rewards" / "agent" / "ep-0.json"
    payload = json.loads(path.read_text())
    later = dict(payload[0], id="unselected", value="not-a-number", created_at="2099-01-01T00:00:00+00:00")
    payload.append(later)
    path.write_text(json.dumps(payload))

    assert audit_run(store, "agent", run.id).status == AuditStatus.VERIFIED
