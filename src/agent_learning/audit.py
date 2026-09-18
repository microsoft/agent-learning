"""Read-only numerical replay of recorded native policy updates.

Verification is consistency with the referenced records, not proof that the
records are immutable or the underlying outcomes are correct.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Any, TypeGuard, TypeVar

from .config import LearnerConfig
from .learners import ReinforceLearner
from .policy import SoftmaxPolicy
from .storage import InMemoryStore, LearningStore, LocalFileStore
from .types import ConsumedInput, Episode, PolicySnapshot, Reward, RewardSource, TrainingRun, TrainingStatus

_REL_TOL = 1e-9
_ABS_TOL = 1e-12
_T = TypeVar("_T")


class AuditStatus(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    INSUFFICIENT_LINEAGE = "insufficient_lineage"
    MISSING_RECORD = "missing_record"


@dataclass
class AuditReport:
    """Serializable report; policy projections omit prompts and tool payloads."""

    agent_id: str
    run_id: str
    status: AuditStatus = AuditStatus.INSUFFICIENT_LINEAGE
    reason_code: str = "not_checked"
    message: str = ""
    task_id: str | None = None
    input_policy: dict[str, Any] | None = None
    output_policy: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    differences: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "tolerances": {"relative": _REL_TOL, "absolute": _ABS_TOL},
            "verification_scope": "Numerical consistency, not immutable provenance or model reasoning.",
        }


class _AuditProblem(Exception):
    def __init__(self, status: AuditStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise _AuditProblem(AuditStatus.MISMATCH, code, message)


def _finite(value: object) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read(read: Callable[[], _T], reference: str) -> _T:
    try:
        value = read()
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise _AuditProblem(
            AuditStatus.MISMATCH, "invalid_record", f"Cannot decode {reference}: {type(exc).__name__}.",
        ) from exc
    if value is None:
        raise _AuditProblem(
            AuditStatus.MISSING_RECORD, "record_unavailable", f"Missing or unreadable {reference}.",
        )
    return value


def _fields(
    document: Any, checks: dict[str, Callable[[Any], bool]], reference: str,
) -> dict[str, Any]:
    """Check original JSON values before ordinary decoders can supply defaults."""
    _require(isinstance(document, dict), "invalid_record", f"{reference} must be a JSON object.")
    for name, check in checks.items():
        _require(
            name in document and check(document[name]), "invalid_record",
            f"Missing or invalid {reference}.{name}.",
        )
    return document


def _local_document(store: LocalFileStore, kind: str, item_id: str, agent_id: str) -> Any:
    # Use the backend's existing safe path encoding and read-error behavior,
    # but validate and decode this SAME read, never a separate preflight read.
    return _read(
        partial(store._read_json, store._path(kind, agent_id, item_id)), f"{kind}/{item_id}",
    )


def _load_run(store: LearningStore, run_id: str, agent_id: str) -> TrainingRun:
    reference = f"training run {run_id}"
    if not isinstance(store, LocalFileStore):
        run = _read(partial(store.get_run, run_id, agent_id), reference)
        assert run is not None
        return run
    document = _fields(_local_document(store, "runs", run_id, agent_id), {"id": _identifier}, reference)
    metadata = document.get("metadata")
    # Old/unsupported runs still receive insufficient_lineage. A native receipt
    # cannot use omitted legacy identity or batch fields as replay evidence.
    if isinstance(metadata, dict) and metadata.get("lineage_version") == 1:
        _fields(document, {
            "agent_id": _identifier,
            "task_id": _identifier,
            "episode_ids": lambda value: isinstance(value, list) and all(_identifier(eid) for eid in value),
        }, reference)
    return _read(partial(TrainingRun.from_dict, document), reference)


def _load_snapshot(store: LearningStore, policy_id: str, agent_id: str) -> PolicySnapshot:
    reference = f"snapshot {policy_id}"
    if not isinstance(store, LocalFileStore):
        snapshot = _read(partial(store.get_policy, policy_id, agent_id), reference)
        assert snapshot is not None
        return snapshot
    document = _fields(_local_document(store, "policies", policy_id, agent_id), {
        "id": _identifier,
        "agent_id": _identifier,
        "task_id": _identifier,
        "version": _count,
        "episodes_seen": _count,
        "updates_applied": _count,
        "baseline": _finite,
        "logits": lambda value: isinstance(value, dict) and all(
            _identifier(key) and _finite(logit) for key, logit in value.items()
        ),
        "actions": lambda value: isinstance(value, list),
        "metadata": lambda value: isinstance(value, dict),
    }, reference)
    for index, action in enumerate(document["actions"]):
        _fields(action, {
            "id": _identifier,
            "description": lambda value: value is None or isinstance(value, str),
            "parameters": lambda value: isinstance(value, dict),
        }, f"{reference}.actions[{index}]")
    return _read(partial(PolicySnapshot.from_dict, document), reference)


def _load_episode(store: LearningStore, episode_id: str, agent_id: str) -> Episode:
    reference = f"episode {episode_id}"
    if not isinstance(store, LocalFileStore):
        episode = _read(partial(store.get_episode, episode_id, agent_id), reference)
        assert episode is not None
        return episode
    document = _fields(_local_document(store, "episodes", episode_id, agent_id), {
        "id": _identifier,
        "agent_id": _identifier,
        "task_id": _identifier,
        "action_id": _identifier,
        "action_logprob": lambda value: value is None or (_finite(value) and value <= 0),
        "intent_summary": _identifier,
        "expected_outcome": _identifier,
        "execution_status": _identifier,
        "result_summary": _identifier,
    }, reference)
    return _read(partial(Episode.from_dict, document), reference)


def _selected_rewards(store: LearningStore, item: ConsumedInput, agent_id: str) -> list[Reward]:
    reference = f"rewards for episode {item.episode_id}"
    if not isinstance(store, LocalFileStore):
        rewards = _read(partial(store.get_rewards_for_episode, item.episode_id, agent_id), reference)
        return [reward for reward in rewards if reward.id == item.reward_id]
    documents = _local_document(store, "rewards", item.episode_id, agent_id)
    _require(isinstance(documents, list), "invalid_record", f"{reference} must be a JSON array.")
    selected = []
    for document in documents:
        _fields(document, {"id": _identifier}, reference)
        if document["id"] != item.reward_id:
            continue
        _fields(document, {
            "episode_id": _identifier,
            "agent_id": _identifier,
            "source": lambda value: isinstance(value, str),
            "value": _finite,
            "created_at": _identifier,
        }, f"reward {item.reward_id}")
        selected.append(_read(partial(Reward.from_dict, document), f"reward {item.reward_id}"))
    return selected


def _configuration(run: TrainingRun) -> tuple[LearnerConfig, float]:
    if (
        run.status != TrainingStatus.SUCCEEDED
        or not isinstance(run.metadata, dict)
        or type(run.metadata.get("lineage_version")) is not int
        or run.metadata.get("lineage_version") != 1
        or run.metadata.get("replay_implementation") != "reinforce_softmax_v1"
        or run.algorithm != "ReinforceLearner"
    ):
        raise _AuditProblem(
            AuditStatus.INSUFFICIENT_LINEAGE, "unsupported_run",
            "Replay requires a successful native reinforce_softmax_v1 run with lineage version 1.",
        )
    if run.consumed_inputs is None or not run.policy_id or not run.output_policy_id:
        raise _AuditProblem(
            AuditStatus.INSUFFICIENT_LINEAGE, "missing_lineage",
            "The run lacks a consumption receipt or an exact input/output snapshot reference.",
        )
    learner_keys = {"learning_rate", "baseline_decay", "entropy_bonus", "importance_clip"}
    parameters = run.hyperparameters
    if (
        not isinstance(parameters, dict)
        or not isinstance(parameters.get("learner"), dict)
        or not isinstance(parameters.get("policy"), dict)
        or not learner_keys.issubset(parameters["learner"])
        or "max_logit_abs" not in parameters["policy"]
    ):
        raise _AuditProblem(
            AuditStatus.INSUFFICIENT_LINEAGE, "missing_configuration",
            "Applied learner parameters and the policy clipping limit must be recorded.",
        )
    learner = parameters["learner"]
    policy = parameters["policy"]
    _require(
        set(parameters) == {"learner", "policy"} and set(learner) == learner_keys
        and set(policy) == {"max_logit_abs"},
        "invalid_configuration", "Unexpected parameters for reinforce_softmax_v1.",
    )
    _require(
        all(_finite(value) for value in learner.values()) and _finite(policy["max_logit_abs"]),
        "invalid_configuration", "Applied parameters must be finite numbers, not strings or booleans.",
    )
    _require(
        learner["learning_rate"] >= 0 and 0 <= learner["baseline_decay"] <= 1
        and learner["entropy_bonus"] >= 0 and learner["importance_clip"] > 0
        and policy["max_logit_abs"] > 0,
        "invalid_configuration", "Applied parameters are outside supported ranges.",
    )
    # Supply every field explicitly so replay never consults current environment defaults.
    return LearnerConfig(
        **learner, max_logit_abs=policy["max_logit_abs"], min_train_episodes=0,
    ), policy["max_logit_abs"]


def _validate_snapshot(snapshot: PolicySnapshot, identifier: str, run: TrainingRun) -> None:
    _require(
        snapshot.id == identifier and snapshot.agent_id == run.agent_id and snapshot.task_id == run.task_id,
        "snapshot_identity", f"Snapshot {identifier} does not belong to the recorded agent/task.",
    )
    _require(isinstance(snapshot.metadata, dict), "invalid_snapshot", "Snapshot metadata must be an object.")
    if (
        snapshot.metadata.get("decision_authority", "low") != "low"
        or snapshot.metadata.get("policy_kind", "softmax") not in ("softmax", "softmax_bandit")
        or "context_weights" in snapshot.metadata
    ):
        raise _AuditProblem(
            AuditStatus.INSUFFICIENT_LINEAGE, "unsupported_policy",
            "Only the marginal learned policy is supported; reasoned and contextual decisions are excluded.",
        )
    action_ids = [action.id for action in snapshot.actions]
    _require(
        bool(action_ids) and all(_identifier(aid) for aid in action_ids)
        and len(set(action_ids)) == len(action_ids),
        "invalid_snapshot", f"Snapshot {identifier} must have unique, nonempty action IDs.",
    )
    _require(
        isinstance(snapshot.logits, dict) and set(snapshot.logits) == set(action_ids)
        and all(_finite(value) for value in snapshot.logits.values()) and _finite(snapshot.baseline)
        and all(_count(value) for value in (snapshot.version, snapshot.episodes_seen, snapshot.updates_applied)),
        "invalid_snapshot", f"Snapshot {identifier} contains invalid numerical state.",
    )


def _inputs(
    store: LearningStore, run: TrainingRun, snapshot: PolicySnapshot,
) -> tuple[list[Episode], list[Reward]]:
    receipt = run.consumed_inputs
    _require(isinstance(receipt, list), "invalid_receipt", "Consumption receipt must be a list.")
    assert receipt is not None
    _require(
        isinstance(run.episode_ids, list) and all(_identifier(eid) for eid in run.episode_ids),
        "invalid_receipt", "Queried episode IDs must be an ordered list of identifiers.",
    )
    action_ids = {action.id for action in snapshot.actions}
    episodes: list[Episode] = []
    rewards: list[Reward] = []
    cached: dict[str, tuple[Episode, Reward]] = {}
    queried = iter(run.episode_ids)
    for item in receipt:
        _require(
            isinstance(item, ConsumedInput) and _identifier(item.episode_id) and _identifier(item.reward_id),
            "invalid_receipt", "Each consumed input requires episode and reward IDs.",
        )
        _require(
            any(eid == item.episode_id for eid in queried),
            "receipt_order", "Consumed episodes must form an ordered subsequence of the queried batch.",
        )
        if item.episode_id in cached:
            episode, reward = cached[item.episode_id]
            _require(
                reward.id == item.reward_id, "conflicting_rewards",
                f"Episode {item.episode_id} has conflicting selected reward references.",
            )
        else:
            episode = copy.deepcopy(_load_episode(store, item.episode_id, run.agent_id))
            _require(
                episode.id == item.episode_id and episode.agent_id == run.agent_id
                and episode.task_id == run.task_id and episode.is_full,
                "episode_identity", f"Episode {item.episode_id} is incomplete or belongs to a different task.",
            )
            _require(
                _identifier(episode.action_id) and episode.action_id in action_ids, "unknown_action",
                f"Episode {item.episode_id} uses an action outside the input snapshot.",
            )
            _require(
                episode.action_logprob is None
                or (_finite(episode.action_logprob) and episode.action_logprob <= 0),
                "invalid_logprob", f"Episode {item.episode_id} has an invalid behavior log probability.",
            )
            selected = _selected_rewards(store, item, run.agent_id)
            if not selected:
                raise _AuditProblem(
                    AuditStatus.MISSING_RECORD, "reward_unavailable",
                    f"Reward {item.reward_id} is unavailable for episode {item.episode_id}.",
                )
            _require(len(selected) == 1, "ambiguous_reward", f"Reward {item.reward_id} is not unique.")
            reward = copy.deepcopy(selected[0])
            _require(
                reward.episode_id == episode.id and reward.agent_id == run.agent_id
                and reward.source == RewardSource.AGGREGATE,
                "reward_identity", f"Reward {item.reward_id} is not the episode's aggregate reward.",
            )
            _require(
                _finite(reward.value) and -1 <= reward.value <= 1,
                "invalid_reward", f"Reward {item.reward_id} must be finite and within [-1, 1].",
            )
            cached[item.episode_id] = (episode, reward)
        episodes.append(episode)
        rewards.append(reward)
    return episodes, rewards


def _projection(snapshot: PolicySnapshot) -> dict[str, Any]:
    probabilities = SoftmaxPolicy.from_snapshot(copy.deepcopy(snapshot)).probabilities()
    return {
        "id": snapshot.id,
        "version": snapshot.version,
        "logits": dict(snapshot.logits),
        "baseline": snapshot.baseline,
        "episodes_seen": snapshot.episodes_seen,
        "updates_applied": snapshot.updates_applied,
        "probabilities": dict(zip((a.id for a in snapshot.actions), probabilities)),
    }


def _compare(report: AuditReport, path: str, recorded: Any, replayed: Any) -> None:
    if isinstance(replayed, dict):
        if isinstance(recorded, dict) and set(recorded) == set(replayed):
            for key, value in replayed.items():
                _compare(report, f"{path}.{key}", recorded[key], value)
            return
    elif type(replayed) is int:
        if type(recorded) is int and recorded == replayed:
            return
    elif isinstance(replayed, str):
        if isinstance(recorded, str) and recorded == replayed:
            return
    elif _finite(replayed) and _finite(recorded) and math.isclose(
        recorded, replayed, rel_tol=_REL_TOL, abs_tol=_ABS_TOL,
    ):
        return
    report.differences.append({
        "field": path,
        "recorded": repr(recorded),
        "replayed": repr(replayed),
    })


def audit_run(store: LearningStore, agent_id: str, run_id: str) -> AuditReport:
    """Reconcile a run without writing, scoring, sampling, or invoking tools.

    Only local and in-memory stores are supported. Other backends are rejected
    before I/O because initialization may provision resources. Storage errors
    not classified as malformed records propagate to the caller.
    """
    report = AuditReport(agent_id=agent_id, run_id=run_id)
    if type(store) not in (LocalFileStore, InMemoryStore):
        report.reason_code = "unsupported_store"
        report.message = "Read-only replay currently supports only LocalFileStore and InMemoryStore."
        return report
    try:
        run = copy.deepcopy(_load_run(store, run_id, agent_id))
        _require(
            run.id == run_id and run.agent_id == agent_id and _identifier(run.task_id),
            "run_identity", "Training run identity does not match the requested agent/run.",
        )
        report.task_id = run.task_id
        config, limit = _configuration(run)
        assert run.output_policy_id is not None
        before = copy.deepcopy(_load_snapshot(store, run.policy_id, agent_id))
        after = copy.deepcopy(_load_snapshot(store, run.output_policy_id, agent_id))
        _validate_snapshot(before, run.policy_id, run)
        _validate_snapshot(after, run.output_policy_id, run)
        _require(
            [a.to_dict() for a in before.actions] == [a.to_dict() for a in after.actions],
            "action_space_changed", "Input and output snapshots must preserve the ordered action definitions.",
        )
        episodes, rewards = _inputs(store, run, before)
        _require(
            isinstance(run.metrics, dict) and _count(run.metrics.get("episodes_used"))
            and run.metrics["episodes_used"] == len(episodes),
            "receipt_count", "Receipt size must equal the recorded consumed-episode count.",
        )
        _require(
            (run.policy_id == run.output_policy_id) == (not episodes),
            "snapshot_transition", "Snapshot identities must change only when inputs were consumed.",
        )
        policy = SoftmaxPolicy.from_snapshot(copy.deepcopy(before), max_logit_abs=limit)
        result = ReinforceLearner(config).update(policy, episodes, rewards)
        _require(
            all(_finite(value) for value in result.logit_deltas.values())
            and all(_finite(value) for value in (result.mean_reward, result.baseline_before, result.baseline_after)),
            "nonfinite_update", "Reconstruction produced a nonfinite update; it cannot be verified.",
        )
        replayed = policy.snapshot()
        _validate_snapshot(replayed, replayed.id, run)
        _require(result.consumed_inputs == run.consumed_inputs, "receipt_mismatch", "Replay consumed different inputs.")

        report.input_policy = _projection(before)
        report.output_policy = _projection(after)
        for key in ("episodes_used", "mean_reward", "baseline_before", "baseline_after", "logit_deltas", "extra"):
            _compare(report, f"metrics.{key}", run.metrics.get(key), getattr(result, key))
        for key in ("version", "baseline", "logits", "episodes_seen", "updates_applied"):
            _compare(report, f"output.{key}", getattr(after, key), getattr(replayed, key))
        _compare(report, "metadata.policy_version", run.metadata.get("policy_version"), replayed.version)
        _compare(
            report, "output.probabilities", report.output_policy["probabilities"],
            dict(zip((a.id for a in before.actions), policy.probabilities())),
        )
        applied = {aid: after.logits[aid] - value for aid, value in before.logits.items()}
        _require(
            all(_finite(value) for value in applied.values()),
            "nonfinite_difference", "The recorded snapshot difference overflowed.",
        )
        report.summary = {
            "queried_episodes": len(run.episode_ids),
            "consumed_episodes": len(episodes),
            "no_update": not episodes,
            "hyperparameters": copy.deepcopy(run.hyperparameters),
            "proposed_deltas": dict(result.logit_deltas),
            "recorded_applied_deltas": applied,
        }
        if not report.differences:
            _attribute(report, before, config, limit, episodes, rewards)
        report.status = AuditStatus.MISMATCH if report.differences else AuditStatus.VERIFIED
        report.reason_code = "numerical_mismatch" if report.differences else "reconciled"
        report.message = (
            "Recorded and reconstructed updates differ."
            if report.differences else "No update applied; recorded no-op reconciles."
            if not episodes else "Referenced inputs reproduce the recorded policy update."
        )
    except _AuditProblem as exc:
        report.status, report.reason_code, report.message = exc.status, exc.code, str(exc)
    return report


def _attribute(
    report: AuditReport, before: PolicySnapshot, config: LearnerConfig, limit: float,
    episodes: list[Episode], rewards: list[Reward],
) -> None:
    # Independent one-sample updates from the SAME baseline expose additive
    # reward terms using the native implementation, without a second gradient formula.
    reward_only = ReinforceLearner(replace(config, entropy_bonus=0.0))
    probabilities = list(report.input_policy["probabilities"].values()) if report.input_policy else []
    action_index = {action.id: index for index, action in enumerate(before.actions)}
    for episode, reward in zip(episodes, rewards):
        policy = SoftmaxPolicy.from_snapshot(copy.deepcopy(before), max_logit_abs=limit)
        result = reward_only.update(policy, [episode], [reward])
        deltas = {aid: delta / len(episodes) for aid, delta in result.logit_deltas.items()}
        _require(
            all(_finite(value) for value in deltas.values()),
            "nonfinite_attribution", "An episode's numerical contribution overflowed.",
        )
        report.contributions.append({
            "episode_id": episode.id,
            "reward_id": reward.id,
            "action_id": episode.action_id,
            "reward": reward.value,
            "advantage": reward.value - before.baseline,
            "behavior_probability": math.exp(episode.action_logprob) if episode.action_logprob is not None else None,
            "importance_weight": reward_only._importance_weight(episode, probabilities, action_index),
            "importance_fallback": episode.action_logprob is None,
            "reward_deltas": deltas,
            "reward_decomposition": _decomposition(reward),
        })
    try:
        summed = {
            aid: math.fsum(item["reward_deltas"][aid] for item in report.contributions)
            for aid in action_index
        }
    except OverflowError as exc:
        raise _AuditProblem(
            AuditStatus.MISMATCH, "nonfinite_attribution", "Batch reward attribution overflowed.",
        ) from exc
    no_entropy_policy = SoftmaxPolicy.from_snapshot(copy.deepcopy(before), max_logit_abs=limit)
    no_entropy = reward_only.update(no_entropy_policy, episodes, rewards)
    _require(
        all(_finite(value) for value in (*summed.values(), *no_entropy.logit_deltas.values())),
        "nonfinite_attribution", "Batch reward attribution overflowed.",
    )
    residuals = {aid: no_entropy.logit_deltas[aid] - summed[aid] for aid in summed}
    proposed = report.summary["proposed_deltas"]
    entropy = {aid: proposed[aid] - no_entropy.logit_deltas[aid] for aid in summed}
    clipping = {
        aid: report.summary["recorded_applied_deltas"][aid] - proposed[aid] for aid in summed
    }
    _require(
        all(_finite(value) for value in (*residuals.values(), *entropy.values(), *clipping.values())),
        "nonfinite_attribution", "An entropy or clipping adjustment overflowed.",
    )
    # Attribution scales/accumulates in a different order from the native
    # batch update. Its rounding residual does not invalidate matching replay.
    rounding_exceeds_tolerance = any(
        not math.isclose(summed[aid], no_entropy.logit_deltas[aid], rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
        for aid in summed
    )
    report.summary["attribution_rounding_residuals"] = residuals
    report.summary["attribution_reward_totals"] = summed
    report.summary["attribution_status"] = "rounding_residual" if rounding_exceeds_tolerance else "reconciled"
    if rounding_exceeds_tolerance:
        report.warnings.append(
            "Episode attribution has a floating-point rounding residual; "
            "the persisted-state replay comparisons are unchanged."
        )
    report.summary["entropy_deltas"] = entropy
    report.summary["clipping_adjustments"] = clipping
    if any(episode.action_logprob is None for episode in episodes):
        report.warnings.append("Missing behavior probability: importance weight uses the native fallback of 1.")
    if any(item["reward_decomposition"] is None for item in report.contributions):
        report.warnings.append("Reward decomposition unavailable for one or more selected aggregates.")


def _decomposition(reward: Reward) -> dict[str, Any] | None:
    if not isinstance(reward.metadata, dict):
        return None
    contributions = reward.metadata.get("metric_contributions")
    penalties = reward.metadata.get("penalties")
    if not isinstance(contributions, list) or not isinstance(penalties, list):
        return None
    # Return only the documented numeric decomposition, not arbitrary metadata.
    if not all(
        isinstance(item, dict) and isinstance(item.get("metric"), str)
        and all(_finite(item.get(key)) for key in ("weight", "signed", "contribution"))
        for item in contributions
    ) or not all(
        isinstance(item, dict) and isinstance(item.get("kind"), str) and _finite(item.get("value"))
        for item in penalties
    ):
        return None
    return {
        "metric_contributions": [
            {key: item[key] for key in ("metric", "weight", "signed", "contribution")} for item in contributions
        ],
        "penalties": [{key: item[key] for key in ("kind", "value")} for item in penalties],
    }


__all__ = ["AuditReport", "AuditStatus", "audit_run"]
