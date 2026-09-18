"""Training-run consumption receipts and effective configuration."""

from __future__ import annotations

import platform
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from agent_learning import __version__
from agent_learning.config import LearnerConfig
from agent_learning.learners import Learner, LearnerResult, ReinforceLearner
from agent_learning.policy import Policy, SoftmaxPolicy
from agent_learning.storage import InMemoryStore, LocalFileStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    ConsumedInput,
    Episode,
    MetricName,
    MetricResult,
    Reward,
    RewardSource,
    TrainingStatus,
)


def _episode(episode_id: str, action_id: str, second: int = 0) -> Episode:
    return Episode(
        id=episode_id,
        agent_id="agent",
        task_id="routing",
        action_id=action_id,
        intent_summary="Choose a routing strategy",
        expected_outcome="Route to the appropriate handler",
        execution_status="completed",
        result_summary="Synthetic routing result",
        created_at=f"2026-09-15T00:00:{second:02d}+00:00",
    )


def _reward(reward_id: str, episode_id: str, value: float, second: int = 0) -> Reward:
    return Reward(
        id=reward_id,
        episode_id=episode_id,
        agent_id="agent",
        source=RewardSource.AGGREGATE,
        value=value,
        created_at=f"2026-09-15T01:00:{second:02d}+00:00",
    )


def test_native_run_persists_receipt_and_effective_configuration(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="agent", task_id="routing", max_logit_abs=3.0,
    )
    before = policy.snapshot()
    store.store_policy(before)
    config = LearnerConfig(
        learning_rate=0.4, baseline_decay=0.8, entropy_bonus=0.02,
        importance_clip=2.0, max_logit_abs=0.01, min_train_episodes=100,
    )
    for ep in (
        _episode("first", "a", 1), _episode("second", "b", 2),
        _episode("unknown", "outside", 3), _episode("unrewarded", "a", 4),
    ):
        store.store_episode(ep)
    for reward in (
        _reward("first-old", "first", -0.5, 1),
        _reward("first-selected", "first", 0.8, 2),
        _reward("second-selected", "second", -0.6, 3),
        _reward("ignored", "unknown", 0.4, 4),
    ):
        store.store_reward(reward)

    run = LearningRunner(
        store=store, policy=policy, metrics=[], learner=ReinforceLearner(config),
        learner_config=LearnerConfig(learning_rate=0.001),
    ).run_offline_batch("agent", task_id="routing", score_missing=False)
    after = policy.snapshot()
    config.learning_rate = 0.9
    del store, policy

    reloaded_store = LocalFileStore(tmp_path)
    reloaded_store.store_reward(_reward("post-training-rescore", "first", -1.0, 10))
    reloaded = reloaded_store.get_run(run.id, "agent")
    assert reloaded is not None
    assert reloaded == run
    assert reloaded.status == TrainingStatus.SUCCEEDED
    assert reloaded.policy_id == before.id
    assert reloaded.output_policy_id == after.id
    assert before.id != after.id
    assert reloaded.episode_ids == ["unrewarded", "unknown", "second", "first"]
    assert reloaded.consumed_inputs == [
        ConsumedInput("second", "second-selected"),
        ConsumedInput("first", "first-selected"),
    ]
    assert reloaded.metrics["episodes_used"] == 2
    assert reloaded.hyperparameters == {
        "learner": {
            "learning_rate": 0.4, "baseline_decay": 0.8,
            "entropy_bonus": 0.02, "importance_clip": 2.0,
        },
        "policy": {"max_logit_abs": 3.0},
    }
    assert reloaded.metadata == {
        "policy_version": after.version,
        "task_id": "routing",
        "score_missing": False,
        "lineage_version": 1,
        "replay_implementation": "reinforce_softmax_v1",
        "runtime_versions": {
            "agent_learning": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    assert reloaded_store.get_policy(before.id, "agent") == before
    assert reloaded_store.get_policy(after.id, "agent") == after
    assert reloaded_store.get_active_policy("agent", "routing") == after
    selected = [
        reward for item in reloaded.consumed_inputs
        for reward in reloaded_store.get_rewards_for_episode(item.episode_id, "agent")
        if reward.id == item.reward_id
    ]
    assert [reward.value for reward in selected] == [-0.6, 0.8]


@pytest.mark.parametrize("with_unrewarded_episode", [False, True])
def test_native_noop_persists_empty_receipt(tmp_path: Path, with_unrewarded_episode: bool) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="agent", task_id="routing",
    )
    before = policy.snapshot()
    store.store_policy(before)
    if with_unrewarded_episode:
        store.store_episode(_episode("unrewarded", "a"))

    run = LearningRunner(store=store, policy=policy, metrics=[]).run_offline_batch(
        "agent", task_id="routing", score_missing=False,
    )
    persisted = LocalFileStore(tmp_path).get_run(run.id, "agent")

    assert persisted is not None
    assert persisted.status == TrainingStatus.SUCCEEDED
    assert persisted.consumed_inputs == []
    assert persisted.episode_ids == (["unrewarded"] if with_unrewarded_episode else [])
    assert persisted.metrics["episodes_used"] == 0
    assert persisted.policy_id == persisted.output_policy_id == before.id
    assert persisted.metadata["lineage_version"] == 1
    assert policy.snapshot() == before


def test_configuration_is_captured_after_scoring_before_update() -> None:
    store = InMemoryStore()
    policy = SoftmaxPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="agent", task_id="routing",
    )
    store.store_episode(_episode("scored", "a"))
    config = LearnerConfig(learning_rate=0.1)

    class ScoringRunner(LearningRunner):
        def evaluate_episode(self, episode: Episode) -> list[MetricResult]:
            config.learning_rate = 0.7
            return [
                MetricResult(
                    metric=MetricName.TASK_COMPLETION, score=1.0,
                    normalized=1.0, status="completed",
                )
            ]

    run = ScoringRunner(
        store=store, policy=policy, learner=ReinforceLearner(config), metrics=[],
    ).run_offline_batch("agent", task_id="routing")

    assert run.hyperparameters["learner"]["learning_rate"] == 0.7
    assert run.consumed_inputs is not None and len(run.consumed_inputs) == 1
    selected = run.consumed_inputs[0]
    assert selected.episode_id == "scored"
    stored = store.get_rewards_for_episode("scored", "agent")
    assert any(r.id == selected.reward_id and r.source == RewardSource.AGGREGATE for r in stored)


class LegacyLearner(Learner):
    def update(
        self, policy: Policy, episodes: Iterable[Episode], rewards: Iterable[Reward],
    ) -> LearnerResult:
        before = policy.snapshot()
        return LearnerResult(0, 0.0, before.baseline, before.baseline)


def test_legacy_custom_learner_still_works_without_lineage() -> None:
    store = InMemoryStore()
    policy = SoftmaxPolicy.from_actions([Action(id="a")])
    before = policy.snapshot()

    run = LearningRunner(
        store=store, policy=policy, metrics=[], learner=LegacyLearner(),
    ).run_offline_batch("default", score_missing=False)

    assert run.status == TrainingStatus.SUCCEEDED
    assert run.output_policy_id == before.id
    assert run.consumed_inputs is None
    assert run.hyperparameters == {}
    assert "lineage_version" not in run.metadata
    assert "replay_implementation" not in run.metadata


class DerivedLearner(ReinforceLearner):
    pass


class DerivedPolicy(SoftmaxPolicy):
    pass


@pytest.mark.parametrize(
    ("learner_type", "policy_type"),
    [(DerivedLearner, SoftmaxPolicy), (ReinforceLearner, DerivedPolicy)],
)
def test_subclasses_are_not_claimed_as_native_replay(
    learner_type: type[ReinforceLearner], policy_type: type[SoftmaxPolicy],
) -> None:
    policy = policy_type.from_actions([Action(id="a")])

    run = LearningRunner(
        store=InMemoryStore(), policy=policy, metrics=[], learner=learner_type(),
    ).run_offline_batch("default", score_missing=False)

    assert run.status == TrainingStatus.SUCCEEDED
    assert run.consumed_inputs == []
    assert run.hyperparameters == {}
    assert "lineage_version" not in run.metadata
    assert "replay_implementation" not in run.metadata


def test_failed_native_update_does_not_record_an_output(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="agent", task_id="routing",
    )
    before = policy.snapshot()
    store.store_policy(before)
    store.store_episode(_episode("invalid", "a"))
    store.store_reward(_reward("invalid-reward", "invalid", 1.5))
    runner = LearningRunner(store=store, policy=policy, metrics=[])

    with pytest.raises(ValueError, match="aggregate reward value must be finite"):
        runner.run_offline_batch("agent", task_id="routing", score_missing=False)

    [run] = LocalFileStore(tmp_path).list_training_runs("agent")
    assert run.status == TrainingStatus.FAILED
    assert run.output_policy_id is None
    assert run.consumed_inputs is None
    assert run.error_message
    assert policy.snapshot() == before
