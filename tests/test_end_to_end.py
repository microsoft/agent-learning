"""End-to-end loop test using a stub metric evaluator."""

from __future__ import annotations

import random
from typing import Any, Dict

from agent_learning.config import LearnerConfig, ShapingConfig
from agent_learning.learners import ReinforceLearner
from agent_learning.metrics.base import MetricEvaluator, MetricRequest
from agent_learning.policy import SoftmaxPolicy
from agent_learning.rewards import RewardShaper, RewardWriter
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
)


class _StubMetric(MetricEvaluator):
    """Always returns the score this fixture was given."""

    def __init__(self, name: MetricName, score_for_action: Dict[str, float]) -> None:
        # Avoid going through the parent's evaluator-config check
        super().__init__(evaluator=object())
        self._score_for_action = score_for_action
        self._name = name

    NAME = MetricName.INTENT_RESOLUTION  # placeholder - overridden below in __init_subclass__

    def _build_evaluator(self) -> Any:  # pragma: no cover - never called
        return self._evaluator

    def _build_kwargs(self, request: MetricRequest) -> Dict[str, Any]:  # pragma: no cover
        return {}

    def _normalize(self, raw: Dict[str, Any]) -> float:  # pragma: no cover
        return raw.get("normalized", 0.0)

    def evaluate(self, request: MetricRequest) -> MetricResult:  # type: ignore[override]
        # Use the action id embedded in the request via system_message hack
        action_id = request.extra.get("action_id", "")
        normalized = self._score_for_action.get(action_id, 0.0)
        return MetricResult(
            metric=self._name,
            score=normalized,
            normalized=normalized,
            status="completed",
            reason="stub",
        )


def test_end_to_end_offline_batch_improves_policy() -> None:
    actions = [Action(id="good"), Action(id="bad")]
    store = InMemoryStore()
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq", rng=random.Random(7))
    store.store_policy(policy.snapshot())

    # Create episodes where action "good" always gets near-perfect scores.
    for i in range(20):
        episode = Episode(
            id=f"good-{i}",
            agent_id="dq",
            user_input="task",
            assistant_output="ok",
            policy_id=policy.snapshot().id,
            policy_version=policy.snapshot().version,
            action_id="good",
        )
        store.store_episode(episode)
    for i in range(20):
        episode = Episode(
            id=f"bad-{i}",
            agent_id="dq",
            user_input="task",
            assistant_output="meh",
            policy_id=policy.snapshot().id,
            policy_version=policy.snapshot().version,
            action_id="bad",
        )
        store.store_episode(episode)

    # Stub metrics that read the action id from the episode metadata.
    score_table = {"good": 1.0, "bad": 0.0}

    def evaluate(metric: MetricName, ep: Episode) -> MetricResult:
        return MetricResult(
            metric=metric,
            score=score_table[ep.action_id or ""],
            normalized=score_table[ep.action_id or ""],
            status="completed",
        )

    class _Runner(LearningRunner):
        def evaluate_episode(self, episode: Episode):  # type: ignore[override]
            return [
                evaluate(MetricName.INTENT_RESOLUTION, episode),
                evaluate(MetricName.TASK_ADHERENCE, episode),
                evaluate(MetricName.TASK_COMPLETION, episode),
            ]

    runner = _Runner(
        store=store,
        policy=policy,
        metrics=[],
        shaper=RewardShaper(
            ShapingConfig(
                intent_resolution_weight=0.4,
                task_adherence_weight=0.3,
                task_completion_weight=0.3,
            )
        ),
        writer=RewardWriter(store),
        learner=ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.0)),
    )

    before = policy.snapshot()
    run = runner.run_offline_batch("dq", episode_limit=100)
    after = policy.snapshot()

    assert run.status.value == "succeeded"
    assert after.logits["good"] > before.logits["good"]
    assert after.logits["bad"] < before.logits["bad"]
    policy_history = store.list_policies("dq", "default")
    assert len(policy_history) == 2
    assert policy_history[0].id == after.id
    assert policy_history[1].id == before.id

    # After a single batch the policy must already lean toward "good".
    # A second batch would push this higher; here we just assert the
    # bandit moved meaningfully past uniform (0.5).
    probs = policy.probabilities()
    good_idx = next(i for i, a in enumerate(policy.actions()) if a.id == "good")
    assert probs[good_idx] > 0.55


def test_offline_batch_only_uses_selected_task() -> None:
    actions = [Action(id="good"), Action(id="bad")]
    store = InMemoryStore()
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq", task_id="chat")
    store.store_policy(policy.snapshot())
    store.store_episode(
        Episode(agent_id="dq", task_id="chat", action_id="good", assistant_output="ok")
    )
    store.store_episode(
        Episode(agent_id="dq", task_id="animation", action_id="bad", assistant_output="bad")
    )

    class _TaskRunner(LearningRunner):
        def evaluate_episode(self, episode: Episode):  # type: ignore[override]
            return [
                MetricResult(metric=name, score=1.0, normalized=1.0, status="completed")
                for name in MetricName
            ]

    runner = _TaskRunner(store=store, policy=policy, metrics=[])
    run = runner.run_offline_batch("dq", task_id="chat")

    assert run.task_id == "chat"
    assert len(run.episode_ids) == 1
    assert store.get_active_policy("dq", "chat") is not None
