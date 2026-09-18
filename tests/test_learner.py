"""Tests for the REINFORCE learner.

We assert two important properties:

1. A consistently positive reward for one action shifts its
   logit upwards (and the other action's logit down).
2. Episodes without an aggregate reward or an unknown action are
   skipped without errors.
"""

from __future__ import annotations

import random

import pytest

from agent_learning.config import LearnerConfig
from agent_learning.learners import ReinforceLearner
from agent_learning.policy import SoftmaxPolicy
from agent_learning.types import Action, ConsumedInput, Episode, Reward, RewardSource


def _make_episode(agent_id: str, action_id: str, logprob: float = None) -> Episode:
    return Episode(agent_id=agent_id, user_input="x", assistant_output="y", action_id=action_id, action_logprob=logprob)


def test_positive_reward_shifts_logit_up() -> None:
    actions = [Action(id="a"), Action(id="b")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba", rng=random.Random(0))
    learner = ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.0))

    episodes = []
    rewards = []
    for _ in range(20):
        ep = _make_episode("nba", "a")
        episodes.append(ep)
        rewards.append(
            Reward(
                episode_id=ep.id,
                agent_id="nba",
                source=RewardSource.AGGREGATE,
                value=0.8,
            )
        )

    before = policy.snapshot()
    result = learner.update(policy, episodes, rewards)
    after = policy.snapshot()

    assert result.episodes_used == 20
    assert result.consumed_inputs == [
        ConsumedInput(ep.id, reward.id) for ep, reward in zip(episodes, rewards)
    ]
    assert after.logits["a"] > before.logits["a"]
    assert after.logits["b"] < before.logits["b"]
    assert after.version == before.version + 1
    assert after.episodes_seen == 20
    # Mean reward should be exactly 0.8 because every aggregate is identical.
    assert abs(result.mean_reward - 0.8) < 1e-9


def test_no_aggregate_reward_yields_noop() -> None:
    actions = [Action(id="a"), Action(id="b")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba")
    learner = ReinforceLearner(LearnerConfig())

    # Pass non-aggregate rewards only - learner must skip them.
    ep = _make_episode("nba", "a")
    rewards = [
        Reward(
            episode_id=ep.id,
            agent_id="nba",
            source=RewardSource.METRIC,
            value=0.9,
        )
    ]
    result = learner.update(policy, [ep], rewards)
    assert result.episodes_used == 0
    assert result.consumed_inputs == []


def test_unknown_action_is_skipped() -> None:
    actions = [Action(id="a")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba")
    learner = ReinforceLearner(LearnerConfig())

    ep = _make_episode("nba", "missing")
    rewards = [
        Reward(
            episode_id=ep.id,
            agent_id="nba",
            source=RewardSource.AGGREGATE,
            value=1.0,
        )
    ]
    result = learner.update(policy, [ep], rewards)
    assert result.episodes_used == 0
    assert result.consumed_inputs == []


def test_most_recent_aggregate_reward_wins() -> None:
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    learner = ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.0))
    episode = _make_episode("default", "a")
    rewards = [
        Reward(
            episode_id=episode.id,
            agent_id=episode.agent_id,
            source=RewardSource.AGGREGATE,
            value=0.0,
            created_at="2026-08-09T00:00:00+00:00",
        ),
        Reward(
            episode_id=episode.id,
            agent_id=episode.agent_id,
            source=RewardSource.AGGREGATE,
            value=0.8,
            created_at="2026-08-09T00:00:01+00:00",
        ),
    ]

    result = learner.update(policy, [episode], rewards)

    assert result.mean_reward == 0.8
    assert result.consumed_inputs == [ConsumedInput(episode.id, rewards[1].id)]
    assert policy.snapshot().logits["a"] > 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1.01, -1.01])
def test_invalid_aggregate_reward_is_rejected_without_policy_update(value: float) -> None:
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    learner = ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.0))
    episode = _make_episode("default", "a")
    reward = Reward(
        episode_id=episode.id,
        agent_id=episode.agent_id,
        source=RewardSource.AGGREGATE,
        value=value,
    )
    before = policy.snapshot()

    with pytest.raises(ValueError, match=r"finite and within \[-1, 1\]"):
        learner.update(policy, [episode], [reward])

    assert policy.snapshot() == before


def test_receipt_preserves_consumption_order_and_repeated_inputs() -> None:
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    first = Episode(id="first", action_id="a")
    second = Episode(id="second", action_id="b")
    unknown = Episode(id="unknown", action_id="outside")
    missing_action = Episode(id="missing-action")
    unrewarded = Episode(id="unrewarded", action_id="a")
    rewards = [
        Reward(id=f"reward-{ep.id}", episode_id=ep.id, source=RewardSource.AGGREGATE, value=0.5)
        for ep in (first, second, unknown, missing_action)
    ]

    result = ReinforceLearner().update(
        policy,
        iter([second, unknown, missing_action, unrewarded, first, second]),
        iter(rewards),
    )

    assert result.episodes_used == 3
    assert result.consumed_inputs == [
        ConsumedInput("second", "reward-second"),
        ConsumedInput("first", "reward-first"),
        ConsumedInput("second", "reward-second"),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_reward_timestamps_record_first_encountered_reward(reverse: bool) -> None:
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    episode = Episode(id="episode", action_id="a")
    rewards = [
        Reward(
            id="positive", episode_id=episode.id, source=RewardSource.AGGREGATE,
            value=0.8, created_at="2026-09-15T00:00:00+00:00",
        ),
        Reward(
            id="negative", episode_id=episode.id, source=RewardSource.AGGREGATE,
            value=-0.8, created_at="2026-09-15T00:00:00+00:00",
        ),
    ]
    if reverse:
        rewards.reverse()

    result = ReinforceLearner().update(policy, [episode], rewards)

    assert result.mean_reward == rewards[0].value
    assert result.consumed_inputs == [ConsumedInput(episode.id, rewards[0].id)]


def test_empty_batch_has_empty_receipt_and_leaves_snapshot_unchanged() -> None:
    policy = SoftmaxPolicy.from_actions([Action(id="a"), Action(id="b")])
    before = policy.snapshot()

    result = ReinforceLearner().update(policy, [], [])

    assert result.consumed_inputs == []
    assert result.episodes_used == 0
    assert policy.snapshot() == before
