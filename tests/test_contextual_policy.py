"""Tests for the contextual softmax bandit policy."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from agent_learning.policy import ContextualSoftmaxPolicy
from agent_learning.types import Action


def _make_policy(
    *, feature_dim: int = 4, seed: int = 0, num_actions: int = 3
) -> ContextualSoftmaxPolicy:
    actions = [Action(id=f"a{i}") for i in range(num_actions)]
    rng = random.Random(seed)
    return ContextualSoftmaxPolicy.from_actions(
        actions, agent_id="dq", feature_dim=feature_dim, rng=rng
    )


def test_uniform_when_weights_are_zero() -> None:
    policy = _make_policy(feature_dim=5, num_actions=4)
    phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    probs = policy.probabilities(phi)
    assert len(probs) == 4
    for p in probs:
        assert math.isclose(p, 0.25, abs_tol=1e-9)


def test_uniform_when_state_is_none() -> None:
    policy = _make_policy(feature_dim=4, num_actions=3)
    probs = policy.probabilities(None)
    for p in probs:
        assert math.isclose(p, 1 / 3, abs_tol=1e-9)


def test_choose_returns_consistent_logprob() -> None:
    policy = _make_policy(feature_dim=4, num_actions=3, seed=42)
    phi = np.array([1.0, 0.5, -0.5, 0.0])
    decision = policy.choose(state=phi)
    idx = next(
        i for i, a in enumerate(policy.actions()) if a.id == decision.action.id
    )
    assert math.isclose(
        math.exp(decision.logprob), decision.probabilities[idx], rel_tol=1e-9
    )


def test_apply_update_vector_delta_changes_weights_and_version() -> None:
    policy = _make_policy(feature_dim=3, num_actions=2)
    before_snap = policy.snapshot()
    delta_a0 = np.array([0.5, -0.3, 0.1])
    delta_a1 = np.array([-0.5, 0.3, -0.1])
    policy.apply_update(
        {"a0": delta_a0, "a1": delta_a1},
        baseline=0.2,
        episodes_seen=7,
    )
    after_snap = policy.snapshot()
    assert after_snap.version == before_snap.version + 1
    assert after_snap.id != before_snap.id
    assert after_snap.task_id == before_snap.task_id
    assert after_snap.baseline == 0.2
    assert after_snap.episodes_seen == 7
    w = after_snap.metadata["context_weights"]
    np.testing.assert_allclose(np.asarray(w["a0"]), delta_a0)
    np.testing.assert_allclose(np.asarray(w["a1"]), delta_a1)


def test_apply_update_scalar_delta_broadcasts_uniformly() -> None:
    policy = _make_policy(feature_dim=3, num_actions=2)
    policy.apply_update({"a0": 0.4, "a1": -0.4}, baseline=0.0, episodes_seen=1)
    w = policy.snapshot().metadata["context_weights"]
    np.testing.assert_allclose(np.asarray(w["a0"]), np.array([0.4, 0.4, 0.4]))
    np.testing.assert_allclose(np.asarray(w["a1"]), np.array([-0.4, -0.4, -0.4]))


def test_weight_clipping() -> None:
    actions = [Action(id="a0")]
    rng = random.Random(0)
    policy = ContextualSoftmaxPolicy.from_actions(
        actions, agent_id="dq", feature_dim=2, rng=rng, max_weight_abs=1.5
    )
    policy.apply_update({"a0": np.array([100.0, -100.0])}, baseline=0.0, episodes_seen=1)
    w = policy.snapshot().metadata["context_weights"]["a0"]
    assert max(abs(v) for v in w) <= 1.5
    assert w[0] == 1.5
    assert w[1] == -1.5


def test_snapshot_roundtrip_preserves_context_weights() -> None:
    policy = _make_policy(feature_dim=3, num_actions=2)
    policy.apply_update(
        {"a0": np.array([0.5, -0.5, 0.25]), "a1": np.array([-0.5, 0.5, -0.25])},
        baseline=0.1,
        episodes_seen=3,
    )
    snap = policy.snapshot()
    # ``feature_dim`` must round-trip via metadata so a restored policy
    # validates incoming state vectors against the right shape.
    assert snap.metadata["feature_dim"] == 3
    assert snap.metadata["policy_kind"] == "contextual_softmax"

    restored = ContextualSoftmaxPolicy(snapshot=snap)
    phi = np.array([0.2, 0.4, 0.6])
    np.testing.assert_allclose(
        np.asarray(policy.probabilities(phi)),
        np.asarray(restored.probabilities(phi)),
        atol=1e-9,
    )


def test_invalid_state_length_raises() -> None:
    policy = _make_policy(feature_dim=4)
    with pytest.raises(ValueError):
        policy.probabilities(np.array([1.0, 2.0]))  # wrong length
