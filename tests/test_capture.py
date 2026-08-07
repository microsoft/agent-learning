"""Tests for structured task episode capture."""

from __future__ import annotations

from agent_learning.capture import EpisodeCapture
from agent_learning.config import CaptureConfig
from agent_learning.storage import InMemoryStore
from agent_learning.types import Episode, PolicySnapshot, TrainingRun


def test_structured_task_fields_are_persisted() -> None:
    store = InMemoryStore()
    capture = EpisodeCapture(
        CaptureConfig(
            enabled=True,
            agent_id="agent-1",
            agent_name="Agent One",
            task_id="chat",
            task_name="Chat",
        ),
        store,
    )
    context = capture.start(
        "What changed?",
        intent_summary="summarize changes",
        action_type="chat",
        action_name="summarize",
        target="repository",
        input_summary="recent changes",
        expected_outcome="a concise summary",
        action_id="summarize",
    )

    episode = capture.end(
        context,
        "The policy changed.",
        execution_status="completed",
        result_summary="returned a concise summary",
    )

    assert episode is not None
    assert episode.agent_name == "Agent One"
    assert episode.task_id == "chat"
    assert episode.task_name == "Chat"
    assert episode.is_full
    assert store.get_episode(episode.id, "agent-1") == episode


def test_new_fields_do_not_shift_existing_positional_arguments() -> None:
    episode = Episode("episode-1", "agent-1", "input", "output")
    policy = PolicySnapshot("policy-1", "agent-1", 3)
    run = TrainingRun("run-1", "agent-1", "policy-1")

    assert episode.user_input == "input"
    assert episode.assistant_output == "output"
    assert policy.version == 3
    assert run.policy_id == "policy-1"