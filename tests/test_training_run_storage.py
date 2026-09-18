"""Additive training-run lineage across all storage backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_learning.storage import CosmosStore, InMemoryStore, LearningStore, LocalFileStore
from agent_learning.types import ConsumedInput, TrainingRun, TrainingStatus


class FakeRunContainer:
    def __init__(self) -> None:
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}

    def upsert_item(self, body: dict[str, Any]) -> None:
        self.documents[(body["agent_id"], body["id"])] = json.loads(json.dumps(body))

    def read_item(self, *, item: str, partition_key: str) -> dict[str, Any]:
        return json.loads(json.dumps(self.documents[(partition_key, item)]))


@pytest.fixture(params=["memory", "local", "cosmos"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> LearningStore:
    if request.param == "memory":
        return InMemoryStore()
    if request.param == "local":
        return LocalFileStore(tmp_path)
    assert request.param == "cosmos"
    cosmos = CosmosStore()
    monkeypatch.setattr(cosmos, "_ensure_ready", lambda: None)
    cosmos._containers["runs"] = FakeRunContainer()
    return cosmos


@pytest.mark.parametrize("receipt", [None, [], [ConsumedInput("ep2", "r2"), ConsumedInput("ep1", "r1")]])
def test_run_lineage_storage_contract(
    store: LearningStore, receipt: list[ConsumedInput] | None,
) -> None:
    run = TrainingRun(
        id="run",
        agent_id="agent",
        task_id="routing",
        policy_id="before",
        output_policy_id="after",
        status=TrainingStatus.SUCCEEDED,
        episode_ids=["ep2", "skipped", "ep1"],
        consumed_inputs=receipt,
        hyperparameters={"learner": {"learning_rate": 0.4}, "policy": {"max_logit_abs": 3.0}},
        metadata={"lineage_version": 1, "replay_implementation": "reinforce_softmax_v1"},
    )

    assert store.store_run(run) == "run"
    reloaded = store.get_run("run", "agent")

    assert reloaded is not None
    assert reloaded.to_dict() == run.to_dict()
    assert reloaded.consumed_inputs == receipt


def test_local_store_reads_legacy_run_json(tmp_path: Path) -> None:
    directory = tmp_path / "runs" / "agent"
    directory.mkdir(parents=True)
    (directory / "old-run.json").write_text(
        json.dumps({
            "id": "old-run", "agent_id": "agent", "task_id": "routing",
            "policy_id": "before", "episode_ids": ["ep1"], "status": "succeeded",
        }),
        encoding="utf-8",
    )

    run = LocalFileStore(tmp_path).get_run("old-run", "agent")

    assert run is not None
    assert run.policy_id == "before"
    assert run.output_policy_id is None
    assert run.consumed_inputs is None
    assert run.hyperparameters == {}
