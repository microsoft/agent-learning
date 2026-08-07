"""Local file-system implementation of :class:`LearningStore`.

Persists each durable record type as JSON on the local disk, laid out
by record type and partitioned by ``agent_id``::

    {root}/
      episodes/{agent_id}/{episode_id}.json
      metrics/{agent_id}/{episode_id}.json     # list[MetricResult]
      rewards/{agent_id}/{episode_id}.json     # list[Reward]
      policies/{agent_id}/{policy_id}.json
      runs/{agent_id}/{run_id}.json

Writes are atomic (temp file + ``os.replace``) so a crash mid-write
never leaves a partially written record. The backend needs no external
services, which makes it a good fit for local development, demos, and
durable single-machine experiments where :class:`InMemoryStore`'s
volatility is undesirable and Cosmos DB is unavailable. It mirrors the
semantics of :class:`InMemoryStore`, so it is a drop-in durable
replacement.

Note: this backend is single-process (last writer wins); it does not
coordinate concurrent writers across processes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import quote

from ..types import AgentInfo, Episode, MetricResult, PolicySnapshot, Reward, TrainingRun
from .base import LearningStore

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = "./data/agent-learning/store"


def _safe_component(value: str) -> str:
    """Encode an id into a single filesystem-safe path component.

    Percent-encodes path separators and every other unsafe character
    and escapes ``.`` so the reserved names ``.``/``..`` can never be
    produced. This keeps user-supplied ids (``agent_id``,
    ``episode_id``, ...) from escaping the store root via path
    traversal.
    """
    encoded = quote(str(value), safe="").replace(".", "%2E")
    return encoded or "_empty_"


class LocalFileStore(LearningStore):
    """File-system-backed store whose data survives process exit.

    Not safe for concurrent writers across processes (last writer
    wins); intended for single-process local development, demos, and
    tests that need persistence between runs.
    """

    def __init__(self, root_dir: Optional[Union[str, "os.PathLike[str]"]] = None) -> None:
        resolved = (
            str(root_dir)
            if root_dir is not None
            else os.getenv("AGENT_LEARNING_LOCAL_STORE_DIR", _DEFAULT_ROOT)
        )
        self._root = Path(resolved).expanduser()

    # ------------------------------------------------------------------
    # Path + I/O helpers
    # ------------------------------------------------------------------

    def _dir(self, kind: str, agent_id: str) -> Path:
        return self._root / kind / _safe_component(agent_id)

    def _path(self, kind: str, agent_id: str, item_id: str) -> Path:
        return self._dir(kind, agent_id) / f"{_safe_component(item_id)}.json"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: serialise to a temp file in the same directory,
        # then replace the target. Prevents torn reads on crash.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_json(self, path: Path) -> Optional[Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("LocalFileStore: failed to read %s: %s", path, exc)
            return None

    def _read_dir_docs(self, kind: str, agent_id: str) -> List[Dict[str, Any]]:
        """Read every one-document-per-file record under a partition."""
        directory = self._dir(kind, agent_id)
        if not directory.is_dir():
            return []
        docs: List[Dict[str, Any]] = []
        for entry in directory.iterdir():
            if entry.suffix != ".json" or not entry.is_file():
                continue
            doc = self._read_json(entry)
            if isinstance(doc, dict):
                docs.append(doc)
        return docs

    def _read_all_docs(self, kind: str) -> List[Dict[str, Any]]:
        """Read all one-document-per-file records across agent partitions."""
        root = self._root / kind
        if not root.is_dir():
            return []
        docs: List[Dict[str, Any]] = []
        for entry in root.glob("*/*.json"):
            if not entry.is_file():
                continue
            doc = self._read_json(entry)
            if isinstance(doc, dict):
                docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def store_episode(self, episode: Episode) -> str:
        self._write_json(
            self._path("episodes", episode.agent_id, episode.id), episode.to_dict()
        )
        return episode.id

    def get_episode(self, episode_id: str, agent_id: str) -> Optional[Episode]:
        doc = self._read_json(self._path("episodes", agent_id, episode_id))
        return Episode.from_dict(doc) if doc else None

    def query_episodes(
        self,
        agent_id: str,
        *,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        policy_id: Optional[str] = None,
        completed_only: bool = False,
    ) -> List[Episode]:
        results = [
            ep
            for ep in (Episode.from_dict(d) for d in self._read_dir_docs("episodes", agent_id))
            if (start_date is None or ep.created_at >= start_date)
            and (end_date is None or ep.created_at <= end_date)
            and (policy_id is None or ep.policy_id == policy_id)
            and (not completed_only or ep.is_complete)
        ]
        results.sort(key=lambda ep: ep.created_at, reverse=True)
        return results[:limit]

    def count_completed_episodes(self, agent_id: str) -> int:
        return sum(
            Episode.from_dict(doc).is_complete
            for doc in self._read_dir_docs("episodes", agent_id)
        )

    def list_agents(self) -> List[AgentInfo]:
        latest: Dict[str, PolicySnapshot] = {}
        for doc in self._read_all_docs("policies"):
            policy = PolicySnapshot.from_dict(doc)
            current = latest.get(policy.agent_id)
            if current is None or policy.version > current.version:
                latest[policy.agent_id] = policy
        return [
            AgentInfo.from_metadata(agent_id, latest[agent_id].metadata)
            for agent_id in sorted(latest)
        ]

    # ------------------------------------------------------------------
    # Metric results
    # ------------------------------------------------------------------

    def store_metric_results(
        self, episode_id: str, agent_id: str, results: Iterable[MetricResult]
    ) -> None:
        path = self._path("metrics", agent_id, episode_id)
        existing = self._read_json(path) or []
        existing.extend(result.to_dict() for result in results)
        self._write_json(path, existing)

    def get_metric_results(self, episode_id: str, agent_id: str) -> List[MetricResult]:
        docs = self._read_json(self._path("metrics", agent_id, episode_id)) or []
        return [MetricResult.from_dict(d) for d in docs]

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def store_reward(self, reward: Reward) -> str:
        path = self._path("rewards", reward.agent_id, reward.episode_id)
        existing = self._read_json(path) or []
        # De-duplicate by id (upsert semantics)
        existing = [r for r in existing if r.get("id") != reward.id]
        existing.append(reward.to_dict())
        self._write_json(path, existing)
        return reward.id

    def get_rewards_for_episode(self, episode_id: str, agent_id: str) -> List[Reward]:
        docs = self._read_json(self._path("rewards", agent_id, episode_id)) or []
        return [Reward.from_dict(d) for d in docs]

    def query_rewards(
        self,
        agent_id: str,
        *,
        episode_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        if episode_id is not None:
            return self.get_rewards_for_episode(episode_id, agent_id)[:limit]
        rewards: List[Reward] = []
        directory = self._dir("rewards", agent_id)
        if directory.is_dir():
            for entry in directory.iterdir():
                if entry.suffix != ".json" or not entry.is_file():
                    continue
                for doc in self._read_json(entry) or []:
                    rewards.append(Reward.from_dict(doc))
        rewards.sort(key=lambda r: r.created_at, reverse=True)
        return rewards[:limit]

    # ------------------------------------------------------------------
    # Policy snapshots
    # ------------------------------------------------------------------

    def store_policy(self, policy: PolicySnapshot) -> str:
        self._write_json(
            self._path("policies", policy.agent_id, policy.id), policy.to_dict()
        )
        return policy.id

    def get_policy(self, policy_id: str, agent_id: str) -> Optional[PolicySnapshot]:
        doc = self._read_json(self._path("policies", agent_id, policy_id))
        return PolicySnapshot.from_dict(doc) if doc else None

    def get_latest_policy(self, agent_id: str) -> Optional[PolicySnapshot]:
        docs = self._read_dir_docs("policies", agent_id)
        if not docs:
            return None
        latest = max(docs, key=lambda d: int(d.get("version", 0)))
        return PolicySnapshot.from_dict(latest)

    # ------------------------------------------------------------------
    # Training runs
    # ------------------------------------------------------------------

    def store_run(self, run: TrainingRun) -> str:
        self._write_json(self._path("runs", run.agent_id, run.id), run.to_dict())
        return run.id

    def get_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        doc = self._read_json(self._path("runs", agent_id, run_id))
        return TrainingRun.from_dict(doc) if doc else None

    def list_training_runs(
        self,
        agent_id: str,
        *,
        limit: int = 100,
    ) -> List[TrainingRun]:
        runs = [TrainingRun.from_dict(d) for d in self._read_dir_docs("runs", agent_id)]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]


__all__ = ["LocalFileStore"]
