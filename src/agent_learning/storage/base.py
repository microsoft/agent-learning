"""Abstract storage interface for the learning SDK.

The store is intentionally narrow: it persists the five durable
record types (episode, metric result, reward, policy snapshot,
training run) and supports the small set of queries needed by the
learner. New backends (Cosmos, SQL, in-memory) implement this
protocol; the rest of the SDK never imports a concrete backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from ..types import AgentInfo, Episode, MetricResult, PolicySnapshot, Reward, TrainingRun


class LearningStore(ABC):
    """Persistence contract for the agent-learning SDK."""

    # ---- Episodes --------------------------------------------------

    @abstractmethod
    def store_episode(self, episode: Episode) -> str:
        """Persist (or upsert) an episode and return its id."""

    @abstractmethod
    def get_episode(self, episode_id: str, agent_id: str) -> Optional[Episode]:
        """Fetch a single episode by id (and partition agent_id)."""

    @abstractmethod
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
        """List episodes filtered by optional time window / policy id."""

    @abstractmethod
    def count_completed_episodes(self, agent_id: str) -> int:
        """Return the number of finished episodes stored for an agent."""

    @abstractmethod
    def list_agents(self) -> List[AgentInfo]:
        """List agents that have a stored policy snapshot."""

    # ---- Metric results -------------------------------------------

    @abstractmethod
    def store_metric_results(
        self, episode_id: str, agent_id: str, results: Iterable[MetricResult]
    ) -> None:
        """Persist all metric results for an episode in one batch."""

    @abstractmethod
    def get_metric_results(self, episode_id: str, agent_id: str) -> List[MetricResult]:
        """Return all metric results stored for an episode."""

    # ---- Rewards ---------------------------------------------------

    @abstractmethod
    def store_reward(self, reward: Reward) -> str:
        """Persist (or upsert) a reward and return its id."""

    @abstractmethod
    def get_rewards_for_episode(self, episode_id: str, agent_id: str) -> List[Reward]:
        """Return all rewards attached to an episode."""

    @abstractmethod
    def query_rewards(
        self,
        agent_id: str,
        *,
        episode_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        """List rewards for an agent, optionally scoped to one episode."""

    # ---- Policy snapshots -----------------------------------------

    @abstractmethod
    def store_policy(self, policy: PolicySnapshot) -> str:
        """Persist (or upsert) a policy snapshot and return its id."""

    @abstractmethod
    def get_policy(self, policy_id: str, agent_id: str) -> Optional[PolicySnapshot]:
        """Fetch a single policy snapshot by id."""

    @abstractmethod
    def get_latest_policy(self, agent_id: str) -> Optional[PolicySnapshot]:
        """Return the highest-version policy snapshot for the agent."""

    # ---- Training runs --------------------------------------------

    @abstractmethod
    def store_run(self, run: TrainingRun) -> str:
        """Persist (or upsert) a training run and return its id."""

    @abstractmethod
    def get_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        """Fetch a training run by id."""

    @abstractmethod
    def list_training_runs(
        self,
        agent_id: str,
        *,
        limit: int = 100,
    ) -> List[TrainingRun]:
        """List recent training runs for an agent."""


__all__ = ["LearningStore"]
