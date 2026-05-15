"""Cosmos DB-backed implementation of :class:`LearningStore`.

The schema uses one container per record type, all partitioned by
``agent_id``. Container names, endpoint, and auth mode are sourced
from :class:`agent_learning.config.CosmosConfig` (overridable via env).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

try:  # azure-cosmos is required only when the Cosmos backend is used.
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore
    from azure.cosmos import exceptions as cosmos_exceptions  # type: ignore

    COSMOS_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded path
    CosmosClient = None  # type: ignore
    PartitionKey = None  # type: ignore
    cosmos_exceptions = None  # type: ignore
    COSMOS_SDK_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential  # type: ignore

    IDENTITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    DefaultAzureCredential = None  # type: ignore
    IDENTITY_AVAILABLE = False

from ..config import CosmosConfig
from ..types import Episode, MetricResult, PolicySnapshot, Reward, TrainingRun
from .base import LearningStore

logger = logging.getLogger(__name__)


class CosmosStore(LearningStore):
    """Cosmos DB-backed store.

    The store is lazily initialised: containers are created on first
    use. If the Cosmos SDK is not installed or the endpoint is not
    configured, an explicit :class:`RuntimeError` is raised when the
    caller attempts to use the store.
    """

    def __init__(
        self,
        config: Optional[CosmosConfig] = None,
        credential: Optional[Any] = None,
    ) -> None:
        self._config = config or CosmosConfig()
        self._credential = credential
        self._client: Optional[Any] = None
        self._database: Optional[Any] = None
        self._containers: Dict[str, Any] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._initialized:
            return
        if not COSMOS_SDK_AVAILABLE:
            raise RuntimeError(
                "azure-cosmos is not installed. Install azure-agents-learning-sdk with its "
                "default dependencies or use storage.InMemoryStore for local testing."
            )
        if not self._config.enabled:
            raise RuntimeError(
                "Cosmos endpoint is not configured. Set AGENT_LEARNING_COSMOS_ENDPOINT "
                "or pass an explicit CosmosConfig."
            )

        # Build the client
        if self._config.auth_mode.lower() == "key" and self._config.account_key:
            self._client = CosmosClient(self._config.endpoint, credential=self._config.account_key)
            logger.info("CosmosStore: connected with shared-key auth")
        else:
            if self._credential is None:
                if not IDENTITY_AVAILABLE:
                    raise RuntimeError(
                        "azure-identity is required for AAD auth but is not installed."
                    )
                self._credential = DefaultAzureCredential()
            self._client = CosmosClient(self._config.endpoint, credential=self._credential)
            logger.info("CosmosStore: connected with AAD auth")

        # Database + containers
        self._database = self._client.create_database_if_not_exists(self._config.database_name)
        pk = PartitionKey(path=f"/{self._config.partition_key_field}")

        for key, name in self._container_names().items():
            try:
                self._containers[key] = self._database.create_container_if_not_exists(
                    id=name, partition_key=pk
                )
            except cosmos_exceptions.CosmosHttpResponseError:
                self._containers[key] = self._database.get_container_client(name)
            logger.info("CosmosStore: container %s ready", name)

        self._initialized = True

    def _container_names(self) -> Dict[str, str]:
        return {
            "episodes": self._config.container_episodes,
            "rewards": self._config.container_rewards,
            "metrics": self._config.container_metrics,
            "policies": self._config.container_policies,
            "runs": self._config.container_runs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert(self, key: str, doc: Dict[str, Any]) -> None:
        self._ensure_ready()
        try:
            self._containers[key].upsert_item(doc)
        except cosmos_exceptions.CosmosHttpResponseError as exc:
            logger.error("Cosmos upsert failed (%s/%s): %s", key, doc.get("id"), exc.message)
            raise

    def _read(self, key: str, item_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_ready()
        try:
            return self._containers[key].read_item(item=item_id, partition_key=agent_id)
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
        except cosmos_exceptions.CosmosHttpResponseError as exc:
            logger.error("Cosmos read failed (%s/%s): %s", key, item_id, exc.message)
            return None

    def _query(
        self,
        key: str,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        *,
        partition_key: Optional[str] = None,
        cross_partition: bool = False,
    ) -> List[Dict[str, Any]]:
        self._ensure_ready()
        kwargs: Dict[str, Any] = {"query": query, "parameters": parameters or []}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        else:
            kwargs["enable_cross_partition_query"] = cross_partition
        try:
            return list(self._containers[key].query_items(**kwargs))
        except cosmos_exceptions.CosmosHttpResponseError as exc:
            logger.error("Cosmos query failed (%s): %s", key, exc.message)
            return []

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def store_episode(self, episode: Episode) -> str:
        self._upsert("episodes", episode.to_dict())
        return episode.id

    def get_episode(self, episode_id: str, agent_id: str) -> Optional[Episode]:
        doc = self._read("episodes", episode_id, agent_id)
        return Episode.from_dict(doc) if doc else None

    def query_episodes(
        self,
        agent_id: str,
        *,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> List[Episode]:
        clauses = ["c.agent_id = @agent_id"]
        params: List[Dict[str, Any]] = [{"name": "@agent_id", "value": agent_id}]
        if start_date:
            clauses.append("c.created_at >= @start_date")
            params.append({"name": "@start_date", "value": start_date})
        if end_date:
            clauses.append("c.created_at <= @end_date")
            params.append({"name": "@end_date", "value": end_date})
        if policy_id:
            clauses.append("c.policy_id = @policy_id")
            params.append({"name": "@policy_id", "value": policy_id})

        query = (
            "SELECT * FROM c WHERE "
            + " AND ".join(clauses)
            + " ORDER BY c.created_at DESC"
            + f" OFFSET 0 LIMIT {int(limit)}"
        )
        docs = self._query("episodes", query, params, partition_key=agent_id)
        return [Episode.from_dict(d) for d in docs]

    # ------------------------------------------------------------------
    # Metric results
    # ------------------------------------------------------------------

    def store_metric_results(
        self, episode_id: str, agent_id: str, results: Iterable[MetricResult]
    ) -> None:
        for result in results:
            doc = {
                "id": f"{episode_id}:{result.metric.value}",
                "episode_id": episode_id,
                "agent_id": agent_id,
                **result.to_dict(),
            }
            self._upsert("metrics", doc)

    def get_metric_results(self, episode_id: str, agent_id: str) -> List[MetricResult]:
        query = "SELECT * FROM c WHERE c.episode_id = @episode_id"
        params = [{"name": "@episode_id", "value": episode_id}]
        docs = self._query("metrics", query, params, partition_key=agent_id)
        return [MetricResult.from_dict(d) for d in docs]

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def store_reward(self, reward: Reward) -> str:
        self._upsert("rewards", reward.to_dict())
        return reward.id

    def get_rewards_for_episode(self, episode_id: str, agent_id: str) -> List[Reward]:
        query = "SELECT * FROM c WHERE c.episode_id = @episode_id"
        params = [{"name": "@episode_id", "value": episode_id}]
        docs = self._query("rewards", query, params, partition_key=agent_id)
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
        query = (
            "SELECT * FROM c WHERE c.agent_id = @agent_id "
            "ORDER BY c.created_at DESC "
            f"OFFSET 0 LIMIT {int(limit)}"
        )
        params = [{"name": "@agent_id", "value": agent_id}]
        docs = self._query("rewards", query, params, partition_key=agent_id)
        return [Reward.from_dict(d) for d in docs]

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------

    def store_policy(self, policy: PolicySnapshot) -> str:
        self._upsert("policies", policy.to_dict())
        return policy.id

    def get_policy(self, policy_id: str, agent_id: str) -> Optional[PolicySnapshot]:
        doc = self._read("policies", policy_id, agent_id)
        return PolicySnapshot.from_dict(doc) if doc else None

    def get_latest_policy(self, agent_id: str) -> Optional[PolicySnapshot]:
        query = (
            "SELECT TOP 1 * FROM c WHERE c.agent_id = @agent_id ORDER BY c.version DESC"
        )
        params = [{"name": "@agent_id", "value": agent_id}]
        docs = self._query("policies", query, params, partition_key=agent_id)
        return PolicySnapshot.from_dict(docs[0]) if docs else None

    # ------------------------------------------------------------------
    # Training runs
    # ------------------------------------------------------------------

    def store_run(self, run: TrainingRun) -> str:
        self._upsert("runs", run.to_dict())
        return run.id

    def get_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        doc = self._read("runs", run_id, agent_id)
        return TrainingRun.from_dict(doc) if doc else None

    def list_training_runs(
        self,
        agent_id: str,
        *,
        limit: int = 100,
    ) -> List[TrainingRun]:
        query = (
            "SELECT * FROM c WHERE c.agent_id = @agent_id "
            "ORDER BY c.created_at DESC "
            f"OFFSET 0 LIMIT {int(limit)}"
        )
        params = [{"name": "@agent_id", "value": agent_id}]
        docs = self._query("runs", query, params, partition_key=agent_id)
        return [TrainingRun.from_dict(d) for d in docs]


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

_default_store: Optional[LearningStore] = None


def get_default_store() -> LearningStore:
    """Return a process-wide singleton Cosmos-backed store.

    Falls back to :class:`InMemoryStore` if Cosmos is not configured;
    this keeps unit tests and local notebooks usable with the same
    code path as production.
    """
    global _default_store
    if _default_store is not None:
        return _default_store

    cfg = CosmosConfig()
    if cfg.enabled and COSMOS_SDK_AVAILABLE:
        _default_store = CosmosStore(cfg)
    else:
        # Lazy import to avoid a cycle at module load time
        from .memory import InMemoryStore  # noqa: WPS433

        logger.warning("CosmosStore unavailable; falling back to InMemoryStore")
        _default_store = InMemoryStore()
    return _default_store


__all__ = ["CosmosStore", "get_default_store"]
