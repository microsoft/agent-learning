"""Command-line interface for the agent-learning SDK.

Examples::

    # Initialise a durable policy for one agent task
    agent-learn task-policy-init --agent-id dq --task-id summary --actions ./actions.json

    # Ask the policy how to handle a task, then record its result
    agent-learn task-intent --agent-id dq --task-id summary --intent "Summarise Q3 sales"
    agent-learn task-complete --agent-id dq --episode-id <id> --output "..."

    # Judge completed episodes and update the policy
    agent-learn train --agent-id dq --task-id summary --limit 500

The CLI defaults to the local file store so state survives separate command
invocations. Explicit ``AGENT_LEARNING_STORE_BACKEND`` configuration still
selects another SDK backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from .capture import redact
from .config import LearnerConfig
from .policy.base import Policy
from .policy.contextual_softmax import ContextualSoftmaxPolicy
from .policy.softmax_bandit import SoftmaxPolicy
from .storage.base import LearningStore
from .storage.cosmos import get_default_store
from .storage.local import LocalFileStore
from .training.runner import LearningRunner
from .types import Action, Episode, PolicySnapshot

logger = logging.getLogger(__name__)


def _add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-dir",
        help=(
            "Local store directory. Defaults to AGENT_LEARNING_LOCAL_STORE_DIR "
            "or ./data/agent-learning/store."
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-learn", description="Native RL CLI for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run one offline learning batch.")
    train.add_argument("--agent-id", required=True)
    train.add_argument(
        "--task-id",
        help="Task policy to train. May be omitted when the agent has exactly one task.",
    )
    train.add_argument("--limit", type=int, default=200)
    train.add_argument("--start-date")
    train.add_argument("--end-date")
    train.add_argument(
        "--min-episodes",
        type=int,
        help=(
            "Minimum completed episodes required to train. Defaults to "
            "AGENT_LEARNING_MIN_TRAIN_EPISODES or 5."
        ),
    )
    train.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring episodes that have no rewards yet.",
    )
    _add_store_argument(train)

    score = sub.add_parser("score", help="Score episodes but skip the policy update.")
    score.add_argument("--agent-id", required=True)
    score.add_argument("--task-id", help="Only score episodes for this task.")
    score.add_argument("--limit", type=int, default=100)
    _add_store_argument(score)

    agents = sub.add_parser("agents-list", help="List agents with stored policies.")
    _add_store_argument(agents)

    episode_count = sub.add_parser(
        "agents-episodes-count",
        help="Print the number of completed episodes for an agent.",
    )
    episode_count.add_argument("agent_id")
    _add_store_argument(episode_count)

    episodes = sub.add_parser(
        "agents-episodes-list",
        help="Print completed episodes with their evaluation and reward details.",
    )
    episodes.add_argument("agent_id")
    episodes.add_argument("--task-id")
    episodes.add_argument("--limit", type=int, default=500)
    _add_store_argument(episodes)

    tasks = sub.add_parser(
        "agent-tasks-list",
        help="List tasks with stored policies for an agent.",
    )
    tasks.add_argument("agent_id")
    _add_store_argument(tasks)

    show = sub.add_parser(
        "task-policy",
        help="Print policy snapshots for an agent task, newest first.",
    )
    show.add_argument("--agent-id", required=True)
    show.add_argument("--task-id", required=True)
    show.add_argument(
        "--history",
        type=int,
        default=1,
        help="Number of policy snapshots to print.",
    )
    _add_store_argument(show)

    init = sub.add_parser(
        "task-policy-init",
        help="Create and activate a policy snapshot for an agent task.",
    )
    init.add_argument("--agent-id", required=True)
    init.add_argument("--task-id", required=True)
    init.add_argument(
        "--agent-name",
        help="Display name used by agents-list. Defaults to --agent-id.",
    )
    init.add_argument(
        "--task-name",
        help="Display name used by agent-tasks-list. Defaults to --task-id.",
    )
    init.add_argument(
        "--actions",
        required=True,
        help="Path to a JSON file containing a list of {id, description, parameters} objects.",
    )
    _add_store_argument(init)

    intent = sub.add_parser(
        "task-intent",
        help="Choose an action for a task and start a locally persisted episode.",
    )
    intent.add_argument("--agent-id", required=True)
    intent.add_argument("--task-id", required=True)
    intent.add_argument("--intent", required=True, help="The user's requested outcome.")
    intent.add_argument(
        "--context",
        "--features-json",
        dest="context",
        default="{}",
        help=(
            "JSON context object (or @path to a JSON file). A JSON array is treated "
            "as the contextual policy's phi feature vector."
        ),
    )
    intent.add_argument(
        "--episode-id",
        help="Optional caller-provided episode ID. A UUID is generated when omitted.",
    )
    _add_store_argument(intent)

    complete = sub.add_parser(
        "task-complete",
        help="Complete a task episode with the agent's output.",
    )
    complete.add_argument("--agent-id", required=True)
    complete.add_argument("--episode-id", required=True)
    complete.add_argument(
        "--output",
        "--result",
        "--assistant-output",
        dest="output",
        required=True,
        help="The final task output recorded for judging and training.",
    )
    _add_store_argument(complete)

    return parser


def _get_cli_store(args: argparse.Namespace) -> LearningStore:
    store_dir = getattr(args, "store_dir", None)
    if store_dir is not None or "AGENT_LEARNING_STORE_BACKEND" not in os.environ:
        return LocalFileStore(store_dir)
    return get_default_store()


def _policy_from_snapshot(snapshot: PolicySnapshot) -> Policy:
    if snapshot.metadata.get("policy_kind") == "contextual_softmax":
        return ContextualSoftmaxPolicy.from_snapshot(snapshot)
    return SoftmaxPolicy.from_snapshot(snapshot)


def _load_context(raw: str) -> dict[str, Any]:
    try:
        if raw.startswith("@"):
            with open(raw[1:], "r", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--context must be valid JSON or @path: {exc}") from exc

    if isinstance(value, list):
        return {"phi": value}
    if not isinstance(value, dict):
        raise TypeError("--context must contain a JSON object or feature-vector array")
    return value


def _redact_json(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(key): _redact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    return value


def _resolve_task_policy(
    store: LearningStore,
    agent_id: str,
    task_id: str | None,
) -> tuple[str, PolicySnapshot]:
    if task_id is None:
        tasks = store.list_agent_tasks(agent_id)
        if not tasks:
            raise ValueError(
                f"No task policy found for agent_id={agent_id!r}. "
                "Run `agent-learn task-policy-init` first."
            )
        if len(tasks) > 1:
            raise ValueError(
                f"Agent {agent_id!r} has multiple tasks; pass --task-id to select one."
            )
        task_id = tasks[0].id

    snapshot = store.get_latest_policy(agent_id, task_id=task_id)
    if snapshot is None:
        raise ValueError(
            f"No policy found for agent_id={agent_id!r}, task_id={task_id!r}. "
            "Run `agent-learn task-policy-init` first."
        )
    return task_id, snapshot


def _cmd_train(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    try:
        task_id, snapshot = _resolve_task_policy(store, args.agent_id, args.task_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    min_episodes = (
        args.min_episodes
        if args.min_episodes is not None
        else LearnerConfig().min_train_episodes
    )
    if min_episodes < 1:
        print(
            "--min-episodes and AGENT_LEARNING_MIN_TRAIN_EPISODES must be at least 1",
            file=sys.stderr,
        )
        return 2
    completed_episodes = store.count_completed_episodes(
        args.agent_id,
        task_id=task_id,
    )
    if completed_episodes < min_episodes:
        print(
            f"Agent {args.agent_id!r} task {task_id!r} has "
            f"{completed_episodes} completed episodes; "
            f"at least {min_episodes} are required to train.",
            file=sys.stderr,
        )
        return 2

    policy = _policy_from_snapshot(snapshot)
    runner = LearningRunner(store=store, policy=policy)
    run = runner.run_offline_batch(
        args.agent_id,
        episode_limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        score_missing=not args.skip_scoring,
        completed_only=True,
        task_id=task_id,
    )
    print(json.dumps(run.to_dict(), indent=2))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    runner = LearningRunner(store=store)
    episodes = store.query_episodes(
        args.agent_id,
        limit=args.limit,
        task_id=args.task_id,
    )
    scored = 0
    for episode in episodes:
        existing = store.get_rewards_for_episode(episode.id, args.agent_id)
        if existing:
            continue
        runner.score_and_record(episode)
        scored += 1
    print(json.dumps({"episodes_seen": len(episodes), "newly_scored": scored}, indent=2))
    return 0


def _cmd_agents_list(args: argparse.Namespace) -> int:
    agents = _get_cli_store(args).list_agents()
    print(json.dumps([agent.to_dict() for agent in agents], indent=2))
    return 0


def _cmd_agents_episodes_count(args: argparse.Namespace) -> int:
    count = _get_cli_store(args).count_completed_episodes(args.agent_id)
    print(json.dumps(count))
    return 0


def _cmd_agents_episodes_list(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    episodes = store.query_episodes(
        args.agent_id,
        limit=args.limit,
        task_id=args.task_id,
        completed_only=True,
    )
    review = []
    for episode in episodes:
        metrics = store.get_metric_results(episode.id, args.agent_id)
        rewards = store.get_rewards_for_episode(episode.id, args.agent_id)
        aggregate_rewards = [
            reward for reward in rewards if reward.source.value == "aggregate"
        ]
        review.append(
            {
                "id": episode.id,
                "task_id": episode.task_id,
                "intent_summary": episode.user_input,
                "chosen_action": episode.action_id,
                "score_breakdown": [metric.to_dict() for metric in metrics],
                "final_reward": (
                    aggregate_rewards[-1].value if aggregate_rewards else None
                ),
                "execution_result": episode.assistant_output,
                "status": episode.metadata.get("status"),
                "policy_id": episode.policy_id,
                "policy_version": episode.policy_version,
                "created_at": episode.created_at,
            }
        )
    print(json.dumps(review, indent=2))
    return 0


def _cmd_agent_tasks_list(args: argparse.Namespace) -> int:
    tasks = _get_cli_store(args).list_agent_tasks(args.agent_id)
    print(json.dumps([task.to_dict() for task in tasks], indent=2))
    return 0


def _cmd_show_task_policy(args: argparse.Namespace) -> int:
    if args.history < 1:
        print("--history must be at least 1", file=sys.stderr)
        return 2
    snapshots = _get_cli_store(args).list_policies(
        args.agent_id,
        task_id=args.task_id,
        limit=args.history,
    )
    if not snapshots:
        print(
            f"No policy found for agent_id={args.agent_id!r}, "
            f"task_id={args.task_id!r}.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps([snapshot.to_dict() for snapshot in snapshots], indent=2))
    return 0


def _cmd_init_task_policy(args: argparse.Namespace) -> int:
    if not args.agent_id.strip() or not args.task_id.strip():
        print("--agent-id and --task-id must not be empty", file=sys.stderr)
        return 2
    with open(args.actions, "r", encoding="utf-8") as f:
        action_payloads = json.load(f)
    if not isinstance(action_payloads, list) or not action_payloads:
        print("--actions file must contain a non-empty JSON list", file=sys.stderr)
        return 2
    actions = [Action.from_dict(item) for item in action_payloads]
    policy = SoftmaxPolicy.from_actions(actions, agent_id=args.agent_id)
    store = _get_cli_store(args)
    snapshot = policy.snapshot()
    current = store.get_latest_policy(args.agent_id, task_id=args.task_id)
    snapshot.task_id = args.task_id
    snapshot.version = current.version + 1 if current is not None else 0

    existing_agents = {
        agent.id: agent.name for agent in store.list_agents()
    }
    snapshot.metadata["agent_name"] = (
        args.agent_name
        or existing_agents.get(args.agent_id)
        or args.agent_id
    )
    snapshot.metadata["task_name"] = (
        args.task_name
        or (current.metadata.get("task_name") if current is not None else None)
        or args.task_id
    )
    store.store_policy(snapshot)
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


def _cmd_task_intent(args: argparse.Namespace) -> int:
    if not args.intent.strip():
        print("--intent must not be empty", file=sys.stderr)
        return 2
    if not args.task_id.strip():
        print("--task-id must not be empty", file=sys.stderr)
        return 2

    try:
        context = _redact_json(_load_context(args.context))
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    store = _get_cli_store(args)
    snapshot = store.get_latest_policy(args.agent_id, task_id=args.task_id)
    if snapshot is None:
        print(
            f"No policy found for agent_id={args.agent_id!r}, "
            f"task_id={args.task_id!r}. "
            "Run `agent-learn task-policy-init` first.",
            file=sys.stderr,
        )
        return 2

    if args.episode_id and store.get_episode(args.episode_id, args.agent_id) is not None:
        print(
            f"Episode {args.episode_id!r} already exists for agent_id={args.agent_id!r}.",
            file=sys.stderr,
        )
        return 2

    policy = _policy_from_snapshot(snapshot)
    state: Any | None = None
    if isinstance(policy, ContextualSoftmaxPolicy):
        state = context.get("phi")
        if state is None:
            print(
                "Contextual policies require --context with a 'phi' feature vector.",
                file=sys.stderr,
            )
            return 2

    try:
        decision = policy.choose(state)
    except (TypeError, ValueError) as exc:
        print(f"Unable to choose an action: {exc}", file=sys.stderr)
        return 2

    probabilities = {
        action.id: probability
        for action, probability in zip(snapshot.actions, decision.probabilities)
    }
    episode = Episode(
        agent_id=args.agent_id,
        task_id=args.task_id,
        user_input=redact(args.intent) or "",
        policy_id=snapshot.id,
        policy_version=snapshot.version,
        action_id=decision.action.id,
        action_logprob=decision.logprob,
        context_features=context,
        metadata={
            "status": "in_progress",
            "decision": {
                "action": _redact_json(decision.action.to_dict()),
                "probabilities": probabilities,
            },
        },
    )
    if args.episode_id:
        episode.id = args.episode_id
    store.store_episode(episode)

    print(
        json.dumps(
            {
                "episode_id": episode.id,
                "agent_id": episode.agent_id,
                "task_id": episode.task_id,
                "policy_id": episode.policy_id,
                "policy_version": episode.policy_version,
                "action_id": decision.action.id,
                "action": decision.action.to_dict(),
                "action_logprob": decision.logprob,
                "probabilities": probabilities,
            },
            indent=2,
        )
    )
    return 0


def _cmd_task_complete(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    episode = store.get_episode(args.episode_id, args.agent_id)
    if episode is None:
        print(
            f"No episode found for episode_id={args.episode_id!r} "
            f"and agent_id={args.agent_id!r}.",
            file=sys.stderr,
        )
        return 2

    completed_at = datetime.now(timezone.utc)
    episode.assistant_output = redact(args.output) or ""
    episode.metadata["status"] = "completed"
    episode.metadata["completed_at"] = completed_at.isoformat()
    try:
        started_at = datetime.fromisoformat(episode.created_at)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        episode.request_latency_ms = max(
            0,
            int((completed_at - started_at.astimezone(timezone.utc)).total_seconds() * 1000),
        )
    except ValueError:
        pass
    store.store_episode(episode)

    print(
        json.dumps(
            {
                "episode_id": episode.id,
                "agent_id": episode.agent_id,
                "task_id": episode.task_id,
                "status": "completed",
                "episode": episode.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "train": _cmd_train,
        "score": _cmd_score,
        "agents-list": _cmd_agents_list,
        "agents-episodes-count": _cmd_agents_episodes_count,
        "agents-episodes-list": _cmd_agents_episodes_list,
        "agent-tasks-list": _cmd_agent_tasks_list,
        "task-policy": _cmd_show_task_policy,
        "task-policy-init": _cmd_init_task_policy,
        "task-intent": _cmd_task_intent,
        "task-complete": _cmd_task_complete,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
