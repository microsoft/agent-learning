"""Command-line interface for the agent-learning SDK.

Examples::

    # Run an offline learning batch over the last 500 episodes
    agent-learn train --agent-id dq --limit 500

    # Score (only) recent episodes without updating the policy
    agent-learn score --agent-id dq --limit 100

    # Inspect the current policy
    agent-learn policy --agent-id dq

The CLI is intentionally thin - it loads default components from
environment variables and delegates to :class:`LearningRunner`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .policy.softmax_bandit import SoftmaxPolicy
from .storage.cosmos import get_default_store
from .training.runner import LearningRunner
from .types import Action


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-learn", description="Native RL CLI for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run one offline learning batch.")
    train.add_argument("--agent-id", required=True)
    train.add_argument("--limit", type=int, default=200)
    train.add_argument("--start-date")
    train.add_argument("--end-date")
    train.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring episodes that have no rewards yet.",
    )

    score = sub.add_parser("score", help="Score episodes but skip the policy update.")
    score.add_argument("--agent-id", required=True)
    score.add_argument("--limit", type=int, default=100)

    show = sub.add_parser("policy", help="Print the latest policy snapshot for an agent.")
    show.add_argument("--agent-id", required=True)

    init = sub.add_parser("init-policy", help="Create the initial policy snapshot from a JSON file.")
    init.add_argument("--agent-id", required=True)
    init.add_argument(
        "--actions",
        required=True,
        help="Path to a JSON file containing a list of {id, description, parameters} objects.",
    )

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    store = get_default_store()
    snapshot = store.get_latest_policy(args.agent_id)
    if snapshot is None:
        print(
            f"No policy found for agent_id={args.agent_id!r}. "
            "Run `agent-learn init-policy` first.",
            file=sys.stderr,
        )
        return 2

    policy = SoftmaxPolicy.from_snapshot(snapshot)
    runner = LearningRunner(store=store, policy=policy)
    run = runner.run_offline_batch(
        args.agent_id,
        episode_limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        score_missing=not args.skip_scoring,
    )
    print(json.dumps(run.to_dict(), indent=2))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    store = get_default_store()
    runner = LearningRunner(store=store)
    episodes = store.query_episodes(args.agent_id, limit=args.limit)
    scored = 0
    for episode in episodes:
        existing = store.get_rewards_for_episode(episode.id, args.agent_id)
        if existing:
            continue
        runner.score_and_record(episode)
        scored += 1
    print(json.dumps({"episodes_seen": len(episodes), "newly_scored": scored}, indent=2))
    return 0


def _cmd_show_policy(args: argparse.Namespace) -> int:
    store = get_default_store()
    snapshot = store.get_latest_policy(args.agent_id)
    if snapshot is None:
        print(f"No policy found for agent_id={args.agent_id!r}.", file=sys.stderr)
        return 2
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


def _cmd_init_policy(args: argparse.Namespace) -> int:
    with open(args.actions, "r", encoding="utf-8") as f:
        action_payloads = json.load(f)
    if not isinstance(action_payloads, list) or not action_payloads:
        print("--actions file must contain a non-empty JSON list", file=sys.stderr)
        return 2
    actions = [Action.from_dict(item) for item in action_payloads]
    policy = SoftmaxPolicy.from_actions(actions, agent_id=args.agent_id)
    store = get_default_store()
    store.store_policy(policy.snapshot())
    print(json.dumps(policy.snapshot().to_dict(), indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "train": _cmd_train,
        "score": _cmd_score,
        "policy": _cmd_show_policy,
        "init-policy": _cmd_init_policy,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
