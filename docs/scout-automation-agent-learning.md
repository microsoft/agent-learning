---
title: Scout agent-learning automation
description: Discover each Scout agent task, verify its active policy and scored episodes, run a bounded task-scoped training batch, and confirm that a new policy snapshot was activated
author: Microsoft
ms.date: 2026-08-08
ms.topic: how-to
---

# Scout agent-learning automation

Run this automation periodically to improve repeated tasks from their recorded
outcomes. Set an automation-owned minimum number of new full episodes before
training; the recommended default is `5`. The CLI accepts a batch limit from `1`
through `500` but does not enforce the automation's minimum threshold.

Configure every automation step to use the same durable local store:

```text
AGENT_LEARNING_STORE_BACKEND=local
AGENT_LEARNING_LOCAL_STORE_DIR=./data/agent-learning/store
```

The environment variables must remain the same across capture, review, and training processes.

## 1. Discover agents

```shell
agent-learn list
```

The command returns the `id` and `name` of each agent. Run the remaining steps once for each returned agent.

## 2. Discover the agent's tasks

```shell
agent-learn tasks-list <agent_id>
```

Policies belong to an agent task. Run the remaining steps separately for every
returned task ID. Task-scoped runs make the episode limit and resulting policy
update unambiguous.

## 3. Verify the active task policy

```shell
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```

Continue only when `current_policy` exists and its action list is non-empty. If
the command reports no active policy, skip the task with reason
`no active policy`; episode-capture setup must initialize it with
`task-policy-init` before automation can train it.

## 4. Select and count new full episodes

```shell
agent-learn task-episodes-count <agent_id> --task-id <task_id>
```

The command prints the task's total number of full episodes. To avoid repeatedly
training on the same records, maintain a checkpoint from the previous successful
batch and select only episodes after that checkpoint. Use the episode
`created_at` values returned by `task-episodes-list` to count new full episodes
inside the intended window. Continue only when that new count meets the
automation's configured minimum.

Choose one fixed `end_date` at the start of the batch. After successful training,
advance the checkpoint to that cutoff. Use ISO 8601 timestamps for date filters.

## 5. Inspect the selected episodes

```shell
agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 500
```

Before training, print and compare the episodes. Review:

- differences in intent handling;
- the recorded intent summary;
- the chosen `action_id` and `metadata.correct_action_id`;
- the intent resolution, task adherence, and task completion score breakdown;
- the final aggregate reward;
- the execution status and result summary;
- differences in task completion quality.

Compare repeated episodes for the same task. Repetition is the learning signal: recorded outcomes should make recurring tasks more reliable as their policies are updated.

## 6. Ensure the selected episodes have rewards

`task-episode-register` persists episodes but does not score them. If any
selected episode has `final_reward: null`, either let `train` score missing
episodes or score the task explicitly first:

```shell
agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 500
```

Use a configured score backend. Re-list the episodes and verify that every
episode intended for learning has a non-null aggregate reward.

## 7. Train one task policy

The complete command contract is:

```shell
agent-learn train --agent-id <agent_id> [--task-id <task_id>] [--limit <1-500>] [--start-date <date>] [--end-date <date>] [--skip-scoring]
```

For periodic automation, prefer a task-scoped, checkpointed invocation:

```shell
agent-learn train --agent-id <agent_id> --task-id <task_id> --limit 500 --start-date <start_date> --end-date <end_date>
```

By default, `train` scores selected episodes that do not yet have rewards and
then updates the active policy. Add `--skip-scoring` only when every selected
episode already has a persisted aggregate reward:

```shell
agent-learn train --agent-id <agent_id> --task-id <task_id> --limit 500 --start-date <start_date> --end-date <end_date> --skip-scoring
```

Without `--task-id`, the CLI selects one agent-wide batch up to `--limit` and
distributes those selected episodes across tasks. That is supported, but a busy
task can consume most of the shared limit; task-scoped automation is more
predictable.

Inspect the JSON result. A successful update appears in `runs`. A task appears
in `skipped` when it has no active policy or no episodes in the selected batch.
The command exits with code `2` when no task policy was trained. Do not advance
the checkpoint unless the expected task has a successful run and
`metrics.episodes_used` is greater than zero.

## 8. Inspect the updated task policy

For every task returned by `tasks-list`, run:

```shell
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```

The output includes the active policy, the previous snapshot, and their
differences. Verify that the policy version increased, `episodes_seen` advanced,
and the action logits or probabilities changed in a direction supported by the
recorded rewards. Only after these checks should the automation persist its new
episode checkpoint.

The store can retain any number of policy snapshot JSON files, but exactly one policy is active for each agent task. Use policy snapshots for learning-loop debugging. They do not replace explicit intent capture or intent resolution, task adherence, and task completion scoring.