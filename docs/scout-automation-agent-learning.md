---
title: Scout agent-learning automation
description: Review complete episodes and train each agent task policy from recorded outcomes
author: Microsoft
ms.date: 2026-08-07
ms.topic: how-to
---

# Scout agent-learning automation

Run this automation periodically to improve repeated tasks from their recorded outcomes. Set the minimum number of full episodes before training; the default is `5`. A training batch is capped at `500` episodes.

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

Policies belong to an agent task. Use each returned task ID when inspecting policy state.

## 3. Count full episodes

```shell
agent-learn task-episodes-count <agent_id>
```

The command prints the number of full episodes. Continue only when the count is greater than or equal to the configured minimum.

## 4. Inspect the episodes

```shell
agent-learn task-episodes-list <agent_id> --limit 500
```

Before training, print and compare the episodes. Review:

- differences in intent handling;
- the recorded intent summary;
- the chosen action;
- the intent resolution, task adherence, and task completion score breakdown;
- the final aggregate reward;
- the execution status and result summary;
- differences in task completion quality.

Compare repeated episodes for the same task. Repetition is the learning signal: recorded outcomes should make recurring tasks more reliable as their policies are updated.

## 5. Train the agent's task policies

```shell
agent-learn train --agent-id <agent_id> --limit 500
```

Training runs at the agent level and updates each task independently from that task's episodes and active policy. The CLI rejects limits greater than `500`.

## 6. Inspect each updated task policy

For every task returned by `tasks-list`, run:

```shell
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```

The output includes the active policy, the previous snapshot, and their differences. Review changes in action logits and action probabilities to understand which actions the task now favors.

The store can retain any number of policy snapshot JSON files, but exactly one policy is active for each agent task. Use policy snapshots for learning-loop debugging. They do not replace explicit intent capture or intent resolution, task adherence, and task completion scoring.