---
title: Scout automation for agent learning
description: Use the agent-learn CLI to inspect completed episodes, train task policies, and review policy changes from a Scout automation.
author: Microsoft
ms.date: 2026-08-07
ms.topic: how-to
keywords:
  - scout
  - automation
  - agent learning
  - policy training
  - episodes
estimated_reading_time: 6
---

## Run the learning automation

Run this automation periodically against the same local store used to capture
task episodes. Repeated tasks are additional learning episodes, not duplicates.

### 1. List agents

```bash
agent-learn agents-list
```

The command returns each agent's `id` and `name`. Process every returned agent.

### 2. Count completed episodes

```bash
agent-learn agents-episodes-count <agent_id>
```

The command returns the number of full, completed episodes for the agent as an
integer. Do not train until the count is at least five. Five is the default
threshold; configure it with `AGENT_LEARNING_MIN_TRAIN_EPISODES` or the
`train --min-episodes` option.

List the agent's tasks before selecting a policy:

```bash
agent-learn agent-tasks-list <agent_id>
```

### 3. Review episodes and train

Before training, print the completed episodes for each task:

```bash
agent-learn agents-episodes-list <agent_id> --task-id <task_id> --limit 500
```

For each episode, inspect:

- differences in intent handling and the recorded `intent_summary`;
- the `chosen_action`;
- the `score_breakdown` for intent resolution, task adherence, and task
  completion;
- the `final_reward`;
- the `execution_result`; and
- differences in task-completion quality across repeated tasks.

This review is the core of the learning loop: repeated tasks should become more
reliable because the system records outcomes and updates the applicable policy.

Train with at most 500 completed episodes:

```bash
agent-learn train --agent-id <agent_id> --task-id <task_id> --limit 500
```

When an agent has exactly one task, `--task-id` can be omitted:

```bash
agent-learn train --agent-id <agent_id> --limit 500
```

Training is task-scoped. If a task has fewer than the configured minimum number
of completed episodes, leave its policy unchanged.

### 4. Inspect the updated policy

Print the active policy and its predecessor:

```bash
agent-learn task-policy \
  --agent-id <agent_id> \
  --task-id <task_id> \
  --history 2
```

The first snapshot is active. Compare it with the previous snapshot, including
action logits, baseline, episode count, and update count, to understand what
the task will now favor.

Policy snapshots are debugging aids. They do not replace explicit intent
capture, execution-result capture, or completion scoring.
