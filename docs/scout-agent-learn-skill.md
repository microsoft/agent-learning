---
name: agent-learn
description: >
  Use the agent-learn SDK to keep a structured learning loop for Scout tasks.
  For every task Scout attempts, record the task, intent, chosen action,
  execution result, and final outcome so repeated work can improve over time.
  Use the local file store for lightweight review and iteration. Do not treat
  the SDK as a passive logger.
---

## Instructions

Use the following behavior whenever this skill is invoked.

### 0. Treat repeated tasks as learning opportunities

Do not treat a repeated task as a duplicate. Record every attempt as another
episode for the same agent task.

### 1. Treat every task as a policy-learning episode

Before execution, determine the task's intent, action class, and expected
completion criteria. Store or emit:

- `agent_id`
- `task_id`
- `task_name`
- `intent_summary`
- `action_type`
- `action_name`
- `target`
- `input_summary`
- `expected_outcome`
- `execution_status`
- `result_summary`

Use `agent-learn agent-tasks-list <agent_id>` and semantically compare
`task_name` values before creating a task. If no existing task is a confident
match, create a new task policy instead of attaching the episode to an
unrelated task.

### 2. Initialize missing task policies

Policies belong to tasks under an agent. If a task has no policy, make its
action space explicit before recording the first episode:

```bash
agent-learn task-policy-init \
  --agent-id <agent_id> \
  --task-id <task_id> \
  --task-name "<task_name>" \
  --actions ./actions.json
```

If the actions file is missing or incomplete, do not guess. Ask for the
missing information or create a minimal, explicit action list first.

The store can retain any number of policy snapshot JSON files. The
highest-version snapshot is the only active policy for a given agent task.

### 3. Capture intent before execution

Record what the user asked for, what the request means, which action was
selected, and what success looks like. For zero-shot tasks or tasks without
history, use an explicit intent summary; a vague summary is not sufficient.

Start the episode and use the active task policy:

```bash
agent-learn task-intent \
  --agent-id <agent_id> \
  --task-id <task_id> \
  --intent "<intent_summary>" \
  --context '<structured_context>'
```

Preserve the returned `episode_id` for completion.

### 4. Record completion and adherence

After execution, capture what happened, whether the task completed, whether it
adhered to the request, and whether it failed, was partial, or needs follow-up:

```bash
agent-learn task-complete \
  --agent-id <agent_id> \
  --episode-id <episode_id> \
  --output "<result_summary>"
```

Evaluate and persist the three core judge signals:

- intent resolution;
- task adherence; and
- task completion.

```bash
agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 1
```

### 5. Do not use the SDK as a black box

Always provide a clear task description, action choice, expected outcome, and
post-execution result summary. If any item is missing, pause and ask for the
missing context before recording the episode.

### 6. Use a consistent action taxonomy

Use only these action types:

- `chat`
- `animation`

Record each action in JSON with:

```json
{
  "action_name": "<name>",
  "target": "<target>",
  "input_summary": "<summary>",
  "execution_status": "<status>",
  "result_summary": "<summary>"
}
```

## Recommended sequence

1. Determine intent.
2. Find the existing agent task, or initialize its policy.
3. Choose an action from the task policy.
4. Capture the episode start.
5. Execute the action.
6. Capture the episode end.
7. Score intent resolution, adherence, and completion.
8. Persist the episode locally.
