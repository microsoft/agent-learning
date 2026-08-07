---
name: agent-learn
description: "Use the agent-learn SDK to keep a structured learning loop for Scout tasks. For every task Scout attempts through chat, animation, or another skill, record the task identity, user intent, chosen action, execution result, and final outcome so repeated tasks improve over time. Use the local file store for lightweight review and iteration, and treat the SDK as a policy-learning system rather than a passive logger."
---

# Agent Learn

Use the following behavior whenever this skill is invoked.

Use one durable local store for capture, review, and training:

```text
AGENT_LEARNING_STORE_BACKEND=local
AGENT_LEARNING_LOCAL_STORE_DIR=./data/agent-learning/store
```

Do not use the default in-memory backend for work that spans multiple processes.

## 0. Treat repeated tasks as learning opportunities

If Scout repeats a task, do not treat it as a duplicate. Record it as another episode for the same stable task ID.

## 1. Treat every task as a policy-learning episode

Before starting a task, determine its intent, action class, and expected completion criteria. Resolve its task ID by semantically comparing the requested work with the task names returned by:

```shell
agent-learn tasks-list <agent_id>
```

Do not invent a task ID when multiple task names are plausible. Ask for clarification.

For every task, store or emit:

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

The SDK learns from discrete action choices and scored episodes, not raw logs.

## 2. Initialize a task policy when no policy exists

Inspect the active policy first:

```shell
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```

If the task has no active policy and no prior episode history, initialize it before recording the first episode:

```shell
agent-learn task-policy-init --agent-id <agent_id> --task-id <task_id> --actions ./actions.json
```

The actions file must contain a non-empty JSON list of explicit actions. If it is missing or incomplete, do not guess. Ask for the missing action definitions or create a minimal list from confirmed choices first.

The store may retain unlimited policy snapshot JSON files. Only one policy is active at a time for a given `(agent_id, task_id)`.

## 3. Capture intent before execution

Before Scout performs the action, start an episode through the SDK capture flow and record:

- what the user asked Scout to do;
- what Scout believes the task means;
- the task ID and task name;
- the action Scout selected;
- what successful completion looks like.

For a zero-shot task or a task with no prior history, write an explicit intent summary. Do not record a vague placeholder.

## 4. Record completion and adherence after execution

After Scout completes or attempts the task, capture:

- what actually happened;
- whether execution completed, failed, or was partial;
- whether the result adhered to the user's request;
- what follow-up remains.

Evaluate and persist all three core scoring signals:

- intent resolution;
- task adherence;
- task completion.

Persist the score breakdown and final aggregate reward with the episode.

## 5. Do not use the SDK as a black box

Every episode must include a clear task description, action choice, expected outcome, and post-execution result summary. If any value is missing, pause and obtain the missing context before persisting a full episode.

A policy snapshot is useful for debugging action preferences. It does not replace explicit intent capture or completion scoring.

## 6. Use a consistent action taxonomy

Use only these `action_type` values for Scout tasks:

- `chat`
- `animation`

Record each action result as JSON-compatible fields:

```json
{
  "action_name": "<action name>",
  "target": "<target>",
  "input_summary": "<input summary>",
  "execution_status": "<status>",
  "result_summary": "<result summary>"
}
```

## Execution sequence

1. Determine intent and resolve the agent task.
2. Inspect or initialize the task policy.
3. Choose an action from that policy's action space.
4. Capture the episode start.
5. Execute the action.
6. Capture the episode end.
7. Score intent resolution, task adherence, and task completion.
8. Persist the episode, scores, and final reward locally.