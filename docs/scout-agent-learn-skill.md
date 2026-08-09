---
name: agent-learn
description: "Use agent-learn whenever Scout executes a repeated task: resolve or initialize its task policy, choose an action from that policy, capture intent and completion, register the full episode with the chosen and correct action IDs, and persist scores and reward so later automation can train the policy."
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
- `action_id`
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

The actions file must contain a non-empty JSON list. Every action needs a stable,
unique `id`; its optional `description` and `parameters` explain how Scout should
execute it:

```json
[
  {
    "id": "answer_in_chat",
    "description": "Answer directly in chat",
    "parameters": {}
  },
  {
    "id": "create_animation",
    "description": "Create and return an animation",
    "parameters": {}
  }
]
```

If the action space is missing or incomplete, do not guess. Ask for the missing
definitions or create a minimal list from confirmed choices. Treat an
already-active policy as authoritative; `task-policy-init` rejects a second
initialization for the same agent task.

The store may retain unlimited policy snapshot JSON files. Only one policy is active at a time for a given `(agent_id, task_id)`.

## 3. Choose and record the policy action

Read `current_policy` from `task-policy`. Choose only an `action_id` listed in
`current_policy.actions`, using the current action probabilities. The CLI exposes
the snapshot and probabilities; an in-process agent can reconstruct
`SoftmaxPolicy` from that snapshot and call `choose()`.

Before execution, retain these decision fields for the episode:

- `policy_id`: `current_policy.id`;
- `policy_version`: `current_policy.version`;
- `action_id`: the action actually selected;
- `action_logprob`: the selected action's behavior log probability, when the
  sampler provides it.

Do not choose an action outside the initialized action space.

## 4. Capture intent before execution

Before Scout performs the action, start an episode through the SDK capture flow and record:

- what the user asked Scout to do;
- what Scout believes the task means;
- the task ID and task name;
- the action Scout selected;
- what successful completion looks like.

For a zero-shot task or a task with no prior history, write an explicit intent summary. Do not record a vague placeholder.

## 5. Record completion and the correct action

After Scout completes or attempts the task, capture:

- what actually happened;
- whether execution completed, failed, or was partial;
- whether the result adhered to the user's request;
- what follow-up remains.

After the outcome can be evaluated, determine the independently confirmed
`correct_action_id`. It must be one of the initialized policy action IDs. Store
it as `metadata.correct_action_id`, and store whether the selected action
completed the task as `metadata.task_completed`.

`action_id` and `correct_action_id` have different meanings:

- `action_id` is what the policy selected and executed;
- `correct_action_id` is the action that the rubric, evaluator, or observed
  outcome says should have been selected.

They may differ on an unsuccessful episode. Do not label the selected action as
correct merely because Scout chose it. If correctness cannot be established,
omit `correct_action_id` rather than guessing and explain the uncertainty in
`result_summary`.

## 6. Register the full episode

After execution, write one JSON object such as `./episode.json`. The CLI supplies
`agent_id`, `task_id`, and a generated episode `id` when they are omitted:

```json
{
  "agent_name": "Scout",
  "task_name": "<stable task name>",
  "user_input": "<original user request>",
  "assistant_output": "<final response or execution output>",
  "intent_summary": "<specific intent>",
  "action_type": "chat",
  "action_id": "<selected policy action id>",
  "action_name": "<selected action description>",
  "target": "<task target>",
  "input_summary": "<concise input summary>",
  "expected_outcome": "<observable completion criteria>",
  "execution_status": "completed",
  "result_summary": "<what happened and what remains>",
  "policy_id": "<current policy id>",
  "policy_version": 0,
  "action_logprob": -0.6931471805599453,
  "metadata": {
    "correct_action_id": "<confirmed correct action id>",
    "task_completed": true
  }
}
```

For failed or partial work, use the corresponding `execution_status` and an
accurate `result_summary`; do not omit the outcome. Register the episode through
the same durable store used for policy inspection:

```shell
agent-learn task-episode-register --agent-id <agent_id> --task-id <task_id> --episode ./episode.json
```

If `agent_id` or `task_id` is present in the JSON, it must exactly match the
command arguments. Keep the returned episode `id` for verification.

## 7. Score and verify the registered episode

`task-episode-register` stores the episode but does not score it. With the score
backend configured, persist missing metric results and aggregate rewards:

```shell
agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 100
```

The score command skips episodes that already have rewards, so it is safe when
the SDK capture flow has already written deterministic or external scores.
Inspect the registered task afterward:

```shell
agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 100
```

Confirm that the episode contains its intent, chosen action, completion result,
score breakdown, and non-null `final_reward`. An episode without an aggregate
reward cannot contribute to a policy update when training uses
`--skip-scoring`.

Evaluate and persist all three core scoring signals:

Evaluate and persist all three core scoring signals:

- intent resolution;
- task adherence;
- task completion.

Persist the score breakdown and final aggregate reward with the episode.

## 8. Do not use the SDK as a black box

Every episode must include a clear task description, action choice, expected outcome, and post-execution result summary. If any value is missing, pause and obtain the missing context before persisting a full episode.

A policy snapshot is useful for debugging action preferences. It does not replace explicit intent capture or completion scoring.

## 9. Use a consistent action taxonomy

Use only these `action_type` values for Scout tasks:

- `chat`
- `animation`

Record each action result as JSON-compatible fields:

```json
{
  "action_id": "<selected policy action id>",
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
3. Choose an action from that policy's action space and retain the policy
  decision metadata.
4. Capture intent and expected outcome before execution.
5. Execute the selected action.
6. Record completion and independently determine `correct_action_id` when the
  outcome supports one.
7. Register the full episode with `task-episode-register`.
8. Score intent resolution, task adherence, and task completion.
9. Verify that the episode has a final aggregate reward.