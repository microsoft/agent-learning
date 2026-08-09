---
name: agent-learn
description: "MANDATORY for every Scout user request, including first-time questions, one-off chat answers, and tool calls: use agent-learn to resolve or initialize a task policy, choose an action, register the full episode with chosen and correct action IDs, and persist its scores and reward. Do not wait for the user to mention learning or repeat a task."
instructions: |
  # Agent Learn

  Apply this workflow to every Scout user request, including the first occurrence
  of a task. It is cross-cutting capture behavior: do not wait for a task to be
  repeated, and do not skip it because another skill or tool answered the request.

  ## 0. Establish one durable store

  Before invoking any `agent-learn` command, establish the same absolute store
  used by the training automation. On Windows PowerShell, run these assignments
  in every shell process that invokes the CLI:

  ```powershell
  $env:AGENT_LEARNING_STORE_BACKEND = "local"
  $env:AGENT_LEARNING_LOCAL_STORE_DIR = Join-Path $env:LOCALAPPDATA "agent-learning\store"
  ```

  The resolved default Windows location is
  `%LOCALAPPDATA%\agent-learning\store`. Use that exact absolute directory for
  capture, review, and training. Do not use a path relative to Scout's working
  directory. Do not use the default in-memory backend: each CLI invocation is a
  separate process, so in-memory policies and episodes disappear before the next
  command.

  For a persistent configuration inherited by newly started Scout processes,
  configure the user environment once and restart Scout:

  ```powershell
  [Environment]::SetEnvironmentVariable("AGENT_LEARNING_STORE_BACKEND", "local", "User")
  [Environment]::SetEnvironmentVariable("AGENT_LEARNING_LOCAL_STORE_DIR", (Join-Path $env:LOCALAPPDATA "agent-learning\store"), "User")
  ```

  ## 1. Treat every task as a learning opportunity

  Register the first attempt and every repeated attempt as an episode under the
  same stable task ID. A successful answer from another tool is still an episode.
  For example, a question answered with `Get Context Usage` must be captured after
  the tool result is returned.

  ## 2. Resolve the task

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

  ## 3. Initialize a task policy when no policy exists

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

  ## 4. Choose and record the policy action

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

  ## 5. Capture intent before execution

  Before Scout performs the action, start an episode through the SDK capture flow and record:

  - what the user asked Scout to do;
  - what Scout believes the task means;
  - the task ID and task name;
  - the action Scout selected;
  - what successful completion looks like.

  For a zero-shot task or a task with no prior history, write an explicit intent summary. Do not record a vague placeholder.

  ## 6. Record completion and the correct action

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

  ## 7. Register the full episode

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

  ## 8. Score and verify the registered episode

  `task-episode-register` stores the episode but does not score it. Persist
  missing metric results and aggregate rewards with the zero-configuration local
  stdlib scorer:

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 100
  ```

  No scoring endpoint or environment variable is required. A configured Azure
  scorer remains an optional override. The score command skips episodes that
  already have usable rewards and automatically replaces prior evaluations whose
  metrics were all skipped.
  Inspect the registered task afterward:

  ```shell
  agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 100
  ```

  Confirm that the episode contains its intent, chosen action, completion result,
  score breakdown, and non-null `final_reward`. An episode without an aggregate
  reward cannot contribute to a policy update when training uses
  `--skip-scoring`.

  Evaluate and persist all three core scoring signals:

  - intent resolution;
  - task adherence;
  - task completion.

  Persist the score breakdown and final aggregate reward with the episode.

  ## 9. Do not use the SDK as a black box

  Every episode must include a clear task description, action choice, expected outcome, and post-execution result summary. If any value is missing, pause and obtain the missing context before persisting a full episode.

  A policy snapshot is useful for debugging action preferences. It does not replace explicit intent capture or completion scoring.

  ## 10. Use a consistent action taxonomy

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

    1. Establish the absolute durable local store in the command process.
    2. Determine intent and resolve the agent task.
    3. Inspect or initialize the task policy.
    4. Choose an action from that policy's action space and retain the policy
     decision metadata.
    5. Capture intent and expected outcome before execution.
    6. Execute the selected action, including any other skill or tool calls.
    7. Record completion and independently determine `correct_action_id` when the
     outcome supports one.
    8. Register the full episode with `task-episode-register` before replying is
      considered complete.
    9. Score intent resolution, task adherence, and task completion.
    10. Verify that the episode has a final aggregate reward.
---