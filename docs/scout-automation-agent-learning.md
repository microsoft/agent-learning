---
title: Scout agent-learning automation
description: Discover each Scout agent task, verify its active policy and scored episodes, run a bounded task-scoped training batch, and confirm that a new policy snapshot was activated
author: Microsoft
ms.date: 2026-08-08
ms.topic: how-to
instructions: |
  # Scout agent-learning automation

  Run this automation periodically to improve repeated tasks from their recorded
  outcomes. Set an automation-owned minimum number of new full episodes before
  training; the recommended default is `5`. The CLI accepts a batch limit from `1`
  through `500` but does not enforce the automation's minimum threshold.

  ## 0. Establish the shared durable store

  Before invoking any `agent-learn` command, establish the same absolute store
  used by episode capture. On Windows PowerShell, run these assignments in every
  shell process that invokes the CLI:

  ```powershell
  $env:AGENT_LEARNING_STORE_BACKEND = "local"
  $env:AGENT_LEARNING_LOCAL_STORE_DIR = Join-Path $env:LOCALAPPDATA "agent-learning\store"
  ```

  The resolved default Windows location is
  `%LOCALAPPDATA%\agent-learning\store`. Never substitute a path relative to the
  automation's working directory. The environment variables and absolute path
  must be identical across capture, review, and training processes.

  ## 1. Discover agents

  ```shell
  agent-learn list
  ```

    The command returns the `id` and `name` of each agent. Run the remaining steps once for each returned agent.

    If the command returns `[]`, stop. Do not report that training succeeded and do
    not advance checkpoints. Diagnose capture first:

    1. Confirm the skill and automation resolve the same absolute
      `AGENT_LEARNING_LOCAL_STORE_DIR`.
    2. Confirm `AGENT_LEARNING_STORE_BACKEND` is `local`, not the default `memory`.
    3. Confirm the live Scout skill was invoked for the preceding request and ran
      `task-policy-init` or `task-episode-register`.
    4. Confirm the store contains `policies/` or `episodes/` JSON files.

    Training only consumes existing records; it cannot reconstruct a user request
    that the capture skill never registered.

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

  Read the checkpoint for the current `(agent_id, task_id)` only. Choose one
  fixed `end_date` before inspecting any task. Then ask the CLI for the exact
  window that `train` will consume:

  ```shell
  agent-learn task-episodes-count <agent_id> --task-id <task_id> --start-date <checkpoint_end_date> --end-date <end_date>
  ```

  Omit `--start-date` when that task has no checkpoint. The returned number is
  the authoritative eligible count. Continue only when it meets the configured
  minimum.

  Never calculate `total episodes - current_policy.episodes_seen`.
  `episodes_seen` counts training usages, not unique episodes, so retraining or
  rescoring a batch can make it larger than the number of stored records. Never
  reuse one task's checkpoint as another task's `start_date`. Keep task-local
  `start_date`, count, and result variables through the whole loop.

  After successful training, advance only that task's checkpoint to the fixed
  cutoff. Use ISO 8601 timestamps for all date filters.

  ## 5. Inspect the selected episodes

  ```shell
  agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 500 --start-date <checkpoint_end_date> --end-date <end_date>
  ```

  Omit `--start-date` for a task without a checkpoint. The list length must equal
  the windowed count when the count is at most `500`. If they differ, stop and do
  not train or advance the checkpoint.

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

  `task-episode-register` persists episodes but does not score them. The CLI uses
  on-device stdlib scoring by default with no endpoint or environment variables.
  If any selected episode has `final_reward: null`, `final_reward: 0.0` backed
  only by skipped metrics, or fewer than three completed core metrics, either let
  `train` score it or score the task explicitly first:

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 500
  ```

  Re-list the episodes and verify that every episode intended for learning has
  completed intent-resolution, task-adherence, and task-completion metrics plus
  a non-null aggregate reward. A configured Azure scorer is optional.

  ## 7. Train one task policy

  The complete command contract is:

  ```shell
  agent-learn train --agent-id <agent_id> [--task-id <task_id>] [--limit <1-500>] [--min-episodes <1-500>] [--start-date <date>] [--end-date <date>] [--skip-scoring]
  ```

  For periodic automation, prefer a task-scoped, checkpointed invocation:

  ```shell
  agent-learn train --agent-id <agent_id> --task-id <task_id> --limit 500 --min-episodes 5 --start-date <checkpoint_end_date> --end-date <end_date>
  ```

  Omit `--start-date` when the task has no checkpoint. `--min-episodes 5`
  independently prevents the CLI from updating a policy when the selected window
  is smaller than the threshold, even if the automation counted incorrectly.

  By default, `train` locally scores selected episodes that do not yet have a
  usable reward and then updates the active policy. Add `--skip-scoring` only
  when every selected episode already has three completed core metrics and a
  persisted aggregate reward:

  ```shell
  agent-learn train --agent-id <agent_id> --task-id <task_id> --limit 500 --min-episodes 5 --start-date <checkpoint_end_date> --end-date <end_date> --skip-scoring
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
---