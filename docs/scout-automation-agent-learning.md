---
title: Scout delegated-decision policy training
description: Train only Scout policies explicitly marked as delegated decisions; exclude questions, reporting tasks, ordinary chat, and the agent-learning automation itself
author: Microsoft
ms.date: 2026-08-09
ms.topic: how-to
instructions: |
  # Scout delegated-decision policy training

  This automation trains reusable delegated decisions only. It is control-plane
  maintenance, not a user decision task.

  **Never create a TaskPolicy or episode for this automation run. Never train a
  `run-agent-learning-automation` policy.** Its status report is operational
  output and must not pass through the agent-learning capture skill.

  ## 0. Establish the shared store and cutoff

  ```powershell
  $env:AGENT_LEARNING_STORE_BACKEND = "local"
  $env:AGENT_LEARNING_LOCAL_STORE_DIR = Join-Path $env:LOCALAPPDATA "agent-learning\store"
  ```

  Choose one fixed UTC `end_date` before inspecting policies. Read and update the
  task-local checkpoint file only after verified successful training.

  ## 1. Discover decision policies only

  ```shell
  agent-learn list
  agent-learn tasks-list <agent_id> --decision-only
  ```

  The second command is authoritative. It returns only policies whose metadata
  has `policy_scope: delegated_decision`.

  Do not fall back to unfiltered `tasks-list`. In particular, ignore legacy or
  accidental policies for:

  - factual questions and informational queries;
  - model context-window reporting;
  - ordinary chat, summaries, or status reports;
  - `run-agent-learning-automation`;
  - scoring, review, or training operations.

  If no decision policies are returned, report that there are no delegated
  decision policies to train. Do not create one and do not write a checkpoint.

  ## 2. Verify policy intent

  For each returned task:

  ```shell
  agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
  ```

  Confirm `current_policy.metadata.policy_scope` is `delegated_decision`,
  `decision_context` describes a real reusable choice, and at least two actions
  map to executable delegates or strategies. Skip anything else.

  ## 3. Count the exact task-local window

  Read this task's checkpoint only. Omit `--start-date` when none exists:

  ```shell
  agent-learn task-episodes-count <agent_id> --task-id <task_id> --start-date <checkpoint_end_date> --end-date <end_date>
  agent-learn task-episodes-count <agent_id> --task-id <task_id> --include-incomplete --start-date <checkpoint_end_date> --end-date <end_date>
  agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 500 --start-date <checkpoint_end_date> --end-date <end_date>
  agent-learn task-episodes-list <agent_id> --task-id <task_id> --limit 500 --include-incomplete --start-date <checkpoint_end_date> --end-date <end_date>
  ```

  The default count and list are full, potentially trainable episodes. The
  `--include-incomplete` count and list are all attempts, including pending
  recommendations. Each count and its corresponding list length must agree when
  the count is at most `500`. Pending count is all attempts minus full episodes.

  Never calculate `total episodes - current_policy.episodes_seen`.
  `episodes_seen` counts training usages, not unique records. Never reuse another
  task's checkpoint.

  ## 4. Verify usable execution feedback

  Require at least five selected episodes. Every episode must:

  - reference this delegated decision policy and one of its actions;
  - contain user-visible execution results;
  - contain three completed core metrics;
  - contain a non-null aggregate reward;
  - include `metadata.correct_action_id` when correctness is independently known.

  Exclude unresolved comparisons and unexecuted recommendations. Repeating a
  prompt is not evidence by itself; each episode must represent an executed
  delegate, explicit user acceptance/rejection, or another observable outcome.

  Do not hide pending attempts in the status report. If a policy has five
  incomplete recommendations and no observed outcomes, report "0 trainable
  episodes; 5 pending feedback" rather than only "0 eligible episodes." Never
  train or advance a checkpoint for pending records.

  Rescore incomplete evaluations locally when needed:

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <task_id> --limit 500
  ```

  ## 5. Train with CLI-side eligibility enforcement

  ```shell
  agent-learn train --agent-id <agent_id> --task-id <task_id> --decision-only --limit 500 --min-episodes 5 --start-date <checkpoint_end_date> --end-date <end_date> --skip-scoring
  ```

  Omit `--start-date` for a task with no checkpoint. `--decision-only` prevents
  accidental question or automation policies from training. `--min-episodes 5`
  prevents an undersized update even if orchestration counted incorrectly.

  ## 6. Verify usefulness for the next execution

  Inspect the new policy:

  ```shell
  agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
  agent-learn task-policy-decide --agent-id <agent_id> --task-id <task_id> --greedy
  ```

  Require all of the following before advancing the checkpoint:

  - a successful run for the expected task;
  - `metrics.episodes_used >= 5`;
  - policy version and `episodes_seen` increased;
  - decision output contains `selected_action_feedback` and
    `historical_feedback` with recent execution scores and result summaries;
  - decision output contains `autonomy.criteria`, `mode`,
    `execute_without_confirmation`, `request_user_feedback`, and
    `observable_outcome_satisfies_feedback` plus `outcome_recording`.

  A probability change alone is not enough. The decision output must expose the
  quality feedback and autonomy gates that Scout will consume before the next
  delegated execution. Report the autonomy mode and unmet criteria, but never
  override the SDK assessment in automation.

  ## 7. Advance only the trained task checkpoint

  Persist the fixed `end_date`, run ID, policy ID/version, and episodes used for
  this `(agent_id, task_id)` only. Do not advance checkpoints for skipped tasks.

  ## Execution sequence

  1. Establish the store and fixed cutoff.
  2. Discover with `tasks-list --decision-only`.
  3. Verify decision metadata and action space.
  4. Count/list full episodes and all attempts in the exact task-local window.
  5. Verify or repair execution scores.
  6. Train with `--decision-only --min-episodes 5`.
  7. Verify policy update and next-execution feedback.
  8. Advance only that decision task's checkpoint.
  9. Return the operational report without capturing this automation as an
     agent-learning episode.
---