---
name: agent-learn
description: "Use when Scout must make a delegated decision among two or more explicit execution alternatives, or when the user accepts, rejects, or reports the outcome of a prior agent-learn recommendation. Examples include choosing a language model, skill, automation, tool, workflow, escalation path, or Azure workload. Do not use for factual questions, ordinary chat answers, summaries, reporting, or running agent-learning scoring/training automation."
instructions: |
  # Agent Learn: delegated decisions only

  Agent learning governs Scout's decisions about how work should be delegated or
  executed. It does not govern every user message.

  ## 0. Eligibility gate

  Create or use a TaskPolicy only when all of these are true:

  1. Scout must choose among at least two explicit executable alternatives.
  2. The alternatives are stable enough to reuse across future requests.
  3. The selected alternative can affect user-visible quality, correctness,
     latency, cost, safety, or completion.
  4. The outcome can later identify whether the choice was correct or useful.

  Eligible examples:

  - choose an Azure workload for a stated use case;
  - choose a language model versus a specialized skill or automation;
  - recommend one of two named models or workloads when the concrete use case,
    optimization priority, and success criteria are known;
  - choose which tool, workflow, retrieval strategy, or escalation path to use;
  - choose among implementation approaches that Scout can actually execute.

  Ineligible examples:

  - answer a factual or informational question;
  - report model context-window size;
  - summarize, explain, or retrieve information when no delegation choice exists;
  - run agent-learning scoring, review, or training automation;
  - create a policy with only one possible action.

  A user question may provide context for an eligible delegation decision, but
  the TaskPolicy represents the decision, not the question. For example,
  `choose-answer-delegate` may choose between a live-data skill and a language
  model; do not create `answer-context-window-question` as a policy.

  A comparison such as "Should I use GPT-5.6 Sol or Terra for my use case?" is
  not yet a reusable decision when the use case and priority are absent. Ask for
  the workload plus the primary optimization target (for example correctness,
  latency, cost, long-context synthesis, or tool-use reliability). Do not infer
  an "apparent use case" and train on that guess. Once the context is supplied,
  the reusable decision may be `choose-model-for-code-agent`, with actions such
  as `use_gpt_5_6_sol` and `use_gpt_5_6_terra`.

  Advice alone is not trainable execution evidence. Preserve a supervised or
  audited recommendation as pending, but complete it only when Scout executes
  the selected delegate, the user accepts/rejects it, or another observable
  outcome can score the choice. Repeating the same unresolved comparison five
  times does not create five trainable decision episodes.

  If the gate fails, answer or execute normally. Do not initialize a policy, do
  not register an episode, and do not invoke the training loop.

  ## 1. Establish the durable store

  Use the same absolute store as decision training:

  ```powershell
  $env:AGENT_LEARNING_STORE_BACKEND = "local"
  $env:AGENT_LEARNING_LOCAL_STORE_DIR = Join-Path $env:LOCALAPPDATA "agent-learning\store"
  ```

  Do not use the in-memory backend across CLI processes.

  ## 2. Resolve a reusable delegated decision

  Discover only marked decision policies:

  ```shell
  agent-learn tasks-list <agent_id> --decision-only
  ```

  Reuse a task ID only when its `decision_context` and action space match the
  present decision. Otherwise create a stable decision ID such as
  `choose-azure-workload` or `choose-answer-delegate`.

  Resolve exactly one decision boundary for the request. Do not initialize
  overlapping policies for different interpretations of the same question. For
  example:

  - `choose-azure-pubsub-message-processor`, with Functions, Container Apps,
    and AKS actions, answers "which Azure compute workload should receive and
    process these messages?";
  - `choose-azure-pubsub-ingress`, with Event Grid, Service Bus, and Event Hubs
    actions, answers "which Azure messaging layer should buffer, route, or fan
    out these messages?"

  A request to "receive and process" messages is the processor decision unless
  the user explicitly asks for an intermediary messaging layer. Do not create
  both policies for one request. If related policies exist and the boundary is
  genuinely ambiguous, ask one clarifying question before selecting or creating
  a policy.

  ## 3. Initialize a new decision policy

  Define at least two non-empty, unique action IDs. Each action must map to a real
  underlying model, skill, automation, tool, workload, or execution strategy.

  ```json
  [
    {
      "id": "use_live_context_skill",
      "description": "Delegate to the live context-usage skill",
      "parameters": {"delegate": "get-context-usage"}
    },
    {
      "id": "use_language_model",
      "description": "Delegate to the selected language model",
      "parameters": {"delegate": "language-model"}
    }
  ]
  ```

  Define a complexity profile. Do not lower complexity by interpreting the
  wording of Scout's own answer. If the user or policy owner has not provided a
  profile, keep the SDK's standard default. Elevate from standard when the
  request clearly establishes high impact, irreversibility, dynamic context, or
  subjective outcomes. Set `requires_human_approval` for deterministic approval
  requirements.

  ```json
  {
    "intent_ambiguity": "medium",
    "context_variability": "variable",
    "outcome_observability": "delayed",
    "decision_impact": "high",
    "reversibility": "costly",
    "requires_human_approval": false,
    "rationale": "Shared integration architecture with delayed production evidence."
  }
  ```

  Initialize once:

  ```shell
  agent-learn task-policy-init --agent-id <agent_id> --task-id <decision_task_id> --decision-context "<stable delegated choice>" --actions ./actions.json --complexity-profile ./complexity.json
  ```

  The CLI marks the policy as `delegated_decision`. Existing unmarked question or
  automation policies are legacy records and are excluded by `--decision-only`.
  Existing marked policies with no profile remain standard. Apply an approved
  profile without creating a policy snapshot:

  ```shell
  agent-learn task-policy-complexity-set --agent-id <agent_id> --task-id <decision_task_id> --profile ./complexity.json
  ```

  ## 4. Consume learned policy feedback before execution

  This step is mandatory for every eligible decision. Do not merely inspect
  `task-policy` and then choose independently.

  ```shell
  agent-learn task-policy-decide --agent-id <agent_id> --task-id <decision_task_id>
  ```

  By default, the command samples from the learned probabilities to preserve a
  small amount of exploration. Use `--greedy` only when deterministic exploitation
  is required.

  The output provides:

  - `selected_action`: the action Scout must execute for this attempt;
  - `recommended_action`: the current highest-probability action;
  - `policy_id`, `policy_version`, probability, and behavior `logprob`;
  - `selected_action_feedback`: attempts, correctness rate, mean reward, recent
    result summaries, and intent/adherence/completion scores;
  - `historical_feedback`: the same evidence for every alternative;
  - `autonomy`: the authoritative mode, evidence criteria, execution permission,
    feedback request, audit sample, outcome-recording strategy, and proportional
    `complexity` calculation.

  Use the feedback as execution context. In particular, avoid repeating failure
  modes named in recent result summaries, and preserve behavior associated with
  high completion/correctness scores. Execute `selected_action`; otherwise the
  recorded policy probability and feedback loop are not causally meaningful.

  Never infer autonomy from a probability, logit, or answer quality. Follow the
  returned `autonomy` object exactly:

  - when `execute_without_confirmation` is `false`, the policy is supervised;
  - when `execute_without_confirmation` is `true`, use `selected_action` without
    routine acceptance or rejection;
  - when `request_user_feedback` is `true`, request feedback for the reason in
    `feedback_reason`, unless `observable_outcome_satisfies_feedback` is true
    and execution produced an independent outcome;
  - when `outcome_recording` is `observable_outcome`, record the independently
    observed execution result instead of asking the user routinely.

  Inspect `autonomy.complexity.profile_source`, `tier`, `risk_floors`, and the
  per-criterion required values. Never override them in Scout. A true
  `requires_human_approval` profile always remains supervised.

  ## 5. Execute or preserve a pending recommendation

  When Scout can perform the chosen delegation now, execute it and evaluate the
  user-visible result. Register a full episode from the observable outcome even
  when no user feedback is requested. Set `metadata.outcome_source` to
  `observable`, set `task_completed` from the actual result, and set
  `correct_action_id` only when the result independently establishes the correct
  alternative. Automatic tool success alone is not task completion.

  For a recommendation without an immediate observable outcome:

  - supervised mode: preserve it as pending and ask for feedback;
  - autonomous mode without an audit: present the recommendation without
    routine acceptance/rejection and do not fabricate a completed episode;
  - autonomous drift audit: preserve it as pending, present it without requiring
    pre-approval, then ask for acceptance/rejection.

  To preserve a pending attempt, generate a stable episode ID and register the
  decision fields, but omit `execution_status`, `result_summary`,
  `metadata.correct_action_id`, and `metadata.task_completed`. Those omissions
  keep the episode ineligible for scoring and training:

  ```json
  {
    "id": "<stable episode UUID>",
    "agent_name": "Scout",
    "task_name": "<delegated decision name>",
    "user_input": "<user request>",
    "assistant_output": "<recommendation and rationale>",
    "intent_summary": "<what the user needs to choose>",
    "action_type": "recommendation",
    "action_id": "<selected_action.id>",
    "action_name": "<selected_action.description>",
    "target": "<decision target>",
    "input_summary": "<decision context and constraints>",
    "expected_outcome": "<observable acceptance or execution criterion>",
    "policy_id": "<policy_id from task-policy-decide>",
    "policy_version": 0,
    "action_logprob": -1.0986122886681098,
    "metadata": {"feedback_status": "pending"}
  }
  ```

  Register the pending episode against the marked policy:

  ```shell
  agent-learn task-episode-register --agent-id <agent_id> --task-id <decision_task_id> --episode ./pending-episode.json --require-decision-policy
  ```

  When `autonomy.request_user_feedback` is true, tell the user why feedback is
  requested and ask for one of:

  - `ACCEPT`, when the selected recommendation is endorsed;
  - `REJECT`, plus the correct action when known;
  - the observed execution or test result.

  Five supervised or audited recommendation prompts with no follow-up are five
  pending attempts, not five trainable outcomes. Autonomous non-audit
  recommendations do not create pending feedback debt.

  ## 6. Complete or register the decision episode

  For immediate execution, build a full episode using the decision command's
  policy fields. For a pending recommendation, retrieve the incomplete record
  with `task-episodes-list --include-incomplete`, update the same episode ID,
  and register it again. Never create a second episode for the feedback. The SDK
  preserves the original decision time in `metadata.decision_created_at` and
  moves completion into the current training window.

  Independently determine `correct_action_id` when evidence supports one. It may
  differ from `action_id`. Do not call an action correct only because Scout
  selected it. Record `task_completed` from the user-visible outcome, not merely
  from successful tool invocation.

  ```json
  {
    "id": "<existing pending ID or new execution episode UUID>",
    "agent_name": "Scout",
    "task_name": "<delegated decision name>",
    "user_input": "<user request that created the decision context>",
    "assistant_output": "<final user-visible result>",
    "intent_summary": "<what the user needed>",
    "action_type": "delegation",
    "action_id": "<selected_action.id>",
    "action_name": "<selected_action.description>",
    "target": "<decision target>",
    "input_summary": "<decision context>",
    "expected_outcome": "<observable correctness and completion criteria>",
    "execution_status": "completed",
    "result_summary": "<quality, correctness, failures, and remaining work>",
    "policy_id": "<policy_id from task-policy-decide>",
    "policy_version": 0,
    "action_logprob": -0.6931471805599453,
    "metadata": {
      "feedback_status": "observed",
      "outcome_source": "user_feedback_or_observable",
      "correct_action_id": "<independently supported action id>",
      "task_completed": true
    }
  }
  ```

  Register only against a marked decision policy:

  ```shell
  agent-learn task-episode-register --agent-id <agent_id> --task-id <decision_task_id> --episode ./episode.json --require-decision-policy
  ```

  For an accepted recommendation, the selected action may become
  `correct_action_id` and `task_completed: true`. For a rejected recommendation
  with a known alternative, record that alternative as `correct_action_id`. If
  the user rejects without identifying a correct action, omit
  `correct_action_id`, set `task_completed: false`, and describe the rejection
  in `result_summary`.

  ## 7. Score and verify execution feedback

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <decision_task_id> --limit 100
  agent-learn task-episodes-list <agent_id> --task-id <decision_task_id> --limit 100
  ```

  Confirm that the episode has three completed core metrics and a non-null final
  reward. The next `task-policy-decide` call will return this execution feedback.

  ## Execution sequence

  1. Apply the eligibility gate; stop if no delegated decision exists.
  2. Establish the durable store.
  3. Resolve or initialize a marked delegated-decision policy.
  4. Call `task-policy-decide` and consume its learned feedback.
    5. Follow `autonomy.execute_without_confirmation`, `request_user_feedback`,
      and `outcome_recording` exactly.
    6. Execute `selected_action`, or preserve a supervised/audited recommendation
      as pending when no observable outcome exists.
  7. Evaluate user-visible correctness and completion.
    8. Complete the same pending ID, or register the automatic observable outcome,
      with `--require-decision-policy`.
  9. Score and verify the completed episode for the next decision.

  ## Smoke-test prompt

  Use a complete request such as:

  > For editing a large Python repository with repeated tool calls, prioritize
  > correctness and tool-use reliability over latency. Choose GPT-5.6 Sol or
  > GPT-5.6 Terra, use the selected model to review this change, and treat whether
  > the review finds the seeded defect as the correctness outcome.

  This supplies alternatives, reusable context, an executed delegate, and an
  observable correctness criterion. Asking only "Sol vs Terra for my use case?"
  does not.

  For a supervised recommendation-only test such as "Which Azure workload
  should receive and process these PubSub messages?", follow each recommendation
  with `ACCEPT`, `REJECT; correct action is <action_id>`, or an execution result.
  Once `autonomy.mode` becomes `autonomous`, stop routine feedback and respond
  only when `request_user_feedback` samples a drift audit.
---