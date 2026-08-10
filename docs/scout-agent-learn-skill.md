---
name: agent-learn
description: "Use when Scout must choose or recommend among two or more executable alternatives, including requests such as 'which Azure workload should we use?', 'recommend an Azure workload', or choosing a language model, skill, automation, tool, workflow, escalation path, or implementation strategy. Also use when the user accepts, rejects, or reports the outcome of a prior agent-learn recommendation. Do not use for factual questions, ordinary chat answers, summaries, reporting, or running agent-learning scoring/training automation."
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

  Define a complexity profile from the stable decision boundary and its real
  consequences, never from the wording or confidence of Scout's own answer. Use
  `low` only for a bounded recommendation (not deployment or mutation) with
  clear intent, stable context, direct acceptance/correctness feedback, low
  immediate impact, and a reversible choice. If those facts are not established,
  keep the SDK's standard default. Elevate when the request establishes high
  impact, irreversibility, dynamic context, or subjective outcomes. Set
  `requires_human_approval` for deterministic approval requirements.

  The `choose-azure-pubsub-message-processor` recommendation described above is
  low complexity when Scout is only recommending among Functions, Container
  Apps, and AKS and the user directly accepts or corrects that recommendation:

  ```json
  {
    "intent_ambiguity": "low",
    "context_variability": "stable",
    "outcome_observability": "direct",
    "decision_impact": "low",
    "reversibility": "reversible",
    "requires_human_approval": false,
    "rationale": "Bounded, reversible workload recommendation with direct user feedback."
  }
  ```

  This profile scores `1` for a three-action policy and maps to the `low` tier.
  Actually provisioning or changing the integration is a separate, higher-impact
  decision and must not reuse this recommendation profile.

  Configure decision authority independently from complexity and execution
  permission:

  - `low` uses the TaskPolicy's learned softmax probabilities and learns from
    explicit accept/reject feedback or independently observed outcomes. This is
    the default for new and legacy policies.
  - `full` lets Scout resolve a structured DecisionFrame with hard constraints,
    Bayesian evidence aggregation, robust utility, and information gain. Use it
    only when the policy owner explicitly delegates that reasoning authority.

  Never infer `full` from Scout's confidence, model capability, policy
  probability, or a low complexity tier. Full decision authority does not
  override `requires_human_approval`, safety controls, service permissions, or
  the returned execution authorization.

  When reusing this PubSub recommendation policy, inspect
  `complexity_profile_source`. If it is `default`, apply the low profile with
  `task-policy-complexity-set` before the next decision. Never replace a
  `configured` profile without the policy owner's approval.

  Initialize once:

  ```shell
  agent-learn task-policy-init --agent-id <agent_id> --task-id <decision_task_id> --decision-context "<stable delegated choice>" --actions ./actions.json --complexity-profile ./complexity.json --decision-authority <low_or_full>
  ```

  The CLI marks the policy as `delegated_decision`. Existing unmarked question or
  automation policies are legacy records and are excluded by `--decision-only`.
  Existing marked policies with no profile remain standard. Apply an approved
  profile without creating a policy snapshot:

  ```shell
  agent-learn task-policy-complexity-set --agent-id <agent_id> --task-id <decision_task_id> --profile ./complexity.json
  ```

  Existing policies with no `decision_authority` are `low`. Apply an explicit
  owner-approved authority without creating a second policy or learned snapshot:

  ```shell
  agent-learn task-policy-authority-set --agent-id <agent_id> --task-id <decision_task_id> --authority <low_or_full>
  ```

  Never create separate reasoned and learned policies for the same
  `(agent_id, task_id)` decision. Both routes use the one active TaskPolicy and
  its exact action taxonomy.

  ## 4. Select through the active TaskPolicy

  This step is mandatory for every eligible decision. Do not merely inspect
  `task-policy` and then choose independently.

  Inspect the active policy first:

  ```shell
  agent-learn task-policy --agent-id <agent_id> --task-id <decision_task_id>
  ```

  Read `current_policy.metadata.decision_authority`. Treat a missing field as
  `low`; do not upgrade it during execution.

  ### Low decision authority

  ```shell
  agent-learn task-policy-decide --agent-id <agent_id> --task-id <decision_task_id> --greedy
  ```

  **User-facing recommendations must use `--greedy`.** This returns the current
  highest-probability action while still exposing all alternatives and their
  evidence. Do not expose a sampled exploratory action as the recommended
  answer. Sampling is allowed only for an explicitly identified exploration
  trial whose selected action will actually be executed and whose independent
  outcome can be observed; run that trial separately without `--greedy`.

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

  ### Full decision authority

  Build one JSON DecisionFrame from the current request and independently
  supported evidence. Include every TaskPolicy action exactly once by
  `action_id`; do not add, remove, or redefine actions in the frame:

  ```json
  {
    "task": "Choose a deployment region for this workload",
    "criteria": [
      {"id": "capacity_fit", "weight": 0.6, "minimum_sources": 2},
      {"id": "latency_fit", "weight": 0.4, "minimum_sources": 1}
    ],
    "constraints": ["data_residency"],
    "options": [
      {
        "action_id": "east",
        "constraint_results": {"data_residency": true},
        "evidence": [
          {"criterion_id": "capacity_fit", "source": "capacity_api", "support": 0.9, "confidence": 0.9},
          {"criterion_id": "capacity_fit", "source": "quota_report", "support": 0.8, "confidence": 0.8},
          {"criterion_id": "latency_fit", "source": "latency_test", "support": 0.7, "confidence": 0.9}
        ]
      },
      {
        "action_id": "west",
        "constraint_results": {"data_residency": true},
        "evidence": [
          {"criterion_id": "capacity_fit", "source": "capacity_api", "support": 0.6, "confidence": 0.9},
          {"criterion_id": "capacity_fit", "source": "quota_report", "support": 0.7, "confidence": 0.8},
          {"criterion_id": "latency_fit", "source": "latency_test", "support": 0.9, "confidence": 0.9}
        ]
      }
    ],
    "minimum_margin": 0.05,
    "uncertainty_penalty": 0.1,
    "max_uncertainty": 1.0
  }
  ```

  Evidence `support` is a probability in `[0,1]`; `confidence` is in `(0,1]`.
  Source names must identify genuinely independent observations. Do not split
  one source into aliases to satisfy `minimum_sources`, and do not convert
  Scout's own recommendation into evidence.

  ```shell
  agent-learn task-policy-decide --agent-id <agent_id> --task-id <decision_task_id> --decision-frame ./decision-frame.json
  ```

  Follow the returned `status` exactly:

  - `resolved`: execute `selected_action` only when
    `autonomy.execute_without_confirmation` is true, then record its observable
    outcome against the returned policy ID and version;
  - `needs_evidence`: gather the first `information_needs` item, update the
    frame, and decide again; do not guess or ask for a tie-break yet;
  - `needs_user_tie_break`: persist the complete result JSON, present
    `proposed_action`, and request exactly `ACCEPT` or `REJECT`;
  - `needs_user_feedback`: deterministic human approval is required; persist
    the result and request exactly `ACCEPT` or `REJECT`;
  - `no_viable_option`: stop and reframe the constraints or action taxonomy;
    never select a ruled-out action.

  A reasoned result has `selection_basis: bayesian_decision` and
  `action_logprob: null`. Never invent a softmax probability or log-probability
  for it.

  For a pending reasoned result, apply one user disposition to that exact file:

  ```shell
  agent-learn task-policy-adjudicate --agent-id <agent_id> --task-id <decision_task_id> --decision-result ./decision-result.json --disposition <accept_or_reject>
  ```

  `accept` resolves the proposed action. For a tie, `reject` advances to the
  next tied action, which still requires explicit acceptance; persist the new
  result before asking again. `rejected` or `no_viable_option` requires a new
  frame. Never interpret rejection as proof that an unreviewed alternative is
  correct. The result is bound to its exact policy ID and version. If the active
  snapshot changes before the user replies, discard the stale result and rerun
  the DecisionFrame against the current TaskPolicy before asking again.

  ## 5. Execute or preserve a pending recommendation

  When Scout can perform the chosen delegation now, execute it and evaluate the
  user-visible result. Register a full episode from the observable outcome even
  when no user feedback is requested. Set `metadata.outcome_source` to
  `observable`, set `task_completed` from the actual result, and set
  `correct_action_id` only when the result independently establishes the correct
  alternative. Automatic tool success alone is not task completion.

  For a low-authority recommendation without an immediate observable outcome:

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

  Pending registration is mandatory before sending a supervised or audited
  recommendation. If registration fails, do not silently return an ordinary
  recommendation. Report that feedback capture failed, include the CLI error,
  and do not claim that the decision will be learned.

  When `autonomy.request_user_feedback` is true and no independent execution
  outcome has satisfied feedback, the final response must end with this visible
  handoff (substitute the selected and alternative action IDs):

  ```text
  Feedback requested (<feedback_reason>).
  Reply with one of:
  - ACCEPT
  - REJECT: <correct_action_id>
  - RESULT: <observed execution or test result>
  ```

  Do not bury this handoff in rationale and do not omit it merely because the
  answer sounds confident. A recommendation-only response has no independent
  observable outcome, so supervised mode always requires the handoff.

  Five supervised or audited recommendation prompts with no follow-up are five
  pending attempts, not five trainable outcomes. Autonomous non-audit
  recommendations do not create pending feedback debt.

  For full authority, preserve the complete reasoned result JSON before a
  binary tie-break or approval request. Do not register an action episode until
  an action has been resolved. After accepted adjudication, use
  `metadata.feedback_status: adjudicated` and
  `metadata.decision_disposition: accepted`; do not use the low
  authority durable `accepted`/`rejected` authorization statuses for a
  frame-local tie-break. A rejected or exhausted result has no selected action;
  retain the result for audit and reframe instead of fabricating an episode.

  ## 6. Complete or register the decision episode

  For immediate execution, build a full episode using the decision command's
  policy fields. For a pending low-authority recommendation, retrieve the incomplete record
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
    "action_logprob": null,
    "metadata": {
      "feedback_status": "<accepted, rejected, observed, or adjudicated>",
      "selection_basis": "<learned_policy or bayesian_decision>",
      "outcome_source": "user_feedback_or_observable",
      "correct_action_id": "<independently supported action id>",
      "task_completed": true
    }
  }
  ```

  Copy the numeric `action_logprob` returned by a low-authority learned
  decision. Keep it `null` for a full-authority reasoned decision; never derive
  one from `action_probabilities`.

  Register only against a marked decision policy:

  ```shell
  agent-learn task-episode-register --agent-id <agent_id> --task-id <decision_task_id> --episode ./episode.json --require-decision-policy
  ```

  For low-authority `ACCEPT`, set `feedback_status: accepted`, set
  `correct_action_id` to the selected action, and set `task_completed: true`.
  Registration immediately
  authorizes that action for this agent and task policy. Future decisions use it
  without confirmation or drift-audit prompts; scoring and training may happen
  afterward. For `REJECT`, set `feedback_status: rejected`. A rejection revokes
  an earlier acceptance. When the user identifies an alternative, record it as
  `correct_action_id`; otherwise omit `correct_action_id`, set
  `task_completed: false`, and describe the rejection in `result_summary`.
  Independently observed results use `feedback_status: observed` and continue
  through the complexity-proportional statistical gates.

  For a full-authority tie-break, use `feedback_status: adjudicated` and record
  the binary response in `decision_disposition`. Acceptance resolves only that
  framed decision; it does not pin the action for later frames. Set
  `correct_action_id` only when execution or another independent observation,
  not the tie-break alone, establishes correctness.

  ## 7. Score, train, and verify execution feedback

  For low authority:

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <decision_task_id> --limit 100
  agent-learn train --agent-id <agent_id> --task-id <decision_task_id> --decision-only --limit 1 --min-episodes 1 --skip-scoring
  agent-learn task-policy --agent-id <agent_id> --task-id <decision_task_id>
  agent-learn task-episodes-list <agent_id> --task-id <decision_task_id> --limit 100
  ```

  Train each newly completed feedback episode exactly once; do not replay an old
  episode merely to manufacture policy stability. Confirm that the episode has
  three completed core metrics and a non-null final reward, the training run
  used one episode, and the policy version advanced. The next
  `task-policy-decide` call will return the updated policy and reassess autonomy.

  For full authority, score completed episodes for quality and audit, but do
  not apply REINFORCE. The reasoned action was not sampled from the softmax
  behavior policy, and logits do not control full-authority selection:

  ```shell
  agent-learn score --agent-id <agent_id> --task-id <decision_task_id> --limit 100
  agent-learn task-policy --agent-id <agent_id> --task-id <decision_task_id>
  agent-learn task-episodes-list <agent_id> --task-id <decision_task_id> --limit 100
  ```

  `agent-learn train --decision-only` rejects full-authority policies with the
  reason `full decision authority uses reasoned resolution, not REINFORCE`.

  ## Execution sequence

  1. Apply the eligibility gate; stop if no delegated decision exists.
  2. Establish the durable store.
  3. Resolve or initialize a marked delegated-decision policy.
  4. Read `decision_authority`: call `task-policy-decide --greedy` for `low`, or
     build a complete frame and call `task-policy-decide --decision-frame` for
     `full`.
  5. Follow `autonomy.execute_without_confirmation`, `request_user_feedback`,
     `observable_outcome_satisfies_feedback`, and `outcome_recording` exactly.
  6. Execute `selected_action`, or register a supervised/audited recommendation
     as pending before responding when no observable outcome exists.
  7. Evaluate user-visible correctness and completion.
  8. End a low-authority pending recommendation with the visible feedback
     handoff and complete the same pending ID on follow-up. For a full-authority
     tie, persist and adjudicate the exact result until accepted or exhausted.
  9. Register observable outcomes with `--require-decision-policy`. Score every
     completed episode; train only low-authority policies, then verify the policy
     before its next decision.

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
  One `ACCEPT` makes that task policy autonomous immediately and suppresses all
  future feedback prompts until `REJECT`. Without explicit acceptance, its
  low-complexity profile can still earn statistical autonomy after three
  consistent, positively rewarded outcomes and policy updates.
---