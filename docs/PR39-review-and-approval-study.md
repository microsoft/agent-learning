# PR39 review and low-authority approval study

Reviewed: 2026-09-19  
Pull request: [microsoft/agent-learning#39](https://github.com/microsoft/agent-learning/pull/39)  
Head commit reviewed: `1d44215e6248fe0c4da91566d839ff22ff212394`

## Executive summary

**Recommendation: request changes before merge.**

The composite-tool architecture is reasonable, and the session-based correlation of a pending decision with a resumed Agent Framework tool call is a good direction. The central problem is not the Agent Framework mechanism. It is that the implementation treats two different concepts as the same event:

1. **Execution approval:** Agent Framework's ordinary `approved=True` response authorizes one pending function call.
2. **Learning acceptance:** agent-learning's `feedback_status: accepted` is durable authorization for the whole `(agent_id, task_id)` policy and pins the accepted action for future decisions.

PR39 currently converts the first event into the second. Consequently, approving one sampled action once makes that action the default for future calls, bypasses the statistical autonomy gates, disables audits, and marks the action as correct.

The surprising downstream behavior is therefore real, but it is not caused by session restoration. The adapter accurately activates an existing agent-learning rule while attaching that rule to the wrong Agent Framework signal.

## What the existing agent-learning contract says

The current core behavior was intentionally added in commit [`e9d5c39`](https://github.com/microsoft/agent-learning/commit/e9d5c39d78a3dd1d469020a5d95e6d9b539da59b), titled **"Grant durable user-approved autonomy in 0.7.0."**

The behavior is explicit in:

- [`docs/autonomy-complexity.md`](https://github.com/microsoft/agent-learning/blob/main/docs/autonomy-complexity.md#explicit-user-acceptance): one accepted episode immediately grants autonomy, pins its action, and sets the audit rate to zero.
- [`docs/decision-making.md`](https://github.com/microsoft/agent-learning/blob/main/docs/decision-making.md#evidence-gated-autonomy): a later rejection revokes the durable approval.
- [`tests/test_autonomy.py`](https://github.com/microsoft/agent-learning/blob/main/tests/test_autonomy.py): `test_one_user_acceptance_grants_durable_autonomy` codifies this behavior.
- [`src/agent_learning/autonomy.py`](https://github.com/microsoft/agent-learning/blob/main/src/agent_learning/autonomy.py): `assess_autonomy()` changes the recommended action to the latest explicitly accepted action and grants immediate eligibility.

This means the PR author did not misunderstand the existing core implementation. The adapter is exposing an ambiguity already present in the terminology:

- Agent Framework calls its one-call permission an **approval**.
- agent-learning calls durable task-level authorization an **acceptance**.

Those terms must not be treated as synonyms at the integration boundary.

## What Agent Framework approval means

Agent Framework distinguishes:

- approve this request once;
- reject this request once;
- approve and remember all future calls to a tool;
- approve and remember future calls only with the same arguments.

The official [`tool_approval_middleware.py`](https://github.com/microsoft/agent-framework/blob/main/python/samples/02-agents/tools/tool_approval_middleware.py) sample demonstrates these as separate host choices. A plain:

```python
request.to_function_approval_response(approved=True)
```

is a one-call execution approval. It is not an endorsement that the selected action is correct, and it is not a standing instruction to use that action on future tasks.

## Current PR39 low-authority flow

The current adapter does the following:

1. Calls `assess_autonomy()`.
2. If autonomy is not yet eligible, samples an action from the low-authority softmax policy.
3. Stores the exact decision and all candidate arguments in `AgentSession.state`.
4. Returns an Agent Framework approval request.
5. On `approved=True`, restores the pending decision and calls:

   ```python
   task_policy.adjudicate(pending.decision, TieBreakDisposition.ACCEPT)
   ```

6. Executes the selected child tool.
7. Captures the episode with:

   ```python
   feedback_status="accepted"
   correct_action_id=action.id
   ```

8. On the next call, `assess_autonomy()` sees the accepted episode, pins that action, grants autonomy immediately, and returns an audit rate of `0.0`.

That explains the behavior described in the PR's "Still needs work" section.

## Recommended semantic model

### A. One-call execution approval

Use this for the ordinary Agent Framework approval response.

- Restore and execute the exact pending sampled decision.
- Do not resample after resume.
- Do not write `feedback_status: accepted`.
- Do not write `correct_action_id` merely because execution was allowed.
- Capture the actual result through `episode_metadata_resolver`.
- Reassess autonomy on the next invocation.

This is the safest default and matches what a caller reasonably means by `approved=True`.

### B. Explicit learning acceptance

If the product still wants durable action pinning, expose it as a separate, explicit operation or response mode, such as **approve and remember this action for this task**.

- Clearly state its scope: all future decisions for `(agent_id, task_id)`.
- Persist `explicit_user_feedback` into the active policy metadata.
- Record who authorized it, when, and optionally its expiration/scope.
- Keep this separate from whether the action's execution outcome was correct.
- Provide an explicit revocation path.

Agent Framework's remembered approval helpers may be useful for standing execution permission, but a remembered tool permission is still not automatically the same as an agent-learning policy preference.

### C. Statistical autonomy

When the policy satisfies evidence gates:

- Select the recommended action greedily.
- Execute without routine approval.
- Preserve the configured drift-audit rate.
- Return a decision whose status reflects that execution was resolved and authorized.

### D. Rejection

For a low-authority sampled recommendation:

- Do not execute the action.
- Record rejection as negative feedback for that sampled action.
- Do not assume another candidate is correct.
- Re-evaluate after learning or on a later decision.

For a full-authority tie:

- Rejecting one candidate may advance to the next tied candidate.

The existing core `TaskPolicy.adjudicate()` already makes this distinction.

## Review findings

### 1. High: one-call approval is converted into durable policy authorization

Locations:

- [`adapter.py` approval adjudication](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L456-L476)
- [`adapter.py` accepted metadata](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L498-L520)

`approved=True` is transformed into `TieBreakDisposition.ACCEPT`, then captured as durable `feedback_status: accepted`. It also sets `correct_action_id` to the selected action.

Impact:

- One approval pins an action across unrelated future inputs for the same task.
- Statistical evidence gates are bypassed.
- The explicit-acceptance path sets audit rate to zero, so the PR description's "except for audit sampling" statement is not accurate.
- Permission to execute is treated as evidence that the action is correct.

Recommended fix:

- Introduce a distinct one-call execution-approval path.
- Reserve `feedback_status: accepted` for an explicit "approve and remember" learning decision.
- Let `episode_metadata_resolver` or another independent outcome source provide `correct_action_id`.

### 2. High: the adapter does not persist the supposedly durable approval

Locations:

- [`adapter.py` accepted episode capture](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L498-L527)
- [`autonomy.py` 500-episode query](https://github.com/microsoft/agent-learning/blob/main/src/agent_learning/autonomy.py#L379-L394)
- [`cli.py` reference persistence behavior](https://github.com/microsoft/agent-learning/blob/main/src/agent_learning/cli.py#L979-L991)

The CLI persists `explicit_user_feedback` into active policy metadata. The adapter only writes an episode. `assess_autonomy()` scans at most 500 full episodes, so an adapter-created acceptance eventually falls outside the query window and silently stops being durable.

Recommended fix:

- If durable acceptance remains supported, update and store active policy metadata exactly as the CLI does.
- If ordinary MAF approval becomes one-shot, do not persist it as explicit feedback at all.

### 3. Medium: autonomous execution returns `needs_user_feedback`

Locations:

- [`adapter.py` low-authority decision](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L400-L448)
- [`decision.py` low-authority result](https://github.com/microsoft/agent-learning/blob/main/src/agent_learning/decision.py#L874-L928)

Even when `assessment.eligible` is true and the adapter executes without asking the user, `TaskPolicy.decide()` returns `DecisionStatus.NEEDS_USER_FEEDBACK`. The adapter returns that unchanged decision alongside an executed result.

Impact:

- The result says feedback is required even though execution was treated as authorized and complete.
- Agents following the documented status may incorrectly ask again or treat a completed action as unresolved.
- Captured decision metadata is internally inconsistent with actual execution.

Recommended fix:

- Add a core API for authorizing a low-authority decision from an `AutonomyAssessment`, or normalize the result to `RESOLVED` with an authorization basis before execution.
- Do not mutate the result ad hoc without preserving policy identity and audit information.

### 4. Medium: malformed approval responses consume pending state before validation

Locations:

- [`adapter.py` approved response restoration](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L611-L636)
- [`adapter.py` rejected response handling](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/src/agent_learning/integrations/agent_framework/adapter.py#L638-L654)

Both paths remove the pending decision with `pop()` before validating the tool name, call ID, and arguments.

Impact:

- A malformed, stale, or tampered response raises an error and irreversibly deletes the pending decision.
- A corrected retry cannot resume the original call.

Recommended fix:

- Read with `get()`.
- Deserialize and validate.
- Remove the entry only after validation succeeds.

### 5. Medium: the new integration has no tests

The PR adds roughly 1,000 lines of integration code plus two large examples but no tests under `tests/`.

At minimum, tests should cover:

1. First supervised low-authority sample requests approval.
2. One-call approval executes the exact pending action without resampling.
3. One-call approval does not pin future actions.
4. Explicit approve-and-remember does pin and persist the action.
5. Rejection records feedback and does not execute.
6. Full-authority ties advance after rejection.
7. `requires_human_approval` prompts on every invocation.
8. Statistical autonomy executes without approval and returns a resolved status.
9. Drift audits occur at the configured rate.
10. Explicit acceptance has zero audits if that remains the documented contract.
11. Approval survives adapter or process reconstruction when intended to be durable.
12. Invalid approval correlation does not destroy pending state.
13. Multiple simultaneous approval requests are handled.
14. Selected child-tool success, exception, and metadata capture are correct.
15. Policy updates while an approval is pending have a defined behavior.

### 6. Low: examples process only the first approval in a response

Locations:

- [`agent_framework_low_authority.py`](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/examples/agent_framework_low_authority.py#L239-L255)
- [`agent_framework_full_authority.py`](https://github.com/microsoft/agent-learning/blob/1d44215e6248fe0c4da91566d839ff22ff212394/examples/agent_framework_full_authority.py#L317-L333)

Both examples use `response.user_input_requests[0]`. Agent Framework can return multiple approval requests in one response, and its official sample sends all corresponding approval responses together.

Recommended fix:

- Either process every request in the response or explicitly constrain/document the adapter to one outstanding composite call.

## Suggested response to the PR's "Still needs work" section

> You are not missing anything in the Agent Framework resume/session mechanism; that part of the implementation looks directionally right. The confusing behavior comes from two different meanings of "approval."
>
> In Agent Framework, `request.to_function_approval_response(approved=True)` approves that pending function call once. In agent-learning today, a completed episode with `feedback_status="accepted"` has a much stronger meaning: it is durable authorization for the `(agent_id, task_id)` policy, pins the accepted action, bypasses the statistical autonomy gates, and sets the audit rate to zero. That behavior is intentional in the current core implementation and is covered by `test_one_user_acceptance_grants_durable_autonomy`, but I do not think a normal MAF execution approval should automatically be promoted to that durable learning signal.
>
> For this adapter, I suggest we separate:
>
> 1. **Approve this execution:** restore the exact pending sampled decision and execute it once. Do not set `feedback_status="accepted"` or `correct_action_id` solely from this approval.
> 2. **Approve and remember this action for this task:** an explicit opt-in that records durable `explicit_user_feedback` in policy metadata and pins the action under the existing agent-learning semantics.
> 3. **Outcome feedback:** record correctness/completion from the actual result or an explicit post-execution user evaluation, not from permission to execute.
>
> A low-authority rejection should remain negative feedback for the sampled action; it should not imply that the next action is correct. A full-authority tie is different: rejection can advance to the next tied candidate.
>
> One correction to the current description: explicit acceptance currently sets `audit_rate=0.0`, so it does not continue audit sampling. Only statistically earned autonomy uses the tier-specific audit rate.
>
> I would treat the approval semantics and tests as merge blockers. The session correlation approach can stay, but the normal approval path should be one-shot by default. If we want durable approval, it needs a separate, explicit UX and must be persisted to policy metadata, as the CLI already does.

## Suggested concise inline review comment

Place near the call to `adjudicate(..., ACCEPT)`:

> A normal Agent Framework `approved=True` response authorizes this pending invocation; it does not mean "this action is correct and should become the durable default." Calling `TaskPolicy.adjudicate(..., ACCEPT)` here causes the captured episode to grant task-wide autonomy, pin this action, and disable audits on later calls. Please keep one-call execution approval separate from explicit agent-learning acceptance. Only an explicit "approve and remember" flow should write `feedback_status="accepted"` and persist `explicit_user_feedback`.

## Study notes

### Three independent axes

Keep these concepts separate when reasoning about the system:

| Axis | Question | Typical scope |
|---|---|---|
| Tool permission | May this concrete function call execute? | One call, one tool, or one tool+argument rule |
| Policy selection | Which action should the policy choose? | `(agent_id, task_id)` |
| Outcome evidence | Was the executed action correct or useful? | One completed episode |

A single user interaction can provide more than one signal, but the product must ask for those signals explicitly. It should not infer all three from a generic "Approve?" response.

### Why session state is still needed

The adapter must preserve the pending decision because replaying the composite tool would otherwise sample or resolve again. The session record should include:

- policy ID and version;
- decision result;
- selected/proposed action;
- candidate action inputs;
- original user input;
- autonomy assessment;
- correlation ID.

On resume, validate the response first, then atomically consume the pending record and execute the exact stored decision.

### Approval scope should be visible

Recommended user choices:

- **Approve once**
- **Reject**
- **Approve and remember for this task**

For higher-risk cases, omit the remember option or require an expiration and explicit scope.

## Validation performed

- Reviewed all 12 changed files in PR39.
- Checked the PR's review comments and discussion: none were present at review time.
- Confirmed GitHub checks were passing:
  - `build-installer`
  - `license/cla`
  - `test`
- Ran the repository test suite from source:
  - **226 passed**
  - **2 skipped**
  - **1 warning** about the local `pytest-asyncio` configuration
- Ran Ruff over the new integration and examples:
  - **All checks passed**
- Imported the integration against installed `agent-framework-core` 1.18.0:
  - **Import succeeded**
- Confirmed there are no Agent Framework integration tests in `tests/`.
- Confirmed `git diff --check main...HEAD` passes.

The passing suite does not validate the new integration because no PR39 tests are collected.
