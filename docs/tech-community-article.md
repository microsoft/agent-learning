# Your Agent Finished the Task. Did It Learn Anything?

**Continuous agent improvement with Agent Learning: turn execution outcomes
into evidence for the next decision, without retraining the foundation model.**

You gave your agent a powerful model. Connected your data. Wired up the tools.
Refined the prompts.

It can act. It can produce an answer. It can complete a workflow.

But ask one more question:

**When it faces the same decision tomorrow, what changes because of what
happened today?**

If the answer is "nothing," the feedback loop is still open. A successful tool
call becomes a log entry. A failed strategy becomes another trace. The evidence
exists, but the next choice does not use it.

That is the opportunity behind
[`agent-learning`](https://github.com/microsoft/agent-learning), an open-source
Python SDK and command-line tool for improving recurring agent decisions.
Define the alternatives. Execute a choice. Measure the outcome. Update a small,
inspectable policy that influences what happens next.

**The foundation model stays frozen. The decisions don't have to.**

No foundation-model fine-tuning. No GPU training job for policy updates. No
hidden prompt rewrites. A decision-and-feedback loop in your existing Python
application, with versioned policies and explicit controls for human oversight.

Not a promise that every run will be better. A concrete way to make experience
count.

## Give experience a job

Consider a support agent that can choose among approved response strategies.
Each strategy is an action the application can execute. After execution, the
application can check whether the response followed the required format,
addressed the request, and satisfied the expected outcome.

The SDK organizes a recurring decision around a **TaskPolicy**, identified by
an `agent_id` and a `task_id`. A TaskPolicy has at least two explicit executable
alternatives. One task might choose a repository search strategy; another might
choose a support workflow. Their histories and policies remain separate.

Start where a choice changes the result: exact search or semantic retrieval;
a lightweight workflow or specialist escalation; one approved response
strategy or another.

**The unit of improvement is a decision, not a conversation.** Your application
defines and executes the alternatives. The SDK does not invent tools or perform
their actions. Repetition alone is not feedback: the policy needs an observable
outcome or explicit user feedback associated with what was chosen.

## Small policy. Meaningful leverage.

The learned decision loop is deliberately concrete:

1. **Choose.** Ask the policy to select an action from its current distribution.
2. **Execute.** Run that action in the host application, respecting the decision's
   confirmation and approval requirements.
3. **Record.** Preserve an episode with the selected action, policy version,
   selection probability, input, output, latency, and outcome evidence.
4. **Score.** Evaluate the episode and combine the measurements into a reward.
5. **Update.** Use a batch of completed, scored episodes to update the policy and
   persist a new snapshot.

Underneath, the built-in learner uses **REINFORCE with a baseline** over a
discrete softmax policy. Softmax turns action scores, called logits, into
probabilities. Better-than-baseline outcomes push the selected action toward a
higher probability; worse outcomes push in the other direction. An exploration
term helps the policy avoid concentrating too early on one alternative.

The object being trained is small: action preferences, not language-model
weights. Updates run on the CPU inside the Python process. Your application
decides when to score and train, manually, on a schedule, or in response to
events. The SDK does not start a background training service for you.

**You do not have to retrain the model to change how your application chooses
to use it.** That is the architectural shift.

## Make evaluation do more than produce a score

The default evaluation pipeline measures three dimensions:

| Dimension | Question |
|---|---|
| Intent resolution | Did the response address the user's intent? |
| Task adherence | Did execution follow the task's requirements? |
| Task completion | Did it achieve the expected outcome? |

A reward shaper combines completed metric scores into a value between -1 and
+1. Configuration controls their relative weights and supports latency
penalties. Individual metric results and aggregate rewards are persisted so
developers can inspect more than the final number.

Scoring is local by default when no Azure evaluator configuration is supplied.
The SDK offers standard-library scorers, scikit-learn-based text scorers, a local
Phi model backend through ONNX, and Azure AI Evaluation-backed evaluators when
configured. Model-based backends have their own dependencies and configuration
requirements; the local model backend also needs a downloaded model bundle.
Applications can supply custom metric evaluators through the runner's metrics
interface without replacing the learner.

Here is the critical connection: **evaluation becomes an input to the next
decision, not just a report about the last one.**

That also makes reward quality non-negotiable. A format check can establish that
required fields exist; it cannot establish that a customer's problem was solved.
Use independent checks, observed tool results, and human feedback where needed.
Compare evaluator scores with actual task outcomes. Learning to optimize the
wrong measurement is still learning, just not the improvement you wanted.

## Don't take the pitch. Run the loop.

The repository includes support-ticket scoring components for three response
templates: a detailed on-topic response, a terse response, and an off-topic
response. Local scorers evaluate required content, prohibited content, and
expected completion tokens. The intent scorer is fitted on a small labeled
corpus. This walkthrough reuses those components with complete episode records.

With Git and Python 3.10 or later installed, run these commands in a terminal.
An activated virtual environment is recommended. Installing dependencies needs
network access; the demo itself needs no Azure credentials or model download.

```shell
git clone https://github.com/microsoft/agent-learning.git
cd agent-learning
python -m pip install -e .
```

Run the following Python with the repository root as the working directory.
It exercises the learning pipeline directly; it does not authorize a real
support workflow or bypass the TaskPolicy approval checks described below.

```python
import random
from datetime import datetime, timedelta, timezone

from agent_learning import Episode, InMemoryStore, LearnerConfig, SoftmaxPolicy
from examples.scored_optimization import (
   ACTIONS, EXPECTED_TOKENS, TEMPLATES, ScoredRunner,
   build_example_scorers, contract_for, render_query,
)

store = InMemoryStore()
policy = SoftmaxPolicy.from_actions(
   ACTIONS, agent_id="article-demo", rng=random.Random(11)
)
store.store_policy(policy.snapshot())
runner = ScoredRunner(
   store=store, policy=policy, metrics=[],
   scorers=build_example_scorers(),
   learner_config=LearnerConfig(learning_rate=0.5, entropy_bonus=0.02),
)
before = dict(zip(TEMPLATES, policy.probabilities()))
started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

for batch in range(30):
   for attempt in range(30):
      decision = policy.choose()
      template = TEMPLATES[decision.action.id]
      ticket = "TICKET-4471"
      response = template.render(ticket)
      snapshot = policy.snapshot()
      store.store_episode(Episode(
         agent_id="article-demo",
         created_at=(
            started_at + timedelta(seconds=batch * 30 + attempt)
         ).isoformat(),
         user_input=render_query(ticket), assistant_output=response,
         intent_summary="Respond to a password-reset request",
         expected_outcome="Meet the response contract and expected content",
         execution_status="completed",
         result_summary="Rendered a template; no account action performed",
         policy_id=snapshot.id, policy_version=snapshot.version,
         action_id=decision.action.id, action_logprob=decision.logprob,
         metadata={
            "contract": contract_for(ticket),
            "expected_tokens": EXPECTED_TOKENS,
            "routing_correct": template.routing_correct,
            "hallucinated_class": template.hallucinated,
         },
      ))
   run = runner.run_offline_batch("article-demo", episode_limit=30)
   assert run.metrics["episodes_used"] == 30

after = dict(zip(TEMPLATES, policy.probabilities()))
assert after["template_rich"] > before["template_rich"]
for action_id in TEMPLATES:
   print(f"{action_id}: {before[action_id]:.2f} -> {after[action_id]:.2f}")
```

Watch the before-and-after probabilities. This runs 30 batches of 30 episodes
and verifies that every batch actually contributes to learning. The response
templates stay the same. The evaluators stay the same. **What changes is which
response the policy is likely to choose.**

Here is the output from a local run with Python 3.12:

```text
template_rich: 0.33 -> 0.95
template_terse: 0.33 -> 0.03
template_offtopic: 0.33 -> 0.02
```

**From roughly one choice in three to nineteen in twenty, without changing a
single response template.** Those numbers are selection probabilities, not
accuracy scores. They show the policy favoring the template rewarded by this
demo's evaluators, not a 95% real-world success rate.

The intent, expected outcome, execution status, and result summary are not
decorative metadata. Along with the selected action, they identify a complete
episode that the runner can use. A pending attempt is not a completed outcome.
Distinct synthetic timestamps keep the newest batch unambiguous in this demo;
real integrations should record actual event times.

The demo uses in-memory storage, so each run starts fresh. It does not contact a
ticketing system, send a reset email, or verify that a real password was changed.
The intent scorer is trained on examples of these same templates; the exercise
is a controlled illustration of the feedback loop, not a held-out evaluation of
generalization. It is not a production accuracy claim, a fine-tuning benchmark,
or evidence of guaranteed cost savings.

For integration into your own agent, the
[decision-making guide](https://github.com/microsoft/agent-learning/blob/main/docs/decision-making.md)
covers the CLI lifecycle: `task-policy-init`, `task-policy-decide`,
`task-episode-register`, `score`, and `train`.

## Learn from history. Reason over current evidence.

Historical learning is not the only selection route. A TaskPolicy declares one
of two decision authority modes:

| Mode | Selection method |
|---|---|
| `low` | Choose using the policy's learned action probabilities. |
| `full` | Compare current evidence in a structured DecisionFrame. |

In `full` mode, the application supplies options, weighted criteria, hard
constraints, and evidence. The resolver checks feasibility and compares utility
with an uncertainty penalty. It can select a unique robust winner, request
additional evidence, require human approval, or ask for an accept/reject
tie-break. It can also report that no option is viable.

This is an evidence-comparison algorithm, not an autonomous research agent.
Evidence must be collected and supplied by the host application. The CLI
training command skips full-authority policies: a reasoned choice was not
sampled from the learned softmax policy and should not be treated as though it
was. Its completed episodes can still be scored and inspected.

## Confidence is not permission

"The policy prefers this action" and "the agent may execute it without asking"
are different statements. **Agent Learning keeps them different.**

For learned policies, the SDK assesses multiple gates before recommending
execution without routine confirmation: sufficient scored outcomes, a
conservative statistical lower bound on correctness, positive mean reward,
action probability, the margin over alternatives, and stability across policy
snapshots.

The criteria scale with the policy's declared complexity and risk. Explicit user
acceptance provides another authorization route: it can pin an accepted action
and stop repeated feedback prompts until a later rejection. A policy configured
to require human approval retains that requirement. Full-authority resolution
has its own evidence and approval checks; it does not need to pass the learned
policy's statistical gates to resolve a decision.

The `task-policy-decide` response exposes `execute_without_confirmation`,
`request_user_feedback`, and `outcome_recording`. The integrating agent must
honor those fields and preserve outcomes. They are not a replacement for the
application's access controls, safety checks, or deployment governance. See the
[autonomy guide](https://github.com/microsoft/agent-learning/blob/main/docs/autonomy-complexity.md)
for the criteria and complexity profiles.

## Put it where decisions matter

You do not need to redesign an entire agent to start. Find one recurring choice
whose results you can measure. These are candidate integrations, not claims of
validated deployments:

| Workflow | Decision to improve | Evidence to collect |
|---|---|---|
| Coding assistance | Exact search or semantic retrieval | Relevant results, task success, latency |
| Customer support | An approved workflow or specialist escalation | Verified resolution, correct escalation, user feedback |
| Document processing | One approved model or another | Field accuracy, schema validity, measured cost |

Keep the action set stable, define success before training, and begin under
supervision. Where exploration could have consequences, restrict the alternatives
and require approval.

For context-dependent behavior, the repository also includes a
[next-best-action example](https://github.com/microsoft/agent-learning/blob/main/examples/next_best_action.md)
that uses context features and a custom contextual learner. The built-in
REINFORCE learner operates on the non-contextual softmax policy; it does not
automatically learn context-dependent routing from arbitrary episode text.

Persistence can use an in-memory store, local JSON files, or Azure Cosmos DB.
Episodes, metric results, rewards, training runs, and policy snapshots provide
records for investigating policy changes. This is inspectability of the
decision layer, not an explanation of the foundation model's internal reasoning.
Captured inputs and outputs may contain sensitive data, so teams must also
decide what to record and how to manage access and retention.

For teams using Azure, the repository includes optional Bicep infrastructure
for a Linux VM, Cosmos DB, and storage, with private endpoints and managed
identity. Azure deployment is not required to try the SDK locally.

## The next challenge: prove it across workloads

The public issue backlog proposes further work on
[human calibration of rewards](https://github.com/microsoft/agent-learning/issues/2),
[convergence and stability benchmarks](https://github.com/microsoft/agent-learning/issues/3),
[cost and break-even analysis](https://github.com/microsoft/agent-learning/issues/4),
and [production governance](https://github.com/microsoft/agent-learning/issues/5).
Other proposals explore
[Microsoft Agent Framework alignment](https://github.com/microsoft/agent-learning/issues/26)
and [broader local model support](https://github.com/microsoft/agent-learning/issues/27).

These are proposals in the public backlog, not released capabilities or delivery
commitments. Avoiding foundation-model training does not establish that policy
learning will be cheaper or more effective for every workload. The next step is
evidence: reproducible comparisons, calibrated rewards, and operational controls.

That is also an invitation to contribute. A real task, a repeatable evaluation,
and an honest result are more useful than another sweeping claim.

## Make the next decision count

Start with one agent. One recurring decision. Two or more executable
alternatives. A success criterion you can actually observe.

Run the example. Connect the decision lifecycle under supervision. Collect
outcomes. Inspect the policy change. Check whether better rewards correspond to
better results on your task.

Explore the [repository](https://github.com/microsoft/agent-learning), try the
[Python package](https://pypi.org/project/agent-learning/), and bring back a
reproducible example or evaluation.

The question is no longer only, "Can this agent complete the task?"

**It is, "What will its next decision learn from this one?"**

Your agent already produces experience. Give that experience a job.