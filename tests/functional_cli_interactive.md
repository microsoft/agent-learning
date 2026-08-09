# Functional CLI Test Analysis

This document explains the patient-triage functional workflow implemented by
[functional_cli_interactive.py](functional_cli_interactive.py) and
[functional_cli_batch.py](functional_cli_batch.py). The scenario is illustrative
test data, not clinical guidance.

## Purpose

The two scripts test the SDK and CLI above the unit-test level by invoking the
CLI in separate Python processes against a shared local file store.

The interactive phase verifies that an agent can:

1. Initialize a task policy from an action fixture.
2. Sample an action from the active policy.
3. Register a complete task episode through the CLI.
4. Store intent, execution outcome, metrics, and an aggregate reward.
5. Repeat the task many times without updating the policy during execution.

The batch phase verifies that another process can discover the stored agent and
task, inspect the episodes, train a new policy snapshot, and observe a changed
action distribution.

## Scenario And Actions

The agent is `triage-nurse`, and the task is `sore-throat-triage`. The patient
has a sore throat and painful swallowing without breathing difficulty. The
fixture
[next_best_action_patient_care_actions.json](../examples/next_best_action_patient_care_actions.json)
marks `order_strep_test` as the one correct action among four choices.

The interactive script initializes the policy with:

```text
agent-learn task-policy-init \
  --agent-id triage-nurse \
  --task-id sore-throat-triage \
  --actions examples/next_best_action_patient_care_actions.json
```

All four initial logits are zero. Softmax therefore assigns every action the
same probability:

$$
P(a_i)=\frac{e^0}{4e^0}=\frac{1}{4}=0.25
$$

## Interactive Episode Capture

The default run uses 32 episodes and random seed 7. The policy remains fixed at
version 0 while these episodes are generated, so every decision is sampled from
the initial uniform distribution.

For each episode, the script:

1. Calls `policy.choose()` and records the selected action and its log
   probability.
2. Treats `order_strep_test` as completed and every other action as failed.
3. Writes the complete episode to a temporary JSON file.
4. invokes `agent-learn task-episode-register` in a child process.
5. Reads the episode back from the shared local store.
6. Writes three deterministic metric results and the shaped reward through the
   SDK.

With seed 7, the 32 choices are:

| Action | Times selected | Outcome | Aggregate reward |
| --- | ---: | --- | ---: |
| `order_strep_test` | 12 | Completed | +0.8 |
| `recommend_supportive_care` | 8 | Failed | -0.8 |
| `prescribe_antibiotics_without_test` | 7 | Failed | -0.8 |
| `refer_to_emergency_department` | 5 | Failed | -0.8 |

Each episode receives the same quality value for intent resolution, task
adherence, and task completion. A completed episode receives normalized quality
1.0 for all three metrics; a failed episode receives 0.0.

The reward shaper maps normalized quality $q_m$ from $[0,1]$ to the signed value
$2q_m-1$. With the default metric weights $0.10$, $0.20$, and $0.50$, the
aggregate reward is:

$$
R=\sum_m w_m(2q_m-1)
$$

Consequently, all-success scores produce $+0.8$, and all-failure scores produce
$-0.8$. The mean reward across the deterministic batch is:

$$
\bar R=\frac{12(0.8)+20(-0.8)}{32}=-0.2
$$

## Batch Policy Update

The batch script executes these CLI operations:

```text
agent-learn list
agent-learn tasks-list triage-nurse
agent-learn task-episodes-count triage-nurse --task-id sore-throat-triage
agent-learn task-episodes-list triage-nurse --task-id sore-throat-triage --limit 500 --include-incomplete
agent-learn train --agent-id triage-nurse --task-id sore-throat-triage --limit 500 --skip-scoring
agent-learn task-policy --agent-id triage-nurse --task-id sore-throat-triage
```

`--skip-scoring` is intentional because the interactive phase has already
stored deterministic metric results and rewards.

The learner applies the REINFORCE softmax gradient from
[reinforce.py](../src/agent_learning/learners/reinforce.py). For action $i$:

$$
\Delta z_i=\frac{\eta}{N}\sum_{t=1}^{N}
(R_t-b)\left(\mathbb{1}[a_t=i]-\pi_i\right)
$$

For this first batch:

- Learning rate $\eta=0.05$.
- Episode count $N=32$.
- Baseline before training $b=0$.
- Initial action probability $\pi_i=0.25$.
- Importance-sampling weight is 1 because the behavior and target policies are
  both the initial uniform policy.
- The entropy-gradient term is zero because the initial policy already has
  maximum entropy.

The total reward is $32(-0.2)=-6.4$. The update can therefore be written as:

$$
\Delta z_i=\frac{0.05}{32}
\left(\sum_{t:a_t=i}R_t-0.25(-6.4)\right)
$$

This reproduces the stored logits exactly:

| Action | Reward from episodes selecting action | Logit update |
| --- | ---: | ---: |
| `order_strep_test` | $12(0.8)=9.6$ | $+0.0175$ |
| `recommend_supportive_care` | $8(-0.8)=-6.4$ | $-0.0075$ |
| `prescribe_antibiotics_without_test` | $7(-0.8)=-5.6$ | $-0.00625$ |
| `refer_to_emergency_department` | $5(-0.8)=-4.0$ | $-0.00375$ |

For example, the correct-action update is:

$$
\Delta z_{strep}=\frac{0.05}{32}\left(9.6-0.25(-6.4)\right)
=\frac{0.05}{32}(11.2)=0.0175
$$

The updates sum to zero, as expected for the softmax gradient. Only relative
logit differences affect the action probabilities. The value baseline also
moves from 0 to:

$$
b_{new}=0.9(0)+0.1(-0.2)=-0.02
$$

## Logits To Probabilities

The policy implementation in
[softmax_bandit.py](../src/agent_learning/policy/softmax_bandit.py) converts
logits to probabilities with softmax:

$$
P(a_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The implementation subtracts the maximum logit before exponentiation for
numerical stability. Subtracting the same constant from every logit does not
change the resulting probabilities.

After one training batch:

| Action | Logit | $e^{z_i}$ | Probability |
| --- | ---: | ---: | ---: |
| `order_strep_test` | 0.01750 | 1.01765 | **0.25440** |
| `recommend_supportive_care` | -0.00750 | 0.99253 | 0.24812 |
| `prescribe_antibiotics_without_test` | -0.00625 | 0.99377 | 0.24843 |
| `refer_to_emergency_department` | -0.00375 | 0.99626 | 0.24905 |

The exponential values sum to approximately $4.00021$, so the correct-action
probability is:

$$
P(\text{order strep test})=\frac{1.01765}{4.00021}\approx0.2544
$$

The probability therefore moves from `0.2500` to `0.2544`, an absolute increase
of about 0.44 percentage points. The change is intentionally modest because it
comes from one averaged update with a learning rate of 0.05.

A negative logit does not mean a negative probability. Exponentiation always
produces a positive value, and normalization makes all probabilities sum to 1.

## Learning Goal And Convergence

The policy starts uniformly because initialization supplies no evidence that
one action is better than another. `SoftmaxPolicy.from_actions()` assigns every
action a zero logit by default. With four actions, equal logits produce four
equal probabilities of 0.25. This is an unbiased starting point rather than a
statement that every action is clinically equivalent.

The learning objective is to maximize expected aggregate reward, not to make a
particular logit or probability large in isolation. Training increases the
correct action's logit relative to competing logits and decreases logits for
actions correlated with lower rewards.

For this deterministic, single-context functional scenario,
`order_strep_test` always receives the positive reward. It should therefore
become the dominant action as fresh evidence accumulates. Softmax cannot produce
an exact probability of 1 for finite logits, but it can approach 1 as the target
logit moves farther above the alternatives.

If the three competing actions have the same logit and the target action is
higher by the logit gap $d$, then:

$$
P(\text{target})=\frac{e^d}{e^d+3}
$$

Solving for the required gap gives:

$$
d=\ln\left(\frac{3P(\text{target})}{1-P(\text{target})}\right)
$$

| Desired target probability | Required logit advantage $d$ |
| ---: | ---: |
| 50% | 1.10 |
| 80% | 2.48 |
| 90% | 3.30 |
| 95% | 4.04 |
| 99% | 5.69 |

The current target logit is only slightly above the alternatives, so one
conservative update changes its probability from 25.00% to 25.44%. This test
uses that increase to verify the direction of learning; it is not a convergence
test.

For this synthetic deterministic case, a separate convergence test could use a
threshold such as 90% or 95%, while also asserting that `order_strep_test` is
the greedy action. A sound iterative workflow would:

1. Execute new episodes using the latest active policy.
2. Record the policy ID, version, and behavior log probability with each
  decision.
3. Score the new outcomes.
4. Train on the newly collected batch.
5. Repeat until held-out behavior and the target probability meet the chosen
  threshold.

Repeatedly training on the same 32 episodes merely to force the probability
upward would reuse old evidence and can overfit the policy. New on-policy
episodes provide a more meaningful convergence signal.

A probability near 100% is not a universal production goal. Exploration may be
valuable when rewards are noisy or the environment changes. If different
patient contexts require different actions, a contextual policy should learn a
separate action distribution from patient features rather than driving one
global action toward 100%. If one action is known with certainty to be mandatory
for every instance, a deterministic rule is more appropriate than learning that
rule through repeated trials.

## Persisted Evidence

The local store is generated under
[`data/functional-tests/patient-care-cli`](../data/functional-tests/patient-care-cli)
and contains:

- Versioned policy snapshots with actions, logits, baseline, and episode count.
- An active-policy pointer identifying the latest snapshot.
- Complete episode documents with intent and execution results.
- Metric and reward documents for every episode.
- A training-run document with mean reward, logit deltas, and policy version.

The snapshots store logits rather than calculated action probabilities. The
`task-policy` CLI command computes probabilities from each snapshot, and the
batch script compares the previous and current values. The exact text
`0.2500 -> 0.2544` is printed to the terminal but is not persisted as a field.

The default functional store is removed and recreated whenever the interactive
script runs, so generated UUIDs are not stable between runs.

## What This Test Establishes

The workflow demonstrates that:

- Independent CLI processes share durable state through the local backend.
- Task policy initialization and episode registration work through real CLI
  argument parsing and JSON serialization.
- Full intent and completion data survives persistence.
- Existing rewards are consumed by offline training.
- Training retains policy history and activates a new version.
- Rewarded behavior becomes the most probable action.

The test does not validate clinical correctness, a production model, Azure
credentials, or live LLM scoring. It uses a deterministic functional simulator
and a non-contextual softmax bandit for one fixed scenario.

## Reproduction

Run the scripts in order from the repository root:

```powershell
python tests/functional_cli_interactive.py
python tests/functional_cli_batch.py
```

With the default seed and configuration, the expected final summary is:

```text
Episodes trained: 32
Policy version: 0 -> 1
Correct-action probability: 0.2500 -> 0.2544
```