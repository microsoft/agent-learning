---
title: Agent Learning and Experiential Reinforcement Learning
description: Equation-level comparison of the agent-learning SDK with the Experiential Reinforcement Learning algorithm, including the relationship to SFT and RLVR shown in the paper's conceptual figure.
author: Microsoft
ms.date: 2026-08-07
ms.topic: concept
keywords:
  - experiential reinforcement learning
  - ERL
  - RLVR
  - REINFORCE
  - policy gradient
  - self-reflection
  - distillation
  - contextual bandit
estimated_reading_time: 14
---

## Executive conclusion

The agent-learning SDK and Experiential Reinforcement Learning (ERL) share the
same mathematical nucleus: a reward-weighted score-function gradient of the
form

$$
\mathbb{E}\!\left[A\,\nabla \log \pi(\text{sample}\mid\text{context})\right].
$$

They do not, however, implement the same algorithm.

The SDK is best classified as a lightweight, RLVR-like contextual-bandit layer
around an agent. It converts evaluation scores into one scalar reward and uses
REINFORCE to change a small distribution over discrete action IDs. The language
model remains frozen. In the paper, the policy is the language model itself,
the sampled objects are token sequences, and training adds a gated
experience-reflection-retry trajectory plus a supervised or KL-based
internalization objective.

Consequently, the SDK corresponds most closely to the **middle RLVR panel** in
the pasted figure, with two qualifications:

1. its policy is a small action selector rather than the generative model; and
2. its reward may come from learned evaluators, so it is not necessarily
   *verifiable* reward in the strict RLVR sense.

Persisting updated logits makes the SDK's action preferences durable, but that
is not ERL internalization. ERL specifically transfers behavior from a richer
reflection-conditioned policy into a deployment policy that must act from the
original input alone.

## Scope

This comparison uses:

- the implemented SDK paths in
  [reinforce.py](../src/agent_learning/learners/reinforce.py),
  [softmax_bandit.py](../src/agent_learning/policy/softmax_bandit.py),
  [contextual_softmax.py](../src/agent_learning/policy/contextual_softmax.py),
  [shaping.py](../src/agent_learning/rewards/shaping.py), and
  [runner.py](../src/agent_learning/training/runner.py);
- the example-only contextual learner in
  [next_best_action.py](../examples/next_best_action.py);
- the SDK's consolidated [mathematical reference](math.md); and
- version 1 of Shi et al., *Experiential Reinforcement Learning*, including its
  main algorithm, Appendix A, and training configuration.

The distinction between production code and example code matters. The packaged
`ReinforceLearner` accepts only `SoftmaxPolicy`. The linear contextual update is
currently demonstrated by `ContextualReinforceLearner` inside an example; it is
not a learner exported by the SDK.

## Reading the pasted figure

The three panels describe progressively richer sources of supervision.

| Figure panel | Mathematical signal | What is learned |
| --- | --- | --- |
| Direct learning (SFT) | A target response $y^*$ | Imitate a fixed example from $x$ |
| Reinforcement learning (RLVR) | A scalar outcome $r$ | Increase probability of rewarded sampled behavior |
| Experiential learning (ERL) | Outcome, textual feedback, reflection $\Delta$, and a successful retry | Revise behavior within an episode, then reproduce the revision without $\Delta$ |

The arrows in the figure are conceptual. For example,
$\pi_\theta(\cdot\mid x)\leftarrow r$ does not assign a scalar to a policy.
The reward changes parameters through an advantage-weighted policy gradient.
Likewise,
$\pi_\theta(\cdot\mid x)\leftarrow\pi_\theta(\cdot\mid x,\Delta)$ denotes
distillation from a reflection-conditioned teacher behavior into the
reflection-free deployment behavior.

The SDK currently has no SFT stage for its control policy and no ERL
reflection/internalization stage. Its `evaluate -> shape -> learn` loop is the
middle panel: action, environment outcome, scalar reward, policy update.

## The SDK's implemented mathematics

### Policy parameterization

For $K$ discrete actions, the production policy stores one scalar logit per
action:

$$
p_k = \pi_z(k)
= \frac{\exp(z_k-z_{\max})}
       {\sum_{j=1}^{K}\exp(z_j-z_{\max})}.
$$

The contextual policy instead stores $W\in\mathbb{R}^{K\times d}$ and uses

$$
z=W\phi, \qquad
p_k=\pi_W(k\mid\phi)=\operatorname{softmax}(W\phi)_k.
$$

In both cases the stochastic object is one categorical action ID. The policy
does not assign probabilities to the agent's response tokens.

### Reward construction

The SDK normalizes each metric to $s_m\in[0,1]$, maps it to a signed value, adds
operational terms, and clamps the result:

$$
R = \operatorname{clip}\!\left(
  \sum_m w_m(2s_m-1)+\sum_j c_j,
    -1,
    1
\right).
$$

Here $w_m$ is a metric weight and $c_j$ is an additive operational
adjustment, such as the latency penalty, route-correctness reward, wrong-route
penalty, or hallucinated-class penalty.

Intent resolution, task adherence, and task completion can therefore provide a
denser signal than the sparse terminal rewards used in the paper's FrozenLake
and Sokoban experiments. The learner receives only the aggregate $R$; it does
not consume textual evaluator feedback as a corrective context.

### Production REINFORCE update

For each usable episode $i$, let $a_i$ be its selected action, let $b$ be the
pre-update EMA baseline, and define the clipped behavior-policy correction

$$
\rho_i = \min\!\left(
  \frac{\pi_{\text{current}}(a_i)}
       {\max(\pi_{\text{behavior},i}(a_i),\epsilon)},
  c
\right),
\qquad \epsilon=10^{-12}.
$$

The code obtains the behavior probability from the stored action log
probability whenever one is present. Its effective advantage is

$$
A_i=\rho_i(R_i-b).
$$

With current probabilities held fixed for the batch, the exact implemented
logit step is

$$
\Delta z_k
=\frac{\eta}{N}\sum_{i=1}^{N}
\left[
A_i\big(\mathbb{1}[a_i=k]-p_k\big)
+\beta\big(-\log\max(p_k,\epsilon)-H(p)\big)
\right],
$$

where

$$
H(p)=-\sum_j p_j\log\max(p_j,\epsilon).
$$

After the policy update, the scalar baseline becomes

$$
b' = \lambda b + (1-\lambda)\bar R.
$$

The default values are $\eta=0.05$, $\lambda=0.9$, $\beta=0.01$, and
$c=5$.

> [!NOTE]
> The implementation calls the exploration term an entropy gradient, but the
> exact softmax derivative of Shannon entropy is
> $p_k(-\log p_k-H(p))$. The code uses $-\log p_k-H(p)$ without the factor
> $p_k$. The implemented term is therefore entropy-inspired rather than the
> literal gradient $\partial H/\partial z_k$. This observation applies equally
> to the formula currently shown in [math.md](math.md).

### Contextual example update

The example replaces the global EMA advantage used in the production learner
with a within-batch mean for gradient calculation:

$$
A_i=\rho_i(R_i-\bar R).
$$

Its action-row update is

$$
\Delta W_k
=\frac{\eta}{N}\sum_{i=1}^{N}
\left[
A_i\big(\mathbb{1}[a_i=k]-p_{i,k}\big)
+\beta\big(-\log p_{i,k}-H(p_i)\big)
\right]\phi_i.
$$

This centering resembles the variance-reduction intuition of group-relative
advantages, but it is not GRPO: there is no per-prompt group of language-model
rollouts, token-level likelihood ratio, PPO clipped surrogate, reference-model
KL, or group standard-deviation normalization.

### The remaining SDK math is auxiliary to policy learning

The logistic-regression, hashing, TF-IDF, cosine-similarity, router, and metric-
normalization equations in [math.md](math.md) do not have direct ERL
counterparts. They sit on either side of the policy-gradient update:

| SDK mathematics | Role in this SDK | Closest ERL role |
| --- | --- | --- |
| Router logistic regression or cosine prototypes | Maps context to a class or refusal before/around policy use | Part of the agent or environment interface, not specified by ERL |
| Binary logistic-regression scorers | Predict intent, adherence, or completion quality | A learned reward/evaluation component |
| Hashing-trick and TF-IDF features | Turn text into scorer inputs | No analogue; ERL's reflection is generated text, not a feature transform |
| Metric normalization | Maps heterogeneous scorer outputs to $[0,1]$ | Environment-specific reward calibration |
| Reward shaping | Produces the scalar $R$ consumed by REINFORCE | The paper's abstract environment reward $r$ |

ERL deliberately treats the environment's reward function as external to its
main contribution. Its experiments use task-specific verification: terminal
success for the grid worlds and answer F1 for HotpotQA. The SDK spends more of
its mathematical surface on constructing and auditing reward, while ERL spends
more on transforming feedback into a new training trajectory. A TF-IDF or LLM
scorer may explain whether an output is good, but merely running that scorer
does not create ERL-style self-reflection unless its feedback is fed into a
reflection-conditioned retry and later internalized.

## The paper's ERL mathematics

ERL uses the language model itself as the trainable policy. One training cycle
contains the following random variables.

### First attempt and environment feedback

$$
y^{(1)}\sim\pi_\theta(\cdot\mid x),
\qquad
(f^{(1)},r^{(1)})=\operatorname{Env}(y^{(1)}).
$$

In the full algorithm, reflection is gated to failed or suboptimal first
attempts:

$$
r^{(1)}<\tau,
\qquad \tau=1\text{ in the reported experiments}.
$$

### Reflection and retry

When the gate opens, the same model samples a textual reflection using the
attempt, textual feedback, scalar reward, and cross-episode memory:

$$
\Delta\sim\pi_\theta(
  \cdot\mid x,y^{(1)},f^{(1)},r^{(1)},m
).
$$

It then samples and evaluates a refined answer:

$$
y^{(2)}\sim\pi_\theta(\cdot\mid x,\Delta),
\qquad
(f^{(2)},r^{(2)})=\operatorname{Env}(y^{(2)}).
$$

The reflection receives the downstream reward

$$
\tilde r=r^{(2)},
$$

so credit reaches the reflection through the quality of the behavior it
induces, not through a separately labeled reflection target.

### Policy-gradient objective

The paper writes a common objective for the first attempt, reflection, and
second attempt:

$$
\mathcal L_{\mathrm{policy}}(\theta)
=-\mathbb E\!\left[
  A\log\pi_\theta(y\mid x,\cdot)
\right].
$$

At the method level, $A$ is left as the advantage associated with each output.
The experiments instantiate the optimizer with GRPO and add actor clipping, KL
regularization, and importance sampling. The reported entropy coefficient is
zero.

Because $y=(y_1,\ldots,y_T)$ is an autoregressive sequence,

$$
\log\pi_\theta(y\mid c)
=\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid c,y_{<t}).
$$

One trajectory-level reward therefore updates the likelihood of every sampled
token in an attempt or reflection. This is a much larger credit-assignment
surface than the SDK's single categorical action.

### Selective internalization

ERL then removes reflection from the conditioning context and imitates only
successful second attempts:

$$
\mathcal L_{\mathrm{distill}}(\theta)
=-\mathbb E\!\left[
  \mathbb{1}[r^{(2)}>0]
  \log\pi_\theta(y^{(2)}\mid x)
\right].
$$

This is the mathematical meaning of the figure's experience-internalization
arrow. The richer policy produced $y^{(2)}$ with access to $\Delta$; the
deployment policy is trained to reproduce that behavior from $x$ alone.

Appendix A also proposes an on-policy alternative:

$$
\mathcal L_{\mathrm{OD}}(\theta)
=\mathbb E_{x\sim\mathcal D}\!\left[
\mathbb{1}[r^{(2)}>0]
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\operatorname{KL}\!\left(
\pi_\theta(\cdot\mid x,\Delta)
\,\|\,
\pi_\theta(\cdot\mid x)
\right)
\right]
\right].
$$

## Where the mathematics agrees

### The same score-function identity

Both methods rely on

$$
\nabla_\vartheta\,
\mathbb E_{u\sim\pi_\vartheta}[R(u)]
=\mathbb E_{u\sim\pi_\vartheta}\!\left[
  R(u)\nabla_\vartheta\log\pi_\vartheta(u)
\right].
$$

For the SDK's categorical softmax,

$$
\frac{\partial\log\pi(a)}{\partial z_k}
=\mathbb{1}[a=k]-\pi(k),
$$

which gives the closed-form update in `ReinforceLearner`. ERL applies the same
identity through backpropagation over an autoregressive model. The difference is
the parameterization and trajectory, not the foundational estimator.

### Baselines and relative credit

Both reduce variance by replacing raw reward with an advantage. The production
SDK uses a scalar EMA baseline; the contextual example centers on its batch
mean; the paper's experiments use group-relative optimization. In all cases,
adding a baseline independent of the sampled action aims to preserve the
expected policy-gradient direction while reducing variance.

### Off-policy stabilization

Both recognize that stored or internally generated trajectories may not exactly
match the current deployment policy. The SDK caps
$\pi_{\mathrm{current}}/\pi_{\mathrm{behavior}}$ for its categorical action.
The ERL experiments combine importance sampling with actor clipping and a
reference-policy KL penalty. ERL's need is stronger because reflection-guided
second attempts and distilled targets are explicitly generated under richer
conditioning than deployment receives.

### No extra inference loop after learning

Both can deploy a single-pass policy after training. The SDK simply loads the
updated action-policy snapshot. ERL uses distillation so that reflection and a
second attempt are training-time scaffolding rather than mandatory inference
cost.

## Where the mathematics differs

| Dimension | agent-learning SDK | ERL paper |
| --- | --- | --- |
| Trainable parameters | $K$ logits, or $K\times d$ example weights | Billions of language-model parameters $\theta$ |
| Sampled object | One discrete action ID | First-response, reflection, and retry token sequences |
| Model weights | Frozen | Updated by RL and distillation |
| Reward | Shaped, signed, multi-metric, clamped to $[-1,1]$ | Environment reward, often sparse and nonnegative |
| Textual feedback | May be captured or used by scorers, but is discarded by the learner | Conditions reflection directly |
| Within-episode correction | None | Gated reflection and second attempt |
| Reflection credit | None | Retry reward $r^{(2)}$ is assigned to $\Delta$ |
| Consolidation | Persist action-policy parameters | Distill $\pi_\theta(\cdot\mid x,\Delta)$ behavior into $\pi_\theta(\cdot\mid x)$ |
| Memory | Episode/reward/policy persistence | Textual cross-episode reflection memory $m$ used in generation |
| Main optimizer | Closed-form categorical REINFORCE | GRPO in the experiments |
| Trust-region controls | Importance-ratio cap and parameter clipping | Importance sampling, PPO-style actor clipping, and reference KL |
| Exploration | Nonstandard entropy-inspired logit term | Sampling plus the reflection/retry data-generation process; entropy coefficient 0 in reported runs |

### Credit assignment is scalar versus structured

The SDK compresses all feedback before learning:

$$
(s_1,\ldots,s_M,\text{penalties})\longrightarrow R
\longrightarrow \Delta z.
$$

That is efficient and interpretable, but information about *why* an episode
failed disappears from the policy update. ERL retains two channels:

$$
\text{outcome}\longrightarrow r
\quad\text{and}\quad
(y^{(1)},f^{(1)},r^{(1)},m)\longrightarrow\Delta.
$$

The scalar channel still supplies the optimization direction, while the
reflection channel changes the next sample's conditioning context. ERL is thus
not a new replacement for policy gradients; it is a structured data-generation
and consolidation procedure around them.

### Persistence is not internalization

An SDK snapshot remembers that action $a$ has become more probable. It does not
learn a reusable explanation of the failure, revise the underlying agent's
reasoning, or teach the frozen language model to emit a corrected response.

ERL internalization is a conditional distribution-matching operation. It asks
the reflection-free policy to match behavior that was only discovered with
extra information. This is absent even when the SDK stores full conversation
history and model output, because `ReinforceLearner` consumes only action ID,
action log probability, and aggregate reward.

### The two methods solve different-sized control problems

The SDK deliberately restricts learning to an interpretable decision boundary:
which prompt, route, tool strategy, or other registered action should be chosen?
It can learn rapidly on CPU and cannot alter linguistic capabilities beyond
choosing among those actions.

ERL modifies the generative policy. It can internalize new reasoning behavior,
but requires token log probabilities, a trainable actor, a reference model,
large optimizer state, rollout infrastructure, and substantially more compute.
The paper's experiments use eight H100 GPUs; this is not an incidental
implementation difference but a consequence of optimizing a different
parameter space.

## Paper-v1 threshold caveat

Version 1 states all three of the following:

1. reflection is triggered when $r^{(1)}<\tau$;
2. $\tau=1$ in the experiments; and
3. reflection memory is updated when $r^{(2)}>\tau$.

The described task rewards have maximum value 1. Taken literally, the strict
memory condition $r^{(2)}>1$ cannot be satisfied. This condition appears in the
paper's LaTeX source as well as the rendered HTML, while the ablation section
reports effects from enabled memory. The conceptual role of memory is clear,
but reproducing the reported memory mechanism requires clarification of the
implemented threshold or comparison operator. This analysis does not assume an
undocumented correction.

## What an ERL-inspired SDK extension would require

There are two materially different implementation targets.

### Option 1: ERL-like learning for the small action policy

The SDK could preserve a frozen language model and internalize successful
reflection-guided *action choices* into its outer policy:

1. sample $a^{(1)}\sim\pi_\psi(\cdot\mid\phi)$ and evaluate $R^{(1)}$;
2. when $R^{(1)}<\tau$, generate a textual reflection from the captured episode
   and metric evidence;
3. encode that reflection as additional context and sample
   $a^{(2)}\sim\pi_\psi(\cdot\mid\phi,\Delta)$;
4. evaluate $R^{(2)}$, persist the linked attempt-reflection-retry record, and
   update the action policy; and
5. imitate a successful second action without reflection:

$$
\mathcal L_{\mathrm{selector\text{-}distill}}(\psi)
=-\mathbb E\!\left[
  \mathbb{1}[R^{(2)}>0]
  \log\pi_\psi(a^{(2)}\mid\phi)
\right].
$$

A distributional version would define a reflection-conditioned teacher $q$
and reflection-free student $p$:

$$
\mathcal L_{\mathrm{KL}}(\psi)
=\mathbb{1}[R^{(2)}>0]
\sum_k q_k\log\frac{q_k}{p_k},
\qquad
\frac{\partial\mathcal L_{\mathrm{KL}}}{\partial z_k}=p_k-q_k.
$$

This would be a useful finite-action analogue of ERL, but it would not be the
paper's algorithm because it still would not train the language model or
internalize response-token behavior. It would also require moving contextual
learning from the example into a supported learner and defining a reflection
encoder.

### Option 2: Full paper-style ERL

A faithful implementation would use the SDK as an orchestration and lineage
layer around an external LM-training stack. It would need:

- token-level log probabilities for $y^{(1)}$, $\Delta$, and $y^{(2)}$;
- linked first-attempt, reflection, and second-attempt trajectory types;
- textual environment feedback rather than only aggregate reward;
- a gated retry controller and reflection-memory policy;
- GRPO/PPO-style actor updates with a reference policy and KL controls;
- selective SFT or on-policy KL distillation; and
- separate evaluation of first-pass deployment behavior, retry behavior, and
  reflection quality.

The current `Episode` schema can persist much of the text as metadata, but the
current learner API and one-action log probability are not sufficient to
compute the paper's token-level objectives.

## Practical judgment

ERL is most relevant when failures contain reusable structure that a language
model can verbalize, especially under sparse rewards and long-horizon dynamics.
The paper's ablations generally show a larger loss from removing structured
reflection than from removing memory, while one Sokoban setting improves when
memory is removed. That supports treating reflection memory as fallible learned
state, not as automatically beneficial context.

The current SDK is a better fit when the desired adaptation is deliberately
narrow: select among a stable set of actions, preserve model immutability, train
locally, retain auditability, and avoid GPU fine-tuning. ERL is not a drop-in
replacement for that design. It is a potential higher-cost layer for tasks where
changing only action-selection probabilities cannot express the needed
correction.

The most defensible synthesis is therefore:

> Keep the SDK's shaped-reward contextual bandit as the outer, auditable control
> policy. Add a gated reflection/retry data path only for low-reward episodes,
> and choose explicitly whether to distill successful retries into the small
> selector or into a trainable language model. Those two destinations have
> similar-looking log-likelihood equations but very different capabilities,
> costs, and operational risks.

## Sources

- Taiwei Shi et al., [*Experiential Reinforcement Learning*, arXiv:2602.13949v1](https://arxiv.org/abs/2602.13949v1), especially Section 2, Algorithm 1, Appendix A, and Appendix C.
- [SDK mathematical reference](math.md).
- [Production REINFORCE learner](../src/agent_learning/learners/reinforce.py).
- [Marginal softmax policy](../src/agent_learning/policy/softmax_bandit.py).
- [Contextual softmax policy](../src/agent_learning/policy/contextual_softmax.py).
- [Contextual REINFORCE example](../examples/next_best_action.py).
- [Reward shaper](../src/agent_learning/rewards/shaping.py).

## Citation

```bibtex
@misc{shi2026experientialreinforcementlearning,
  title={Experiential Reinforcement Learning},
      author={Taiwei Shi and Sihao Chen and Bowen Jiang and Linxin Song and Longqi Yang and Jieyu Zhao},
      year={2026},
      eprint={2602.13949},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.13949},
}
```