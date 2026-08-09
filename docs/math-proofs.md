---
title: Mathematical reference for the agent-learning SDK
description: Consolidated linear-algebra and probability formulas with proofs behind the policies, REINFORCE learner, reward shaper, logistic-regression classifiers, router, and scorers of the agent-learning SDK.
author: Microsoft
ms.date: 2026-07-27
ms.topic: reference
keywords:
  - reinforcement learning
  - policy gradient
  - REINFORCE
  - softmax
  - logistic regression
  - reward shaping
  - cosine similarity
  - linear algebra
estimated_reading_time: 35
---

# Mathematical reference for the agent-learning SDK

## Overview

This document collects the linear-algebra and probability identities used by the agent-learning SDK and adds derivations for the formulas used by the policy, learner, reward shaper, router, logistic-regression classifiers, and text scorers.

The SDK design assumes a small, auditable policy over discrete actions rather than direct model-weight fine-tuning. In that setting, the key mathematical object is a probability distribution over actions. Learning adjusts either action logits directly or a contextual linear map whose output becomes action logits.

All vectors are column vectors unless stated otherwise. All softmax computations are assumed to use the max-subtraction trick for numerical stability.

## Notation

| Symbol | Meaning |
| --- | --- |
| $K$ | Number of discrete actions or classes. |
| $d$ | Feature dimension of a context vector. |
| $n$ | Feature dimension for a classifier vector, usually including a bias column. |
| $a$ | The action actually sampled by the policy. |
| $k$ | A generic action or class index. |
| $\phi \in \mathbb{R}^{d}$ | Context feature vector. |
| $x \in \mathbb{R}^{n}$ | Classifier feature vector, often $x = [\phi, 1]$ or $x = [\phi \Vert \mathbf{onehot}(a) \Vert 1]$. |
| $z \in \mathbb{R}^{K}$ | Action logits or class logits. |
| $W \in \mathbb{R}^{K \times d}$ | Contextual policy weight matrix. |
| $W_k$ | Row $k$ of $W$. |
| $w \in \mathbb{R}^{n}$ | Binary logistic-regression weight vector. |
| $w_k \in \mathbb{R}^{n}$ | Multiclass logistic-regression row for class $k$. |
| $\pi(a)$ | Policy probability of action $a$. |
| $\pi(a \mid \phi)$ | Contextual policy probability of action $a$ given context $\phi$. |
| $R$ | Aggregate episode reward, typically clipped to $[-1, 1]$. |
| $b$ | Value baseline. |
| $A$ | Advantage, $A = R - b$. |
| $\eta$ | Learning rate. |
| $\beta$ | Entropy regularization coefficient. |
| $\gamma$ | L2 weight-decay coefficient. |
| $\mathbb{1}[\cdot]$ | Indicator function. |

---

# 1. Linear-algebra conventions

## 1.1 Dot product

For $u, v \in \mathbb{R}^{d}$,

$$
u \cdot v = u^{\top}v = \sum_{j=1}^{d} u_j v_j.
$$

The dot product is bilinear:

$$
(\alpha u + \beta r) \cdot v = \alpha(u \cdot v) + \beta(r \cdot v),
$$

and symmetric:

$$
u \cdot v = v \cdot u.
$$

### Proof

By definition,

$$
(\alpha u + \beta r) \cdot v
= \sum_{j=1}^{d}(\alpha u_j + \beta r_j)v_j
= \alpha \sum_{j=1}^{d}u_jv_j + \beta \sum_{j=1}^{d}r_jv_j
= \alpha(u \cdot v) + \beta(r \cdot v).
$$

Symmetry follows because scalar multiplication is commutative:

$$
u \cdot v = \sum_{j=1}^{d}u_jv_j = \sum_{j=1}^{d}v_ju_j = v \cdot u.
$$

## 1.2 Euclidean norm

The Euclidean norm is

$$
\lVert u \rVert_2 = \sqrt{u^{\top}u} = \sqrt{\sum_{j=1}^{d}u_j^2}.
$$

It is nonnegative and equals zero if and only if $u = \mathbf{0}$.

### Proof

Each $u_j^2 \ge 0$, so $\sum_j u_j^2 \ge 0$ and its square root is nonnegative. If $\lVert u \rVert_2 = 0$, then $\sum_j u_j^2 = 0$. A sum of nonnegative terms can be zero only if every term is zero, so $u_j = 0$ for every $j$. Conversely, if $u = \mathbf{0}$, then every term is zero and the norm is zero.

## 1.3 Matrix-vector product

For $W \in \mathbb{R}^{K \times d}$ and $\phi \in \mathbb{R}^{d}$,

$$
z = W\phi, \qquad z_k = W_k \phi = W_k \cdot \phi = \sum_{j=1}^{d} W_{k,j}\phi_j.
$$

This is the contextual policy's linear logit map.

### Proof

The definition of matrix-vector multiplication gives the $k$th coordinate of $W\phi$ as the dot product of row $k$ with $\phi$:

$$
(W\phi)_k = \sum_{j=1}^{d}W_{k,j}\phi_j.
$$

Thus each row independently contributes one scalar logit.

---

# 2. Softmax policies

## 2.1 Stable softmax

Given logits $z \in \mathbb{R}^{K}$, the softmax distribution is

$$
\pi(k) = \frac{\exp(z_k)}{\sum_{c=1}^{K}\exp(z_c)}.
$$

For numerical stability, compute it as

$$
\pi(k) = \frac{\exp(z_k - m)}{\sum_{c=1}^{K}\exp(z_c - m)},
\qquad m = \max_b z_b.
$$

### Proof of equivalence

Since $m$ is a scalar independent of $k$,

$$
\frac{\exp(z_k - m)}{\sum_c\exp(z_c - m)}
= \frac{\exp(z_k)\exp(-m)}{\sum_c\exp(z_c)\exp(-m)}
= \frac{\exp(z_k)\exp(-m)}{\exp(-m)\sum_c\exp(z_c)}
= \frac{\exp(z_k)}{\sum_c\exp(z_c)}.
$$

The probabilities are unchanged, but the largest exponent is now $\exp(0) = 1$, which avoids unnecessary overflow.

## 2.2 Softmax produces a valid probability distribution

For every $k$,

$$
\pi(k) > 0,
$$

and

$$
\sum_{k=1}^{K}\pi(k)=1.
$$

### Proof

Exponentials are strictly positive, so $\exp(z_k-m)>0$. The denominator is a sum of positive terms, so it is positive. Therefore each probability is positive. Also,

$$
\sum_{k=1}^{K}\pi(k)
= \sum_{k=1}^{K}\frac{\exp(z_k-m)}{\sum_c\exp(z_c-m)}
= \frac{\sum_k\exp(z_k-m)}{\sum_c\exp(z_c-m)}
= 1.
$$

## 2.3 Uniform distribution when contextual logits are zero

For a contextual linear-softmax policy,

$$
z = W\phi.
$$

If $\phi = \mathbf{0}$, then $z = \mathbf{0}$ and

$$
\pi(k \mid \phi)=\frac{1}{K}.
$$

### Proof

If $\phi = \mathbf{0}$, then for each row $W_k$,

$$
z_k = W_k\phi = W_k\mathbf{0} = 0.
$$

So every logit is zero. Softmax gives

$$
\pi(k) = \frac{\exp(0)}{\sum_{c=1}^{K}\exp(0)} = \frac{1}{K}.
$$

## 2.4 Softmax is invariant to adding a constant

For any scalar $c$,

$$
\operatorname{softmax}(z + c\mathbf{1})_k = \operatorname{softmax}(z)_k.
$$

### Proof

$$
\operatorname{softmax}(z+c\mathbf{1})_k
= \frac{\exp(z_k+c)}{\sum_j\exp(z_j+c)}
= \frac{\exp(c)\exp(z_k)}{\exp(c)\sum_j\exp(z_j)}
= \operatorname{softmax}(z)_k.
$$

This is why subtracting the maximum logit is mathematically safe.

## 2.5 Log-probability clamping

The policy returns

$$
\log \pi(a) = \log(\max(\pi(a), \epsilon)),
\qquad \epsilon = 10^{-12}.
$$

### Why this is used

Since probabilities may underflow in finite-precision arithmetic, a direct logarithm can produce $-\infty$. Replacing $\pi(a)$ with $\max(\pi(a), \epsilon)$ ensures the returned value is finite:

$$
\log(\max(\pi(a), \epsilon)) \ge \log(\epsilon).
$$

With $\epsilon=10^{-12}$,

$$
\log(\epsilon) = \log(10^{-12}) = -12\log(10).
$$

This is a numerical guard, not a change to the ideal mathematical softmax.

## 2.6 Weighted sampling by inverse CDF

Given probabilities $\pi(0),\dots,\pi(K-1)$ and $u \sim \mathcal{U}[0,1)$, the selected action is

$$
a^* = \min\left\{k : \sum_{j=0}^{k}\pi(j) \ge u\right\}.
$$

### Proof that action $k$ is sampled with probability $\pi(k)$

Define cumulative sums

$$
F_k = \sum_{j=0}^{k}\pi(j), \qquad F_{-1}=0.
$$

The rule selects $k$ exactly when

$$
F_{k-1} \le u < F_k.
$$

Because $u$ is uniform on $[0,1)$,

$$
\Pr(a^*=k) = \Pr(F_{k-1} \le u < F_k) = F_k - F_{k-1} = \pi(k).
$$

Thus inverse-CDF sampling realizes the intended categorical distribution.

---

# 3. Gradient of the softmax log-likelihood

The central identity behind REINFORCE for a softmax policy is

$$
\frac{\partial \log \pi(a)}{\partial z_k}
= \mathbb{1}[k=a] - \pi(k).
$$

## 3.1 Derivation

Start with

$$
\pi(a) = \frac{\exp(z_a)}{\sum_c\exp(z_c)}.
$$

Taking the logarithm,

$$
\log\pi(a) = z_a - \log\left(\sum_c\exp(z_c)\right).
$$

Differentiate with respect to $z_k$:

$$
\frac{\partial\log\pi(a)}{\partial z_k}
= \frac{\partial z_a}{\partial z_k}
- \frac{1}{\sum_c\exp(z_c)}\frac{\partial}{\partial z_k}\sum_c\exp(z_c).
$$

The first term is

$$
\frac{\partial z_a}{\partial z_k} = \mathbb{1}[k=a].
$$

The second derivative is

$$
\frac{\partial}{\partial z_k}\sum_c\exp(z_c) = \exp(z_k).
$$

Therefore

$$
\frac{\partial\log\pi(a)}{\partial z_k}
= \mathbb{1}[k=a] - \frac{\exp(z_k)}{\sum_c\exp(z_c)}
= \mathbb{1}[k=a] - \pi(k).
$$

## 3.2 Consequence: gradients sum to zero

$$
\sum_{k=1}^{K}\frac{\partial\log\pi(a)}{\partial z_k}=0.
$$

### Proof

$$
\sum_k(\mathbb{1}[k=a]-\pi(k))
= \sum_k\mathbb{1}[k=a] - \sum_k\pi(k)
= 1-1=0.
$$

This matters because increasing the taken action's logit necessarily redistributes probability mass from other actions.

---

# 4. REINFORCE with baseline

## 4.1 Objective

For a one-step episodic bandit-style policy, the expected reward is

$$
J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[R(a)].
$$

The score-function identity gives

$$
\nabla_\theta J(\theta)=\mathbb{E}_{a\sim\pi_\theta}\left[R(a)\nabla_\theta\log\pi_\theta(a)\right].
$$

### Proof of the score-function identity

Write the expectation as a finite sum:

$$
J(\theta)=\sum_a \pi_\theta(a)R(a).
$$

Assuming $R(a)$ is treated as observed reward and not directly differentiated through the policy parameters,

$$
\nabla_\theta J(\theta)=\sum_a R(a)\nabla_\theta\pi_\theta(a).
$$

Use

$$
\nabla_\theta\pi_\theta(a)=\pi_\theta(a)\nabla_\theta\log\pi_\theta(a),
$$

which follows from differentiating $\log\pi_\theta(a)$:

$$
\nabla_\theta\log\pi_\theta(a) = \frac{\nabla_\theta\pi_\theta(a)}{\pi_\theta(a)}.
$$

Then

$$
\nabla_\theta J(\theta)
= \sum_a \pi_\theta(a)R(a)\nabla_\theta\log\pi_\theta(a)
= \mathbb{E}_{a\sim\pi_\theta}\left[R(a)\nabla_\theta\log\pi_\theta(a)\right].
$$

## 4.2 Baseline does not bias the policy-gradient estimator

The baseline update uses advantage

$$
A = R-b.
$$

The gradient estimator becomes

$$
(R-b)\nabla_\theta\log\pi_\theta(a).
$$

If $b$ does not depend on the sampled action $a$, then subtracting it does not change the expected gradient.

### Proof

We need to show

$$
\mathbb{E}_{a\sim\pi_\theta}\left[b\nabla_\theta\log\pi_\theta(a)\right]=0.
$$

Since $b$ is independent of $a$,

$$
\mathbb{E}[b\nabla_\theta\log\pi_\theta(a)]
= b\sum_a\pi_\theta(a)\nabla_\theta\log\pi_\theta(a).
$$

Using the score identity inside the sum,

$$
\sum_a\pi_\theta(a)\nabla_\theta\log\pi_\theta(a)
= \sum_a\nabla_\theta\pi_\theta(a)
= \nabla_\theta\sum_a\pi_\theta(a)
= \nabla_\theta 1
= 0.
$$

Therefore

$$
\mathbb{E}[(R-b)\nabla_\theta\log\pi_\theta(a)]
= \mathbb{E}[R\nabla_\theta\log\pi_\theta(a)].
$$

The baseline can reduce variance without changing the expected gradient direction.

## 4.3 Marginal softmax REINFORCE update

Using the logit gradient from Section 3,

$$
\Delta z_k = \eta(R-b)(\mathbb{1}[k=a]-\pi(k)).
$$

### Derivation

Gradient ascent on expected reward uses

$$
z_k \leftarrow z_k + \eta(R-b)\frac{\partial\log\pi(a)}{\partial z_k}.
$$

Substitute

$$
\frac{\partial\log\pi(a)}{\partial z_k}=\mathbb{1}[k=a]-\pi(k).
$$

Thus

$$
\Delta z_k=\eta(R-b)(\mathbb{1}[k=a]-\pi(k)).
$$

## 4.4 Sign of the update

If $R>b$, then $A=R-b>0$ and the taken action's logit increases because

$$
\Delta z_a = \eta A(1-\pi(a)) \ge 0.
$$

All non-taken action logits decrease because for $k\ne a$,

$$
\Delta z_k = -\eta A\pi(k) \le 0.
$$

If $R<b$, the signs reverse.

### Proof

Since $0\le\pi(k)\le1$, for the selected action $a$,

$$
1-\pi(a)\ge0.
$$

For a non-selected action, $\mathbb{1}[k=a]=0$, so the term is $-\pi(k)\le0$. Multiplying by a positive advantage preserves signs; multiplying by a negative advantage reverses signs.

## 4.5 Batch averaging

For $N$ usable episodes, the average gradient is

$$
\Delta z_k = \frac{\eta}{N}\sum_{i=1}^{N}(R_i-b)(\mathbb{1}[k=a_i]-\pi_i(k)).
$$

### Proof

The empirical average approximates the expectation:

$$
\mathbb{E}[g]\approx \frac{1}{N}\sum_{i=1}^{N}g_i.
$$

Using one sample gradient per episode,

$$
g_{i,k}=(R_i-b)(\mathbb{1}[k=a_i]-\pi_i(k)).
$$

Gradient ascent with step size $\eta$ gives the stated update.

## 4.6 EMA value baseline

The baseline update is

$$
b_t = \lambda b_{t-1} + (1-\lambda)\bar{R}_t,
\qquad
\bar{R}_t = \frac{1}{N}\sum_{i=1}^{N}R_i.
$$

### Closed-form proof

Unrolling the recurrence,

$$
b_t = \lambda\left(\lambda b_{t-2}+(1-\lambda)\bar{R}_{t-1}\right)+(1-\lambda)\bar{R}_t.
$$

So

$$
b_t = \lambda^2 b_{t-2}+\lambda(1-\lambda)\bar{R}_{t-1}+(1-\lambda)\bar{R}_t.
$$

Continuing,

$$
b_t = \lambda^t b_0 + (1-\lambda)\sum_{s=1}^{t}\lambda^{t-s}\bar{R}_s.
$$

For $0\le\lambda<1$, older batch rewards receive exponentially smaller weights.

---

# 5. Entropy regularization

The Shannon entropy of the action distribution is

$$
H(\pi) = -\sum_{k=1}^{K}\pi(k)\log\pi(k).
$$

The SDK-style entropy contribution to the logit update is proportional to

$$
-\log \pi(k) - H(\pi).
$$

This term favors lower-probability actions and discourages premature collapse.

## 5.1 Full derivative of entropy with respect to a logit

Let $\pi_i=\operatorname{softmax}(z)_i$. Then

$$
\frac{\partial H}{\partial z_k}
= \pi_k\left(-\log\pi_k - H\right).
$$

### Proof

First, the softmax Jacobian is

$$
\frac{\partial \pi_i}{\partial z_k} = \pi_i(\mathbb{1}[i=k]-\pi_k).
$$

Entropy is

$$
H=-\sum_i\pi_i\log\pi_i.
$$

Differentiate:

$$
\frac{\partial H}{\partial z_k}
= -\sum_i \frac{\partial \pi_i}{\partial z_k}(\log\pi_i+1).
$$

Substitute the softmax Jacobian:

$$
\frac{\partial H}{\partial z_k}
= -\sum_i \pi_i(\mathbb{1}[i=k]-\pi_k)(\log\pi_i+1).
$$

Split the sum:

$$
= -\pi_k(\log\pi_k+1) + \pi_k\sum_i\pi_i(\log\pi_i+1).
$$

Now

$$
\sum_i\pi_i(\log\pi_i+1)=\sum_i\pi_i\log\pi_i + \sum_i\pi_i = -H+1.
$$

Therefore

$$
\frac{\partial H}{\partial z_k}
= -\pi_k(\log\pi_k+1)+\pi_k(-H+1)
= \pi_k(-\log\pi_k - H).
$$

## 5.2 Relation to the implemented entropy bonus form

The mathematically exact entropy-gradient contribution to logit $z_k$ is

$$
\beta\pi_k(-\log\pi_k-H).
$$

Some implementations apply the unweighted row term

$$
\beta(-\log\pi_k-H)
$$

as a direct exploration bonus. This keeps the same sign structure per action but omits the multiplicative $\pi_k$ factor. It is therefore best read as an entropy-shaped heuristic unless the code explicitly multiplies by $\pi_k$.

### Sign analysis

If $\pi_k$ is small, then $-\log\pi_k$ is large and positive. This makes

$$
-\log\pi_k-H
$$

more likely to be positive, increasing the low-probability action's logit. If $\pi_k$ is large, the term can be negative, reducing the high-probability action's logit.

---

# 6. Contextual REINFORCE

For contextual softmax,

$$
z_k = W_k\phi,
\qquad
\pi(k\mid\phi)=\operatorname{softmax}(W\phi)_k.
$$

The contextual REINFORCE update is

$$
\Delta W_k = \eta(R-b)(\mathbb{1}[k=a]-\pi_k)\phi^{\top}.
$$

If rows are stored as one-dimensional arrays, this is written componentwise as

$$
\Delta W_{k,j}=\eta(R-b)(\mathbb{1}[k=a]-\pi_k)\phi_j.
$$

## 6.1 Derivation by chain rule

From Section 3,

$$
\frac{\partial\log\pi(a\mid\phi)}{\partial z_k}
=\mathbb{1}[k=a]-\pi_k.
$$

Since

$$
z_k=\sum_jW_{k,j}\phi_j,
$$

we have

$$
\frac{\partial z_k}{\partial W_{r,j}}=\mathbb{1}[r=k]\phi_j.
$$

Therefore

$$
\frac{\partial\log\pi(a\mid\phi)}{\partial W_{r,j}}
=\sum_k\frac{\partial\log\pi(a\mid\phi)}{\partial z_k}\frac{\partial z_k}{\partial W_{r,j}}
= (\mathbb{1}[r=a]-\pi_r)\phi_j.
$$

Using advantage $A=R-b$ and gradient ascent,

$$
\Delta W_{r,j}=\eta A(\mathbb{1}[r=a]-\pi_r)\phi_j.
$$

In vector form for row $r$,

$$
\Delta W_r=\eta A(\mathbb{1}[r=a]-\pi_r)\phi^{\top}.
$$

## 6.2 Outer-product view

Let

$$
y \in \mathbb{R}^K, \qquad y_k=\mathbb{1}[k=a].
$$

Then the full matrix update is

$$
\Delta W = \eta A (y-\pi)\phi^{\top}.
$$

### Proof of dimensions

$y-\pi\in\mathbb{R}^{K}$ and $\phi^{\top}\in\mathbb{R}^{1\times d}$. Their outer product has shape

$$
(K\times 1)(1\times d)=K\times d,
$$

matching $W$.

## 6.3 Row step norm

The row update norm is

$$
\lVert \Delta W_k\rVert_2
= \sqrt{\sum_{j=1}^{d}(\Delta W_{k,j})^2}.
$$

Using the contextual update,

$$
\lVert \Delta W_k\rVert_2
= \eta |A|\,|\mathbb{1}[k=a]-\pi_k|\,\lVert\phi\rVert_2.
$$

### Proof

Substitute

$$
\Delta W_{k,j}=\eta A(\mathbb{1}[k=a]-\pi_k)\phi_j.
$$

Then

$$
\lVert\Delta W_k\rVert_2
=\sqrt{\sum_j\left(\eta A(\mathbb{1}[k=a]-\pi_k)\phi_j\right)^2}
$$

$$
=\sqrt{\eta^2A^2(\mathbb{1}[k=a]-\pi_k)^2\sum_j\phi_j^2}
=\eta |A|\,|\mathbb{1}[k=a]-\pi_k|\,\lVert\phi\rVert_2.
$$

---

# 7. Off-policy importance weighting

When an episode was logged under a behavior policy $\mu$ but updated under target policy $\pi$, the likelihood ratio is

$$
\rho(a)=\frac{\pi(a)}{\mu(a)}.
$$

With clipping,

$$
\rho_c(a)=\min\left(\rho(a),c\right).
$$

The weighted update is

$$
\Delta z_k=\eta\rho_c(a)(R-b)(\mathbb{1}[k=a]-\pi(k)).
$$

## 7.1 Why the likelihood ratio appears

For any function $f(a)$,

$$
\mathbb{E}_{a\sim\pi}[f(a)]
=\sum_a\pi(a)f(a)
=\sum_a\mu(a)\frac{\pi(a)}{\mu(a)}f(a)
=\mathbb{E}_{a\sim\mu}\left[\rho(a)f(a)\right],
$$

provided $\mu(a)>0$ whenever $\pi(a)>0$.

This converts an expectation under the target policy into an expectation under the logged behavior policy.

## 7.2 What clipping changes

Clipping replaces $\rho$ with $\rho_c$. Since

$$
0\le \rho_c(a)\le c,
$$

single examples cannot produce arbitrarily large scaled updates. The trade-off is that clipping generally introduces bias, because

$$
\mathbb{E}_{\mu}[\rho_c(a)f(a)]\ne\mathbb{E}_{\pi}[f(a)]
$$

unless no clipping occurs or the clipped mass contributes zero to the relevant function.

---

# 8. Reward shaping

Each metric scorer returns a normalized score

$$
s_m\in[0,1].
$$

The signed score is

$$
\operatorname{signed}_m = 2s_m-1.
$$

The final reward is

$$
R=\operatorname{clip}\left(\sum_m w_m\operatorname{signed}_m + \sum_jp_j, -1, 1\right),
$$

where $p_j$ are additive penalties or bonuses.

## 8.1 Proof that signed metric scores lie in $[-1,1]$

If $0\le s_m\le1$, multiply by $2$:

$$
0\le2s_m\le2.
$$

Subtract $1$:

$$
-1\le2s_m-1\le1.
$$

Thus

$$
\operatorname{signed}_m\in[-1,1].
$$

## 8.2 Interpretation of endpoint mapping

If $s_m=1$,

$$
2s_m-1=1.
$$

If $s_m=0$,

$$
2s_m-1=-1.
$$

If $s_m=0.5$,

$$
2s_m-1=0.
$$

Thus the transformation maps pass/fail-neutral scoring into a symmetric reward signal.

## 8.3 Clipping bounds the final reward

For any scalar $q$,

$$
\operatorname{clip}(q,-1,1)=
\begin{cases}
-1, & q<-1,\\
q, & -1\le q\le1,\\
1, & q>1.
\end{cases}
$$

Therefore $R\in[-1,1]$ by construction.

### Proof

The definition returns one of three values: $-1$, $q$ where $q\in[-1,1]$, or $1$. Every possible branch lies in $[-1,1]$.

---

# 9. Logistic regression primitives

## 9.1 Numerically stable sigmoid

The sigmoid is

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

The stable piecewise form is

$$
\sigma(x)=
\begin{cases}
\dfrac{1}{1+e^{-x}}, & x\ge0,\\[2ex]
\dfrac{e^x}{1+e^x}, & x<0.
\end{cases}
$$

### Proof of equivalence for $x<0$

Starting from the standard formula,

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

Multiply numerator and denominator by $e^x$:

$$
\sigma(x)=\frac{e^x}{e^x+1}=\frac{e^x}{1+e^x}.
$$

When $x<0$, $e^x$ is small, avoiding overflow from $e^{-x}$.

## 9.2 Binary logistic-regression loss

For label $y\in\{0,1\}$ and prediction

$$
p=\sigma(w\cdot x),
$$

the binary cross-entropy loss is

$$
\ell(w)= -y\log p -(1-y)\log(1-p).
$$

The gradient is

$$
\nabla_w\ell=(p-y)x.
$$

### Proof

Let

$$
t=w\cdot x.
$$

Then

$$
p=\sigma(t).
$$

We need two derivatives:

$$
\frac{d\sigma(t)}{dt}=\sigma(t)(1-\sigma(t))=p(1-p),
$$

and

$$
\frac{dt}{dw}=x.
$$

Differentiate the loss with respect to $p$:

$$
\frac{d\ell}{dp}= -\frac{y}{p}+\frac{1-y}{1-p}.
$$

Now chain the derivatives:

$$
\nabla_w\ell
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)x.
$$

Simplify:

$$
=\left(-y(1-p)+(1-y)p\right)x
=\left(-y+yp+p-yp\right)x
=(p-y)x.
$$

## 9.3 Mini-batch SGD with L2 weight decay

For mini-batch $\mathcal{B}$,

$$
g_j=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}(p_i-y_i)x_{i,j}+\gamma w_j,
$$

and

$$
w_j\leftarrow w_j-\eta g_j.
$$

### Proof from regularized objective

The batch objective with L2 penalty is

$$
L(w)=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\ell_i(w)+\frac{\gamma}{2}\lVert w\rVert_2^2.
$$

The gradient of the data term is

$$
\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}(p_i-y_i)x_i.
$$

The gradient of the penalty is

$$
\nabla_w\frac{\gamma}{2}\lVert w\rVert_2^2
=\nabla_w\frac{\gamma}{2}\sum_jw_j^2
=\gamma w.
$$

Thus component $j$ of the gradient is the stated $g_j$. Gradient descent subtracts $\eta g_j$.

---

# 10. Multinomial logistic regression

For $K$ classes, logits are

$$
z_k=w_k\cdot x.
$$

The predicted class probabilities are

$$
p_k=\operatorname{softmax}(z)_k.
$$

For true label $y$, the cross-entropy loss is

$$
\ell(W)=-\log p_y.
$$

The row gradient is

$$
\nabla_{w_k}\ell=(p_k-\mathbb{1}[k=y])x.
$$

## 10.1 Proof

From Section 3,

$$
\frac{\partial\log p_y}{\partial z_k}=\mathbb{1}[k=y]-p_k.
$$

Since

$$
\ell=-\log p_y,
$$

we have

$$
\frac{\partial\ell}{\partial z_k}=p_k-\mathbb{1}[k=y].
$$

Also,

$$
z_k=w_k\cdot x,
\qquad
\frac{\partial z_k}{\partial w_k}=x.
$$

By chain rule,

$$
\nabla_{w_k}\ell=(p_k-\mathbb{1}[k=y])x.
$$

## 10.2 Batch-averaged L2-regularized update

For mini-batch $\mathcal{B}$,

$$
g_{k,j}=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}(p_{k,i}-\mathbb{1}[k=y_i])x_{i,j}+\gamma w_{k,j},
$$

and

$$
w_{k,j}\leftarrow w_{k,j}-\eta g_{k,j}.
$$

### Proof

Use the same regularized objective as in binary logistic regression, but sum cross-entropy over multiclass examples. The data gradient for row $k$ is the average of per-example row gradients. The L2 penalty contributes $\gamma w_{k,j}$ to component $(k,j)$. Gradient descent gives the stated update.

---

# 11. Router and classifier math

## 11.1 Logistic-regression router

The router appends a bias and forms

$$
x=[\phi,1].
$$

Class logits are

$$
z_k=w_k\cdot x.
$$

The predicted class is

$$
\hat{k}=\operatorname*{arg\,max}_k\operatorname{softmax}(z)_k.
$$

## 11.2 Argmax of softmax equals argmax of logits

$$
\operatorname*{arg\,max}_k\operatorname{softmax}(z)_k
=
\operatorname*{arg\,max}_k z_k.
$$

### Proof

Softmax is

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

The denominator is positive and identical for every $k$. The exponential function is strictly increasing. Therefore for any two classes $r$ and $s$,

$$
z_r>z_s \iff e^{z_r}>e^{z_s} \iff p_r>p_s.
$$

Thus the index with maximum probability is the index with maximum logit.

## 11.3 Refusal threshold

If

$$
\max_k p_k<\tau,
$$

the router refuses and reports complement confidence

$$
1-\max_kp_k.
$$

This is not a probability that the refusal label is semantically correct. It is a monotone uncertainty score: as top-class confidence decreases, refusal confidence increases.

---

# 12. Prototype router and cosine similarity

For query feature vector $\phi$ and class prototype $p^{(k)}$,

$$
\operatorname{cos}(\phi,p^{(k)})=rac{\phi\cdot p^{(k)}}{\lVert\phi\rVert_2\lVert p^{(k)}\rVert_2}.
$$

## 12.1 Cosine similarity equals the cosine of the angle

For nonzero vectors $u$ and $v$,

$$
u\cdot v=\lVert u\rVert_2\lVert v\rVert_2\cos\theta.
$$

So

$$
\frac{u\cdot v}{\lVert u\rVert_2\lVert v\rVert_2}=\cos\theta.
$$

### Proof sketch

The identity follows from the law of cosines. Consider the triangle with sides $u$, $v$, and $u-v$. The law of cosines gives

$$
\lVert u-v\rVert_2^2=\lVert u\rVert_2^2+\lVert v\rVert_2^2-2\lVert u\rVert_2\lVert v\rVert_2\cos\theta.
$$

But expanding the squared norm using dot products gives

$$
\lVert u-v\rVert_2^2=(u-v)\cdot(u-v)=\lVert u\rVert_2^2-2u\cdot v+\lVert v\rVert_2^2.
$$

Equating the two expressions yields

$$
u\cdot v=\lVert u\rVert_2\lVert v\rVert_2\cos\theta.
$$

## 12.2 Range of cosine similarity

For nonzero $u$ and $v$,

$$
-1\le\operatorname{cos}(u,v)\le1.
$$

### Proof

By Cauchy-Schwarz,

$$
|u\cdot v|\le\lVert u\rVert_2\lVert v\rVert_2.
$$

Divide both sides by the positive denominator:

$$
\left|\frac{u\cdot v}{\lVert u\rVert_2\lVert v\rVert_2}\right|\le1.
$$

Therefore cosine similarity lies in $[-1,1]$.

## 12.3 Zero-vector denominator guard

If either vector is zero, the mathematical cosine similarity is undefined because the denominator is zero. A denominator floor such as

$$
\max(\lVert u\rVert_2,1)\max(\lVert v\rVert_2,1)
$$

prevents division by zero but changes the mathematical quantity. It should be interpreted as a defensive implementation convention for degenerate vectors.

---

# 13. Binary scorers

A binary scorer uses a feature vector

$$
x=[\phi\Vert\mathbf{onehot}(a)\Vert1]\in\mathbb{R}^{d+K+1}.
$$

The pass probability is

$$
p=\sigma(w\cdot x).
$$

The label rule is

$$
\text{label}=\begin{cases}
\text{pass},&p\ge0.5,\\
\text{fail},&p<0.5.
\end{cases}
$$

## 13.1 Threshold proof

Because sigmoid is strictly increasing,

$$
p\ge0.5 \iff \sigma(w\cdot x)\ge0.5.
$$

Now

$$
\sigma(0)=\frac{1}{2}.
$$

Since $\sigma$ is strictly increasing,

$$
\sigma(w\cdot x)\ge\sigma(0) \iff w\cdot x\ge0.
$$

Thus the probability threshold $0.5$ is equivalent to the linear decision boundary

$$
w\cdot x=0.
$$

---

# 14. Feature extraction

## 14.1 Hashing-trick bag-of-words

For token $t$, define a bucket

$$
h(t)=\operatorname{int}_{32}(\operatorname{md5}(t))\bmod D.
$$

The count vector is

$$
v_i=\sum_{t\in\text{tokens}}\mathbb{1}[h(t)=i],\qquad i\in[0,D).
$$

In binary mode,

$$
v_i=\mathbb{1}[\exists t:h(t)=i].
$$

## 14.2 Count-vector proof

The indicator $\mathbb{1}[h(t)=i]$ contributes $1$ only when token $t$ maps to bucket $i$. Summing over all tokens counts how many tokens map to bucket $i$. Binary mode changes the aggregation from total count to existence.

## 14.3 Collision behavior

If two distinct tokens $t_1$ and $t_2$ satisfy

$$
h(t_1)=h(t_2),
$$

they add to the same coordinate. The hashing trick therefore trades explicit vocabulary storage for possible collisions.

---

# 15. TF-IDF plus logistic regression

A common term-frequency-inverse-document-frequency feature is

$$
\operatorname{tfidf}(t,\delta)=\operatorname{tf}(t,\delta)\log\frac{|\mathcal{D}|}{1+|\{\delta':t\in\delta'\}|}.
$$

## 15.1 Interpretation

The term-frequency factor increases with the frequency of term $t$ in document $\delta$. The inverse-document-frequency factor decreases as the number of documents containing $t$ increases.

## 15.2 Monotonicity proof for IDF factor

Let

$$
N=|\mathcal{D}|,
\qquad
d_t=|\{\delta':t\in\delta'\}|.
$$

The IDF term is

$$
\log\frac{N}{1+d_t}=\log N-\log(1+d_t).
$$

As $d_t$ increases, $\log(1+d_t)$ increases, so the IDF factor decreases. Thus common terms receive smaller weights than rare terms, all else equal.

---

# 16. Metric normalization

## 16.1 Intent-resolution normalization

For a raw score $s\in[1,5]$,

$$
\operatorname{norm}(s)=\frac{\operatorname{clip}(s,1,5)-1}{4}.
$$

## 16.2 Proof of range

Since

$$
1\le\operatorname{clip}(s,1,5)\le5,
$$

subtracting $1$ gives

$$
0\le\operatorname{clip}(s,1,5)-1\le4.
$$

Dividing by $4$ gives

$$
0\le\operatorname{norm}(s)\le1.
$$

## 16.3 Endpoint proof

For $s=1$,

$$
\operatorname{norm}(1)=\frac{1-1}{4}=0.
$$

For $s=5$,

$$
\operatorname{norm}(5)=\frac{5-1}{4}=1.
$$

Thus the mapping sends the lowest score to $0$ and the highest score to $1$.

---

# 17. Clipping of logits and weights

The policy clips logit or weight updates into a symmetric box:

$$
z_a\leftarrow\operatorname{clip}(z_a+\Delta z_a,-z_{\max},z_{\max}),
$$

or

$$
W_{k,j}\leftarrow\operatorname{clip}(W_{k,j}+\Delta W_{k,j},-w_{\max},w_{\max}).
$$

## 17.1 Bound proof

By definition,

$$
\operatorname{clip}(q,-c,c)=
\begin{cases}
-c,&q<-c,\\
q,&-c\le q\le c,\\
c,&q>c.
\end{cases}
$$

Every branch lies in $[-c,c]$. Therefore repeated clipped updates keep each clipped scalar inside the configured interval.

## 17.2 Practical implication

Clipping bounds the logit scale. Since softmax probabilities become increasingly sharp as logit gaps grow, clipping limits how deterministic the policy can become. This is an implementation stability control, not part of the ideal policy-gradient theorem.

---

# 18. Summary identity sheet

## Softmax

$$
\pi_k=\frac{e^{z_k-m}}{\sum_j e^{z_j-m}},\qquad m=\max_jz_j.
$$

## Softmax log-gradient

$$
\frac{\partial\log\pi_a}{\partial z_k}=\mathbb{1}[k=a]-\pi_k.
$$

## Marginal REINFORCE

$$
\Delta z_k=\eta(R-b)(\mathbb{1}[k=a]-\pi_k).
$$

## Contextual REINFORCE

$$
\Delta W_k=\eta(R-b)(\mathbb{1}[k=a]-\pi_k)\phi^{\top}.
$$

## Entropy gradient

$$
\frac{\partial H}{\partial z_k}=\pi_k(-\log\pi_k-H).
$$

## Binary logistic-regression gradient

$$
\nabla_w\ell=(p-y)x.
$$

## Multiclass logistic-regression gradient

$$
\nabla_{w_k}\ell=(p_k-\mathbb{1}[k=y])x.
$$

## Cosine similarity

$$
\operatorname{cos}(u,v)=\frac{u\cdot v}{\lVert u\rVert_2\lVert v\rVert_2}.
$$

## Reward shaping

$$
R=\operatorname{clip}\left(\sum_m w_m(2s_m-1)+\sum_jp_j,-1,1\right).
$$

---

# 19. Implementation notes for SDK readers

1. The REINFORCE update is gradient ascent because the learner maximizes expected reward.
2. Logistic-regression training is gradient descent because the classifier minimizes cross-entropy loss.
3. The same softmax identity drives both policy-gradient learning and multiclass logistic-regression training; the only sign difference comes from maximizing reward versus minimizing loss.
4. A baseline changes variance, not the expected score-function gradient, provided it does not depend on the sampled action.
5. A contextual policy is just a linear map followed by a softmax; the gradient becomes an outer product between an action-error vector and the context features.
6. Reward shaping determines what signal the policy optimizes. The learner can only improve behavior relative to the reward signal it receives.
7. Clipping, probability floors, and denominator guards are numerical and operational safeguards. They are important for robust code but should be documented separately from the ideal mathematical identities.
