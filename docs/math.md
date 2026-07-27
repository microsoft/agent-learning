---
title: Mathematical reference for the agent-learning SDK
description: Consolidated linear-algebra and probability formulas behind the policies, REINFORCE learner, reward shaper, logistic-regression classifiers, router, and judges of the agent-learning SDK.
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
estimated_reading_time: 15
---

## Overview

This document collects every linear-algebra and probability formula the SDK relies on in one place. Each section names the source module so the math can be traced back to the implementation. All vectors are column vectors unless stated otherwise, and all softmaxes are evaluated with the max-subtraction trick for numerical stability.

## Notation

| Symbol | Meaning |
| --- | --- |
| $K$ | Number of discrete actions (or classes). |
| $d$ | Feature dimension of a context vector. |
| $\phi \in \mathbb{R}^{d}$ | Context feature vector (`phi`). |
| $z \in \mathbb{R}^{K}$ | Action logits. |
| $W \in \mathbb{R}^{K \times d}$ | Contextual policy weight matrix. |
| $w \in \mathbb{R}^{n}$ | Logistic-regression weight vector (with bias column). |
| $\pi(a)$ | Policy probability of action $a$. |
| $R$ | Aggregate episode reward, $R \in [-1, 1]$. |
| $b$ | Value baseline. |
| $\eta$ | Learning rate. |
| $\mathbb{1}[\cdot]$ | Indicator function (1 if true, else 0). |

## Softmax policies

### Marginal softmax bandit

Source: [../src/agent_learning/policy/softmax_bandit.py](../src/agent_learning/policy/softmax_bandit.py)

The policy stores one logit $z_a$ per action. Probabilities use the numerically stable softmax (subtract the max logit before exponentiating):

$$
\pi(a) = \frac{\exp\!\left(z_a - \max_b z_b\right)}{\displaystyle\sum_{c} \exp\!\left(z_c - \max_b z_b\right)}
$$

The log-probability returned with each decision is clamped away from $-\infty$:

$$
\log \pi(a) = \log\big(\max(\pi(a),\, 10^{-12})\big)
$$

Logit updates are additive and clipped to a symmetric box $[-z_{\max}, z_{\max}]$:

$$
z_a \leftarrow \operatorname{clip}\big(z_a + \Delta z_a,\ -z_{\max},\ z_{\max}\big)
$$

### Contextual (linear) softmax

Source: [../src/agent_learning/policy/contextual_softmax.py](../src/agent_learning/policy/contextual_softmax.py)

The policy stores a weight matrix $W \in \mathbb{R}^{K \times d}$. Given a context vector $\phi$, the logits are the matrix–vector product and the distribution is the stable softmax:

$$
z = W \phi, \qquad
\pi(a \mid \phi) = \frac{\exp\!\left(z_a - \max_b z_b\right)}{\displaystyle\sum_{c} \exp\!\left(z_c - \max_b z_b\right)}
$$

When no context is supplied ($\phi = \mathbf{0}$) the policy is uniform, $\pi(a) = 1/K$. Weight updates are element-wise additive per action row and clipped:

$$
W_a \leftarrow \operatorname{clip}\big(W_a + \Delta W_a,\ -w_{\max},\ w_{\max}\big)
$$

### Weighted sampling

Both policies sample an action by inverse-CDF over the probability vector. For a uniform draw $u \sim \mathcal{U}[0,1)$, the chosen index is:

$$
a^{*} = \min\left\{ k : \sum_{j=0}^{k} \pi(j) \ge u \right\}
$$

## REINFORCE-with-baseline learner

Source: [../src/agent_learning/learners/reinforce.py](../src/agent_learning/learners/reinforce.py)

For a softmax policy, the gradient of the log-likelihood of the taken action $a$ with respect to logit $z_k$ is $\mathbb{1}[k = a] - \pi(k)$. The per-episode REINFORCE-with-baseline update for logit $z_k$ is therefore:

$$
\Delta z_k = \eta \,(R - b)\,\big(\mathbb{1}[k = a] - \pi(k)\big)
$$

### Entropy regularization

The Shannon entropy of the current action distribution keeps exploration alive:

$$
H(\pi) = -\sum_{k} \pi(k)\,\log \max\!\big(\pi(k),\, 10^{-12}\big)
$$

Its gradient with respect to logit $z_k$ contributes an entropy bonus scaled by $\beta$:

$$
\Delta z_k \mathrel{+}= \beta \,\big(-\log \max(\pi(k), 10^{-12}) - H(\pi)\big)
$$

### Batch averaging

Gradients are accumulated over the $N$ usable episodes in a batch, averaged, and scaled by the learning rate:

$$
\Delta z_k = \frac{\eta}{N} \sum_{i=1}^{N} \Delta z_k^{(i)}
$$

### EMA value baseline

The scalar baseline is an exponential moving average of the mean batch reward $\bar{R}$ with decay $\lambda$:

$$
b \leftarrow \lambda\, b + (1 - \lambda)\, \bar{R}, \qquad \bar{R} = \frac{1}{N}\sum_{i=1}^{N} R_i
$$

### Off-policy importance weighting

When an episode was logged under an older behaviour policy, its advantage is reweighted by the clipped likelihood ratio:

$$
\rho = \min\!\left(\frac{\pi_{\text{target}}(a)}{\pi_{\text{behaviour}}(a)},\ c\right),
\qquad
\pi_{\text{behaviour}}(a) = \exp\big(\text{logprob}_a\big)
$$

with both probabilities floored at $10^{-12}$ and $c$ the importance clip.

## Contextual REINFORCE (worked example)

Source: [../examples/next_best_action.py](../examples/next_best_action.py)

For a linear-softmax policy $\pi = \operatorname{softmax}(W\phi)$, the REINFORCE-with-baseline gradient for action row $W_k$ is the outer-product-style term:

$$
\Delta W_k = \eta \,(R - b)\,\big(\mathbb{1}[k = a] - \pi_k\big)\,\phi
$$

with the same per-row entropy bonus as the marginal learner, added before multiplying by $\phi$:

$$
g_k = (R - b)\big(\mathbb{1}[k = a] - \pi_k\big) + \beta\big(-\log\max(\pi_k, 10^{-12}) - H(\pi)\big),
\qquad
\Delta W_k = \eta \, g_k \, \phi
$$

This example uses a **batch-mean baseline** (lower variance than the EMA) that centres the advantages each round:

$$
b = \bar{R} = \frac{1}{N}\sum_{i=1}^{N} R_i
$$

The reported per-row step size is the Euclidean norm of the averaged weight delta:

$$
\left\lVert \Delta W_k \right\rVert_2 = \sqrt{\sum_{j=1}^{d} \big(\Delta W_{k,j}\big)^2}
$$

### Context feature encoding

The context vector $\phi$ concatenates a bias term, the continuous variables, and one-hot encodings of the categorical variables:

$$
\phi = \big[\,\underbrace{1}_{\text{bias}},\ \underbrace{x_1, \dots, x_m}_{\text{continuous}},\ \underbrace{\mathbf{onehot}(c_1), \dots, \mathbf{onehot}(c_p)}_{\text{categorical}}\,\big]
$$

so the feature dimension is $d = 1 + m + \sum_{i=1}^{p} |\mathcal{C}_i|$, where $|\mathcal{C}_i|$ is the number of categories of variable $i$.

### Greedy recommendation

The greedy next best action is the arg-max over the action distribution:

$$
a_{\text{greedy}} = \operatorname*{arg\,max}_{k} \ \pi(k \mid \phi)
$$

## Reward shaping

Source: [../src/agent_learning/rewards/shaping.py](../src/agent_learning/rewards/shaping.py)

Each judge produces a normalized score $s_m \in [0, 1]$. The shaper maps it to a signed contribution in $[-1, 1]$ and forms a weighted sum with behavioural penalties, then clamps:

$$
\text{signed}_m = 2 s_m - 1
$$

$$
R = \operatorname{clip}\!\left(\sum_{m} w_m \,\text{signed}_m \;+\; \sum_{j} p_j,\ -1,\ 1\right)
$$

where $w_m$ are the per-metric weights and $p_j$ are additive penalty/bonus terms (latency, routing correctness, hallucinated-class).

## Logistic-regression primitives

Source: [../src/agent_learning/classifiers/base.py](../src/agent_learning/classifiers/base.py)

These pure-Python primitives back the router and every stdlib/NLP judge.

### Numerically stable sigmoid

$$
\sigma(x) =
\begin{cases}
\dfrac{1}{1 + e^{-x}}, & x \ge 0 \\[2ex]
\dfrac{e^{x}}{1 + e^{x}}, & x < 0
\end{cases}
$$

### Softmax and dot product

$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i - \max_j z_j)}{\sum_{j} \exp(z_j - \max_j z_j)},
\qquad
w \cdot x = \sum_{i} w_i x_i
$$

### Binary logistic regression (mini-batch SGD)

For a feature vector $x$ (with an appended bias column) and label $y \in \{0, 1\}$, the prediction and per-example gradient are:

$$
p = \sigma(w \cdot x), \qquad \nabla_w = (p - y)\, x
$$

Over a mini-batch $\mathcal{B}$ with L2 weight decay $\gamma$ and learning rate $\eta$:

$$
g_j = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (p_i - y_i)\, x_{i,j} + \gamma\, w_j,
\qquad
w_j \leftarrow w_j - \eta\, g_j
$$

### Multinomial logistic regression (softmax classifier)

For $K$ classes with weight rows $w_k$, class probabilities and the per-example gradient for row $k$ are:

$$
p_k = \operatorname{softmax}\big(w_k \cdot x\big),
\qquad
\nabla_{w_k} = \big(p_k - \mathbb{1}[k = y]\big)\, x
$$

with the same batch-averaged L2-regularized descent step applied per row:

$$
g_{k,j} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \big(p_{k,i} - \mathbb{1}[k = y_i]\big)\, x_{i,j} + \gamma\, w_{k,j},
\qquad
w_{k,j} \leftarrow w_{k,j} - \eta\, g_{k,j}
$$

## Router / classifier

Source: [../src/agent_learning/classifiers/router.py](../src/agent_learning/classifiers/router.py)

### Logistic-regression mode

The router appends a bias to $\phi$, computes class logits, and takes the softmax arg-max. Let $x = [\phi, 1]$:

$$
z_k = w_k \cdot x, \qquad
\hat{k} = \operatorname*{arg\,max}_k \operatorname{softmax}(z)_k
$$

If the top probability is below the refusal threshold $\tau$, the router refuses with complement confidence:

$$
\text{if } \max_k p_k < \tau: \quad \text{label} = \text{refused}, \quad \text{confidence} = 1 - \max_k p_k
$$

### Prototype (cosine-similarity) mode

Each class stores a prototype vector. The router scores the query $\phi$ against every prototype $p^{(k)}$ by cosine similarity, then softmaxes the similarities:

$$
\operatorname{cos}\big(\phi, p^{(k)}\big) = \frac{\phi \cdot p^{(k)}}{\lVert \phi \rVert_2\, \big\lVert p^{(k)} \big\rVert_2},
\qquad
\hat{k} = \operatorname*{arg\,max}_k \ \operatorname{softmax}\big(\cos(\phi, p^{(\cdot)})\big)_k
$$

Norms are floored at $1$ (via `or 1.0`) to avoid division by zero on all-zero vectors.

## Binary judges

Source: [../src/agent_learning/classifiers/judges/_base.py](../src/agent_learning/classifiers/judges/_base.py)

Each judge is a binary logistic-regression classifier over a feature vector that concatenates the context, a one-hot action encoding, and a bias term:

$$
x = \big[\,\phi \ \Vert \ \mathbf{onehot}(a) \ \Vert \ 1\,\big] \in \mathbb{R}^{d + K + 1}
$$

$$
p = \sigma(w \cdot x), \qquad
\text{label} =
\begin{cases}
\text{pass}, & p \ge 0.5 \\
\text{fail}, & p < 0.5
\end{cases}
$$

with confidence $p$ when passing and $1 - p$ when failing.

## Feature extraction

### Hashing-trick bag-of-words (Tier 1 stdlib)

Source: [../src/agent_learning/judges/stdlib/_text.py](../src/agent_learning/judges/stdlib/_text.py)

Tokens are hashed into a fixed number of buckets $D$, removing the need to persist a vocabulary. For token $t$, the bucket is derived from the first 32 bits of its MD5 digest:

$$
h(t) = \big(\text{int}_{32}(\operatorname{md5}(t))\big) \bmod D
$$

The count-based feature vector accumulates over the token stream:

$$
v_i = \sum_{t \in \text{tokens}} \mathbb{1}\big[h(t) = i\big], \qquad i \in [0, D)
$$

(In binary mode, $v_i = \mathbb{1}[\exists\, t : h(t) = i]$.) The intent judge concatenates the query and response token streams before hashing.

### TF-IDF + logistic regression (Tier 2 NLP)

Source: [../src/agent_learning/judges/nlp_text/_base.py](../src/agent_learning/judges/nlp_text/_base.py)

Tier 2 judges delegate to scikit-learn's `TfidfVectorizer` (unigram+bigram) feeding a `LogisticRegression` head. The term-frequency–inverse-document-frequency weight of term $t$ in document $\delta$ over corpus $\mathcal{D}$ is:

$$
\text{tfidf}(t, \delta) = \text{tf}(t, \delta)\cdot\log\!\frac{|\mathcal{D}|}{1 + |\{\delta' : t \in \delta'\}|}
$$

## Metric normalization

Source: [../src/agent_learning/metrics/](../src/agent_learning/metrics/)

Judges emit scores on heterogeneous scales; each metric normalizes to $[0, 1]$ before shaping.

| Metric | Raw range | Normalization |
| --- | --- | --- |
| Intent resolution | $[1, 5]$ | $\dfrac{\operatorname{clip}(s, 1, 5) - 1}{4}$ |
| Task adherence | $[0, 1]$ | $\operatorname{clip}(s, 0, 1)$ |
| Task completion | $[0, 1]$ | $\operatorname{clip}(s, 0, 1)$ |

The intent-resolution mapping sends a perfect score of $5$ to $1.0$ and the worst score of $1$ to $0.0$, so a perfect judgment becomes the maximum positive reward signal after shaping.
