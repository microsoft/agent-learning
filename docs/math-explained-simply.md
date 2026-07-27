---
title: The SDK math, explained simply
description: A plain-language, fifth-grade-level walkthrough of every formula in the agent-learning SDK, with everyday analogies and small worked examples you can use to teach others.
author: Microsoft
ms.date: 2026-07-27
ms.topic: overview
keywords:
  - reinforcement learning
  - explain like I'm five
  - softmax
  - reward
  - learning
  - beginner
estimated_reading_time: 20
---

## Read me first

This is the friendly, no-scary-symbols version of [math.md](math.md). Every idea here has a matching grown-up formula in that file, so once an idea "clicks," you can peek at the real math and recognize it.

**The big picture:** this SDK is a little robot helper that makes choices, watches how they turn out, and slowly gets smarter. Almost all the math is about two things:

1. **Making a choice** when you're not 100% sure (turning "scores" into "chances").
2. **Learning from what happened** (nudging yourself to do more of what worked).

Keep that in mind and everything below is just details.

## 1. Turning scores into chances (softmax)

> Grown-up name: **softmax**. See [math.md](math.md#softmax-policies).

Imagine you have three snacks and you give each a score for how much you like it:

- Pizza: **2**
- Apple: **1**
- Broccoli: **0**

You don't want to *always* pick pizza (boring, and maybe you'd miss out). You want pizza to be *most likely* but still give the others a chance. Softmax turns those scores into chances that add up to 100%:

- Pizza → about **67%**
- Apple → about **24%**
- Broccoli → about **9%**

The trick: bigger scores get much bigger chances, but nothing ever drops to a flat zero.

**One-sentence version:** *Softmax is a fair spinner wheel where the choices you like get bigger slices.*

> Tiny grown-up peek: you raise a special number $e$ (about 2.718) to the power of each score, then divide each by the total. Higher score → way bigger slice.

## 2. Choices that depend on the situation (contextual softmax)

> Grown-up name: **contextual softmax** / **W · phi**. See [math.md](math.md#contextual-linear-softmax).

Sometimes the best choice changes depending on what's going on. You'd pick a different snack at a birthday party than before bedtime.

The "situation" gets written down as a list of numbers called **phi** (say it "fye"). For example: *is it morning? is it hot out? am I hungry?* → `[1, 0, 1]`.

The robot keeps a little **grade book** (called **W**) that says how much each part of the situation should push each choice up or down. It multiplies the situation by the grade book to get fresh scores, then uses the same softmax spinner from Step 1.

**One-sentence version:** *Same spinner wheel as before, but the slice sizes change based on what's happening right now.*

## 3. Rolling the weighted dice (sampling)

> Grown-up name: **inverse-CDF sampling**. See [math.md](math.md#weighted-sampling).

Once you have chances (67% / 24% / 9%), how do you actually pick? Imagine a raffle:

- Pizza gets tickets numbered 0–66.
- Apple gets 67–90.
- Broccoli gets 91–99.

You spin a random number from 0 to 99 and see whose tickets it landed in. More tickets = more likely to win, but the little guys can still surprise you.

**One-sentence version:** *Give everyone raffle tickets based on their chances, then draw one ticket.*

## 4. Learning from rewards (REINFORCE with a baseline)

> Grown-up name: **REINFORCE-with-baseline**. See [math.md](math.md#reinforce-with-baseline-learner).

This is the heart of the whole thing. After the robot makes a choice, it gets a **reward** — like points in a video game. Good result = high points. Bad result = low points.

The rule is simple: **do more of what earns points, less of what loses them.**

But there's a clever twist called the **baseline**. The baseline is "what I *usually* score." You only get excited if you beat your usual. Two examples:

- You normally score 5. This time you got 8. That's **better than usual** (+3), so do more of that choice.
- You normally score 5. This time you got 2. That's **worse than usual** (−3), so do less of that choice.

Without a baseline, *everything* looks like a win and the robot can't tell great from just-okay. The baseline is the "compared to normal" ruler.

**One-sentence version:** *If a choice does better than your usual, lean into it; if it does worse, back off.*

> Tiny grown-up peek: the nudge is `learning-rate × (reward − baseline) × (how surprising the choice was)`. The "learning rate" is just how big each step is.

## 5. Staying curious (entropy)

> Grown-up name: **entropy** and the **entropy bonus**. See [math.md](math.md#entropy-regularization).

If the robot always picks pizza, it will *never* discover it actually loves tacos. So we add a small reward just for **keeping an open mind** and trying different things now and then.

**Entropy** is a fancy word for "how spread out are my choices?"

- Always picking one thing → **low** entropy (closed-minded).
- Giving everything a fair shot → **high** entropy (curious).

The "entropy bonus" gently pushes the robot to stay a little curious so it doesn't get stuck too soon.

**One-sentence version:** *A little reward for staying curious, so you don't lock onto one answer before exploring.*

## 6. Remembering how you've been doing (moving average baseline)

> Grown-up name: **EMA (exponential moving average) baseline**. See [math.md](math.md#ema-value-baseline).

Remember the "usual score" (baseline) from Step 4? How do we keep it updated? We use a **moving average** — an average that leans toward recent results.

Think of your grade in a class: it's mostly your old average, but each new test nudges it a bit. One great test doesn't erase the whole year, but it moves the number a little.

**One-sentence version:** *Keep a running average of your scores, letting recent results count a little more.*

## 7. Learning fairly from old memories (importance weighting)

> Grown-up name: **importance weighting**. See [math.md](math.md#off-policy-importance-weighting).

Sometimes the robot learns from **old memories** — choices it made back when it was a "younger, different robot." That's a little unfair, because it might not make those same choices today.

So it puts old memories on a scale: memories that still match how it thinks today count fully; memories that don't match get counted less. There's also a cap so no single old memory can shout too loudly.

**One-sentence version:** *When learning from the past, trust memories more if they still match how you'd act today.*

## 8. Turning report cards into one score (reward shaping)

> Grown-up name: **reward shaping**. See [math.md](math.md#reward-shaping).

The robot gets graded by several "judges" on different things (Did it understand the question? Did it stay on task? Did it finish the job?). Each judge gives a grade from **0 to 1**. We need to squish all those into **one** final score.

**Step A — turn grades into good/bad points.** A grade of 0.5 is "meh" (zero points). Above 0.5 is good (plus points); below is bad (minus points):

- Grade 0.8 → **+0.6 points** (nice!)
- Grade 0.5 → **0 points** (meh)
- Grade 0.3 → **−0.4 points** (not great)

**Step B — add them up, but some judges matter more.** Each judge has a "weight" (importance). Multiply and add.

**Step C — add bonuses and penalties.** Was it too slow? Small penalty. Did it route the request correctly? Small bonus.

**Step D — cap it.** The final score is squeezed to stay between −1 and +1 so it's never wild.

**One-sentence version:** *Turn each grade into good/bad points, add them up (some count more), then cap the total.*

## 9. The yes/no squisher (sigmoid)

> Grown-up name: **sigmoid** / **logistic function**. See [math.md](math.md#numerically-stable-sigmoid).

Some parts of the robot answer yes/no questions, like "Did this answer actually help?" They start with a plain number that could be anything, and they need to turn it into a **confidence from 0 to 1** (0% sure to 100% sure).

The **sigmoid** is a machine that squishes any number into that 0-to-1 range using a smooth S-shaped curve:

- A big positive number → close to **1** ("almost certainly yes").
- **0** → exactly **0.5** ("total coin flip").
- A big negative number → close to **0** ("almost certainly no").

**One-sentence version:** *A squisher that turns any number into a "how sure am I?" between 0 and 1.*

## 10. Weighted voting (dot product)

> Grown-up name: **dot product**. See [math.md](math.md#softmax-and-dot-product).

How does the robot turn a situation into a single number to feed the squisher? With a **weighted vote**.

Say the clues are `[2, 5]` and their importances are `[3, 1]`. You multiply each pair and add:

$$3 \times 2 \;+\; 1 \times 5 \;=\; 6 + 5 \;=\; 11$$

Clues that matter more (bigger importance) push the final number more.

**One-sentence version:** *Multiply each clue by how much it matters, then add it all up.*

## 11. Getting better by tiny steps (gradient descent)

> Grown-up name: **gradient descent**. See [math.md](math.md#binary-logistic-regression-mini-batch-sgd).

How do the yes/no parts *learn* the right importances? By practicing on examples and fixing mistakes a little at a time.

Imagine tasting soup and adjusting the salt: too bland → add a pinch; too salty → add water. You never fix it in one giant dump; you nudge and taste, nudge and taste. Each round the guesses get a little closer to right.

The **learning rate** is how big each pinch is. Too big and you overshoot; too small and it takes forever.

**One-sentence version:** *Practice, check how wrong you were, and nudge the settings a tiny bit in the better direction — over and over.*

## 12. Picking one of many (the router)

> Grown-up name: **multinomial logistic regression** / **router**. See [math.md](math.md#router--classifier).

Sometimes you must pick **one** option out of many — like a receptionist deciding which department should handle your call.

The router looks at the situation, gives every option a score, runs the softmax spinner (Step 1) to get chances, and picks the top one. If **no** option looks good enough (the best chance is still low), it's allowed to say **"I'm not sure"** instead of guessing — that's called *refusing*, and it's a feature, not a bug.

**One-sentence version:** *A smart receptionist that scores every option, picks the best — and can say "not sure" if nothing fits.*

## 13. Do these point the same way? (cosine similarity)

> Grown-up name: **cosine similarity**. See [math.md](math.md#prototype-cosine-similarity-mode).

The router has a second way to choose: comparing **arrows**. Picture each option and your situation as arrows.

- Arrows pointing the **same way** → very similar → score near **1**.
- Arrows at a **right angle** → unrelated → score near **0**.
- Arrows pointing **opposite** → opposites → score near **−1**.

Cool part: it only cares about the **direction** the arrow points, not how long it is. So "I love dogs a lot" and "I love dogs a little" point the same way — both are about loving dogs.

**One-sentence version:** *Two things are similar if their arrows point the same direction, no matter how long the arrows are.*

## 14. Turning words into numbers (bag-of-words with a sorting hat)

> Grown-up name: **hashing-trick bag-of-words**. See [math.md](math.md#hashing-trick-bag-of-words-tier-1-stdlib).

Computers can't read words, only numbers. So we **count words** — but with a twist.

A normal way is to keep a giant dictionary listing every word. Instead, we use a **magic sorting hat**: each word gets tossed into one of, say, 1,000 numbered buckets. Then we just count how many words landed in each bucket. That list of bucket-counts becomes the numbers the robot reads.

Why the sorting hat? Because you never have to build or store a dictionary — the hat always sends the same word to the same bucket.

**One-sentence version:** *Toss every word into a numbered bucket and count the buckets — no dictionary needed.*

## 15. Which words actually matter? (TF-IDF)

> Grown-up name: **TF-IDF**. See [math.md](math.md#tf-idf--logistic-regression-tier-2-nlp).

Not all words are useful. Words like "the," "a," and "is" show up everywhere and tell you almost nothing. Rare words that pop up a lot in *one* message are the juicy ones.

TF-IDF gives each word an importance score using two ideas:

- **Shows up a lot in this message?** → more important.
- **Shows up in basically every message?** → less important (probably boring, like "the").

**One-sentence version:** *A word matters if it's common in this message but rare everywhere else.*

## 16. Making grades fair to compare (normalization)

> Grown-up name: **normalization**. See [math.md](math.md#metric-normalization).

Different judges grade on different scales. One gives stars from **1 to 5**; another gives **0 or 1**. To compare fairly, we stretch everyone onto the same **0-to-1** ruler.

For the 1-to-5 judge, we slide it down and shrink it:

- 5 stars → **1.0** (perfect)
- 3 stars → **0.5** (middle)
- 1 star → **0.0** (worst)

Now every grade speaks the same language before we mix them in Step 8.

**One-sentence version:** *Stretch every grade onto the same 0-to-1 ruler so comparisons are fair.*

## Putting it all together

Here's the whole robot in one breath:

1. A situation comes in, written as numbers (**phi**, Step 2).
2. The robot scores each choice and turns scores into chances (**softmax**, Step 1).
3. It draws from a weighted raffle to pick (**sampling**, Step 3).
4. The result gets graded by judges (**sigmoid** yes/no answers, Step 9), each grade stretched onto the same ruler (**normalization**, Step 16).
5. The grades are mixed into one reward (**reward shaping**, Step 8).
6. The robot compares that reward to its usual (**baseline**, Steps 4 & 6) and nudges itself to do more of what worked (**learning**, Steps 4 & 11), while staying a little curious (**entropy**, Step 5).
7. Over many rounds, it gets steadily better.

That's it. Everything in [math.md](math.md) is just a precise way of writing down these seven simple ideas.

## How to explain this to someone in 30 seconds

> "It's a robot that makes choices like a spinner wheel — better options get bigger slices. After each choice it gets points, like a video game. If a choice beat its usual score, it makes that slice bigger next time; if it did worse, it shrinks the slice. It stays a little curious so it keeps exploring, and it uses simple graders to decide how many points each result earns. Do that thousands of times and it slowly learns the best choices."
