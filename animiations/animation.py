#!/usr/bin/env python3
"""
Agent Learning — Native reinforcement learning for AI agents (Manim).

Visualises the three-part learning loop described in the project
README. Each act on screen maps to one README bullet:

1. POLICY  — a softmax distribution over ``N`` discrete agent actions
             (e.g. "use prompt template A / B / C"). It lives in
             Python and updates in milliseconds.

2. JUDGES  — every episode is scored by three Azure AI Evaluation
             evaluators — ``IntentResolutionEvaluator``,
             ``TaskAdherenceEvaluator`` and ``TaskCompletionEvaluator``
             — whose scores are shaped into a single scalar reward.

3. LEARNER — a REINFORCE-with-baseline learner nudges the policy
             logits directly from logged episodes: tiny gradient steps
             that run on CPU and persist through a pluggable store
             (in-memory / local files / Cosmos DB).

The script runs a small, self-contained softmax-bandit simulation so
the probability bars, judge scores, rewards and logit updates shown on
screen are *real* numbers produced by the same REINFORCE update the
SDK ships (see ``src/agent_learning``). No Azure credentials, no log
file and no LaTeX toolchain are required — every label is drawn with
Manim ``Text`` so the animation renders standalone.

Usage:
    manim -pql animation.py AgentLearning
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from manim import *

# ── Palette ─────────────────────────────────────────────────────────
BG = "#0d1117"
ACCENT = "#6ee7d0"
MUTED = "#8b949e"
PANEL = "#161b22"
REWARD_COLOR = "#f2c94c"
LEARNER_COLOR = "#e85c5c"

# Per-action colours (Prompt A / B / C), reused across every act.
ACTION_COLORS = ["#4c9be8", "#c792ea", "#5fb878"]
ACTIONS = ["Prompt A", "Prompt B", "Prompt C"]
ACTION_SHORT = ["A", "B", "C"]

# Hidden "true quality" of each template — unknown to the policy; it is
# what the judges implicitly measure and what the learner discovers.
TRUE_QUALITY = [0.35, 0.55, 0.82]

# Judge colours + names (three Azure AI Evaluation evaluators).
JUDGE_NAMES = ["IntentResolution", "TaskAdherence", "TaskCompletion"]
JUDGE_COLORS = ["#e8a24c", "#5fb878", "#c792ea"]

# Reward-shaping weights — match src/agent_learning/config.py
# ShapingConfig defaults (biased toward task completion).
W_INTENT = 0.10
W_ADHERE = 0.20
W_COMPLETE = 0.50

# Learner hyper-parameters — match LearnerConfig defaults, except the
# on-screen learning rate is amplified so the bars visibly converge in
# a handful of batches (the SDK default is 0.05).
LR_DISPLAY = 0.35
BASELINE_DECAY = 0.9
ENTROPY_BONUS = 0.01

# Fonts available on Windows; Manim falls back gracefully otherwise.
SANS = "Segoe UI"
MONO = "Consolas"


# ── Self-contained softmax-bandit simulation ────────────────────────
class SoftmaxSim:
    """Mirror of the SDK's SoftmaxPolicy + RewardShaper + ReinforceLearner.

    Kept deliberately tiny so the animation can drive real numbers
    through the exact update equations documented in the SDK source.
    """

    def __init__(self, n: int, lr: float, decay: float, entropy: float, seed: int = 7) -> None:
        self.n = n
        self.lr = lr
        self.decay = decay
        self.entropy = entropy
        self.logits: List[float] = [0.0] * n
        self.baseline = 0.0
        self.rng = random.Random(seed)

    def probs(self) -> List[float]:
        m = max(self.logits)
        exps = [math.exp(z - m) for z in self.logits]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    def choose(self) -> int:
        p = self.probs()
        roll = self.rng.random()
        cumulative = 0.0
        for i, pi in enumerate(p):
            cumulative += pi
            if roll <= cumulative:
                return i
        return self.n - 1

    def judge(self, action: int) -> Tuple[float, float, float]:
        """Three normalised judge scores in [0, 1] for the chosen action."""
        q = TRUE_QUALITY[action]

        def score() -> float:
            return max(0.0, min(1.0, q + self.rng.uniform(-0.12, 0.12)))

        return score(), score(), score()

    def shape(self, scores: Tuple[float, float, float]) -> float:
        """Weighted sum of signed judge scores, clamped to [-1, 1]."""
        intent, adhere, complete = scores
        total = (
            W_INTENT * (2.0 * intent - 1.0)
            + W_ADHERE * (2.0 * adhere - 1.0)
            + W_COMPLETE * (2.0 * complete - 1.0)
        )
        return max(-1.0, min(1.0, total))

    def update_batch(self, episodes: List[Tuple[int, float]]) -> float:
        """One REINFORCE-with-baseline step over a batch of episodes."""
        p = self.probs()
        entropy = -sum(pi * math.log(max(pi, 1e-12)) for pi in p)
        deltas = [0.0] * self.n
        reward_sum = 0.0
        for action, reward in episodes:
            advantage = reward - self.baseline
            for i in range(self.n):
                indicator = 1.0 if i == action else 0.0
                deltas[i] += advantage * (indicator - p[i])
                deltas[i] += self.entropy * (-math.log(max(p[i], 1e-12)) - entropy)
            reward_sum += reward
        used = max(len(episodes), 1)
        for i in range(self.n):
            self.logits[i] += self.lr * deltas[i] / used
            self.logits[i] = max(-10.0, min(10.0, self.logits[i]))
        mean_reward = reward_sum / used
        self.baseline = self.decay * self.baseline + (1.0 - self.decay) * mean_reward
        return mean_reward


# ── Manim scene ─────────────────────────────────────────────────────
class AgentLearning(Scene):
    """Three-act explainer of the native agent-learning loop."""

    def construct(self) -> None:
        self.camera.background_color = BG
        random.seed(7)

        self.sim = SoftmaxSim(len(ACTIONS), LR_DISPLAY, BASELINE_DECAY, ENTROPY_BONUS, seed=7)
        self.logit_trackers = [ValueTracker(0.0) for _ in ACTIONS]
        self.judge_trackers = [ValueTracker(0.0) for _ in range(3)]
        self.reward_tracker = ValueTracker(0.0)
        self.baseline_tracker = ValueTracker(0.0)
        self.redraws: list = []
        self.chosen = 0
        self.scores: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.reward_value = 0.0

        self._intro()
        self._act_policy()
        self._act_judges()
        self._act_learner()
        self._closing()

    # ── Shared helpers ─────────────────────────────────────────────
    def _probs(self) -> List[float]:
        zs = [t.get_value() for t in self.logit_trackers]
        m = max(zs)
        exps = [math.exp(z - m) for z in zs]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    def _header(self, num: int, title: str, subtitle: str, color: str) -> VGroup:
        badge = RoundedRectangle(
            width=0.7, height=0.7, corner_radius=0.12,
            stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.15,
        )
        n = Text(str(num), font=SANS, font_size=34, color=color, weight=BOLD)
        n.move_to(badge.get_center())
        badge_grp = VGroup(badge, n)
        t = Text(title, font=SANS, font_size=30, color=WHITE, weight=BOLD)
        s = Text(subtitle, font=SANS, font_size=19, color=MUTED)
        text_grp = VGroup(t, s).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        grp = VGroup(badge_grp, text_grp).arrange(RIGHT, buff=0.3)
        grp.to_corner(UL, buff=0.5)
        return grp

    def _fill_rect(self, i: int, left_x: float, y: float, width: float, height: float) -> Rectangle:
        p = self._probs()[i]
        w = max(p * width, 0.03)
        r = Rectangle(
            width=w, height=height - 0.14, stroke_width=0,
            fill_color=ACTION_COLORS[i], fill_opacity=0.95,
        )
        r.move_to([left_x + 0.02, y, 0], aligned_edge=LEFT)
        return r

    def _pct_text(self, i: int, x: float, y: float) -> Text:
        return Text(
            f"{self._probs()[i] * 100:5.1f}%", font=MONO, font_size=22, color=ACTION_COLORS[i]
        ).move_to([x, y, 0])

    def _make_bars(self, left_x: float, rows_y: List[float], width: float = 2.8,
                   height: float = 0.5, show_short: bool = True) -> VGroup:
        group = VGroup()
        for y in rows_y:
            track = RoundedRectangle(
                width=width, height=height, corner_radius=0.1,
                stroke_color=MUTED, stroke_width=1.5, fill_color=PANEL, fill_opacity=1.0,
            )
            track.move_to([left_x + width / 2.0, y, 0])
            group.add(track)
        for i, y in enumerate(rows_y):
            fill = always_redraw(lambda i=i, y=y: self._fill_rect(i, left_x, y, width, height))
            self.redraws.append(fill)
            group.add(fill)
        if show_short:
            for i, y in enumerate(rows_y):
                lbl = Text(ACTION_SHORT[i], font=SANS, font_size=24, color=WHITE, weight=BOLD)
                lbl.move_to([left_x - 0.4, y, 0])
                group.add(lbl)
        for i, y in enumerate(rows_y):
            pct = always_redraw(lambda i=i, y=y: self._pct_text(i, left_x + width + 0.7, y))
            self.redraws.append(pct)
            group.add(pct)
        return group

    def _transition(self) -> None:
        for m in self.redraws:
            m.clear_updaters()
        self.redraws = []
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    # ── Intro ──────────────────────────────────────────────────────
    def _intro(self) -> None:
        title = Text("Agent Learning", font=SANS, font_size=64, color=WHITE, weight=BOLD)
        sub = Text(
            "Native reinforcement learning for AI agents",
            font=SANS, font_size=28, color=ACCENT,
        )
        sub.next_to(title, DOWN, buff=0.3)
        grp = VGroup(title, sub).move_to(UP * 0.4)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)

        steps = VGroup(
            Text("policy", font=SANS, font_size=26, color=ACTION_COLORS[0], weight=BOLD),
            Text("→", font=SANS, font_size=26, color=MUTED),
            Text("judges", font=SANS, font_size=26, color=JUDGE_COLORS[2], weight=BOLD),
            Text("→", font=SANS, font_size=26, color=MUTED),
            Text("learner", font=SANS, font_size=26, color=LEARNER_COLOR, weight=BOLD),
        ).arrange(RIGHT, buff=0.28).next_to(grp, DOWN, buff=0.8)
        self.play(FadeIn(steps, shift=UP * 0.2), run_time=0.7)
        self.wait(0.9)
        self._transition()

    # ── Act 1 · Policy ─────────────────────────────────────────────
    def _act_policy(self) -> None:
        header = self._header(
            1, "Policy", "softmax distribution over N discrete actions", ACTION_COLORS[0]
        )
        self.play(FadeIn(header), run_time=0.5)

        rows_y = [1.4, 0.1, -1.2]
        boxes_cx = -4.6
        bars_left = 0.3
        bars_w = 3.0

        boxes = VGroup()
        for i, y in enumerate(rows_y):
            box = RoundedRectangle(
                width=2.9, height=1.0, corner_radius=0.12,
                stroke_color=ACTION_COLORS[i], stroke_width=2,
                fill_color=ACTION_COLORS[i], fill_opacity=0.08,
            )
            box.move_to([boxes_cx, y, 0])
            name = Text(ACTIONS[i], font=SANS, font_size=22, color=WHITE, weight=BOLD)
            name.move_to([boxes_cx, y + 0.18, 0])
            zlbl = always_redraw(
                lambda i=i, y=y: Text(
                    f"logit z = {self.logit_trackers[i].get_value():+.2f}",
                    font=MONO, font_size=18, color=MUTED,
                ).move_to([boxes_cx, y - 0.22, 0])
            )
            self.redraws.append(zlbl)
            boxes.add(box, name, zlbl)

        arrows = VGroup(
            *[
                Arrow(
                    [boxes_cx + 1.55, y, 0], [bars_left - 0.15, y, 0],
                    buff=0.1, color=MUTED, stroke_width=3,
                    max_tip_length_to_length_ratio=0.12,
                )
                for y in rows_y
            ]
        )
        softmax_lbl = Text("softmax", font=SANS, font_size=24, color=ACCENT, weight=BOLD)
        softmax_lbl.move_to([-1.9, 0.85, 0])
        softmax_eq = Text("π(a) = softmax(z)", font=MONO, font_size=20, color=MUTED)
        softmax_eq.move_to([-1.9, -0.95, 0])

        bars = self._make_bars(bars_left, rows_y, width=bars_w, show_short=False)

        self.play(FadeIn(boxes, shift=RIGHT * 0.2), run_time=0.7)
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), GrowArrow(arrows[2]),
                  FadeIn(softmax_lbl), FadeIn(softmax_eq), run_time=0.6)
        self.play(FadeIn(bars), run_time=0.6)

        note = Text(
            "Untrained logits are all 0  →  the policy is uniform (⅓ each).",
            font=SANS, font_size=22, color=MUTED,
        ).move_to([0, -3.0, 0])
        self.play(FadeIn(note), run_time=0.5)
        self.wait(0.6)

        # Demonstrate the softmax map: raise one logit, watch its bar grow.
        demo = Text(
            "Raise a logit  →  its probability rises (softmax).",
            font=SANS, font_size=22, color=WHITE,
        ).move_to([0, -3.0, 0])
        self.play(Transform(note, demo), self.logit_trackers[2].animate.set_value(1.4), run_time=1.1)
        self.wait(0.4)
        self.play(self.logit_trackers[2].animate.set_value(0.0), run_time=0.9)

        # choose(): sample one action from the distribution.
        self.chosen = self.sim.choose()
        choose_lbl = Text("policy.choose()", font=MONO, font_size=24, color=REWARD_COLOR, weight=BOLD)
        choose_lbl.move_to([0, -3.0, 0])
        self.play(Transform(note, choose_lbl), run_time=0.4)

        selector = RoundedRectangle(
            width=bars_w + 1.1, height=0.72, corner_radius=0.12,
            stroke_color=REWARD_COLOR, stroke_width=3.5,
        )
        sel_x = bars_left + bars_w / 2.0 + 0.15
        hop_order = [0, 1, 2, 0, 1, 2, self.chosen]
        selector.move_to([sel_x, rows_y[hop_order[0]], 0])
        self.play(Create(selector), run_time=0.3)
        for idx in hop_order[1:]:
            self.play(selector.animate.move_to([sel_x, rows_y[idx], 0]), run_time=0.18)

        picked = Text(
            f"sampled action:  {ACTIONS[self.chosen]}",
            font=SANS, font_size=24, color=REWARD_COLOR, weight=BOLD,
        ).move_to([0, -3.0, 0])
        self.play(
            Transform(note, picked),
            Indicate(boxes[3 * self.chosen], color=REWARD_COLOR, scale_factor=1.12),
            run_time=0.8,
        )
        self.wait(1.0)
        self._transition()

    # ── Act 2 · Judges → reward ────────────────────────────────────
    def _act_judges(self) -> None:
        header = self._header(
            2, "Judges", "three evaluators combine into one scalar reward", JUDGE_COLORS[2]
        )
        self.play(FadeIn(header), run_time=0.5)

        episode = RoundedRectangle(
            width=4.0, height=0.9, corner_radius=0.12,
            stroke_color=ACTION_COLORS[self.chosen], stroke_width=2,
            fill_color=ACTION_COLORS[self.chosen], fill_opacity=0.1,
        ).move_to([0, 2.75, 0])
        ep_lbl = Text(
            f"episode · {ACTIONS[self.chosen]} output",
            font=SANS, font_size=22, color=WHITE, weight=BOLD,
        ).move_to(episode.get_center())
        self.play(FadeIn(VGroup(episode, ep_lbl), shift=DOWN * 0.2), run_time=0.6)

        self.scores = self.sim.judge(self.chosen)
        self.reward_value = self.sim.shape(self.scores)
        weights = [W_INTENT, W_ADHERE, W_COMPLETE]
        centers = [-4.5, 0.0, 4.5]

        cards = VGroup()
        gauge_w = 2.9
        for i, cx in enumerate(centers):
            card = RoundedRectangle(
                width=3.4, height=1.7, corner_radius=0.12,
                stroke_color=JUDGE_COLORS[i], stroke_width=2,
                fill_color=JUDGE_COLORS[i], fill_opacity=0.07,
            ).move_to([cx, 1.05, 0])
            name = Text(JUDGE_NAMES[i], font=SANS, font_size=21, color=JUDGE_COLORS[i], weight=BOLD)
            name.move_to([cx, 1.55, 0])
            track = RoundedRectangle(
                width=gauge_w, height=0.34, corner_radius=0.08,
                stroke_color=MUTED, stroke_width=1.2, fill_color=PANEL, fill_opacity=1.0,
            ).move_to([cx, 1.0, 0])
            gleft = cx - gauge_w / 2.0
            fill = always_redraw(
                lambda i=i, gleft=gleft: self._gauge_fill(i, gleft, 1.0, gauge_w, 0.28)
            )
            self.redraws.append(fill)
            sval = always_redraw(
                lambda i=i, cx=cx: Text(
                    f"score = {self.judge_trackers[i].get_value():.2f}",
                    font=MONO, font_size=18, color=WHITE,
                ).move_to([cx, 0.55, 0])
            )
            self.redraws.append(sval)
            wlbl = Text(f"weight w = {weights[i]:.2f}", font=MONO, font_size=17, color=MUTED)
            wlbl.move_to([cx, 0.28, 0])
            cards.add(VGroup(card, name, track, fill, sval, wlbl))

        self.play(FadeIn(cards, shift=UP * 0.2), run_time=0.7)
        self.play(
            *[self.judge_trackers[i].animate.set_value(self.scores[i]) for i in range(3)],
            run_time=1.2,
        )
        self.wait(0.5)

        s0, s1, s2 = self.scores
        shape_line = Text(
            "shape:  signed sᵢ = 2·sᵢ − 1        reward R = clip( Σ wᵢ·signed sᵢ ,  −1, +1 )",
            font=SANS, font_size=20, color=WHITE,
        ).move_to([0, -0.55, 0])
        calc_line = Text(
            f"{W_INTENT:.2f}·({2 * s0 - 1:+.2f})  +  {W_ADHERE:.2f}·({2 * s1 - 1:+.2f})  +  "
            f"{W_COMPLETE:.2f}·({2 * s2 - 1:+.2f})   →   R = {self.reward_value:+.2f}",
            font=MONO, font_size=20, color=ACCENT,
        ).move_to([0, -1.15, 0])
        self.play(FadeIn(shape_line), run_time=0.6)
        self.play(FadeIn(calc_line), run_time=0.6)

        # Reward meter from -1 to +1.
        meter_cx, meter_cy, halfw = 0.0, -2.25, 3.2
        track = RoundedRectangle(
            width=2 * halfw, height=0.5, corner_radius=0.12,
            stroke_color=MUTED, stroke_width=1.5, fill_color=PANEL, fill_opacity=1.0,
        ).move_to([meter_cx, meter_cy, 0])
        zero = Line([meter_cx, meter_cy - 0.35, 0], [meter_cx, meter_cy + 0.35, 0],
                    color=WHITE, stroke_width=2)
        neg = Text("−1", font=MONO, font_size=18, color=LEARNER_COLOR).move_to([-halfw - 0.45, meter_cy, 0])
        pos = Text("+1", font=MONO, font_size=18, color=REWARD_COLOR).move_to([halfw + 0.45, meter_cy, 0])
        fill = always_redraw(lambda: self._reward_fill(meter_cx, meter_cy, halfw, 0.42))
        self.redraws.append(fill)
        rval = always_redraw(
            lambda: Text(
                f"R = {self.reward_tracker.get_value():+.2f}",
                font=MONO, font_size=24, color=REWARD_COLOR, weight=BOLD,
            ).move_to([0, meter_cy - 0.75, 0])
        )
        self.redraws.append(rval)
        self.play(FadeIn(VGroup(track, zero, neg, pos, fill, rval)), run_time=0.5)
        self.play(self.reward_tracker.animate.set_value(self.reward_value), run_time=1.0)
        self.wait(1.1)
        self._transition()

    def _gauge_fill(self, i: int, left_x: float, y: float, width: float, height: float) -> Rectangle:
        v = self.judge_trackers[i].get_value()
        w = max(v * width, 0.02)
        r = Rectangle(width=w, height=height, stroke_width=0,
                      fill_color=JUDGE_COLORS[i], fill_opacity=0.9)
        r.move_to([left_x, y, 0], aligned_edge=LEFT)
        return r

    def _reward_fill(self, cx: float, cy: float, halfw: float, height: float) -> Rectangle:
        R = self.reward_tracker.get_value()
        w = max(abs(R) * halfw, 0.02)
        col = REWARD_COLOR if R >= 0 else LEARNER_COLOR
        r = Rectangle(width=w, height=height, stroke_width=0, fill_color=col, fill_opacity=0.9)
        r.move_to([cx, cy, 0], aligned_edge=LEFT if R >= 0 else RIGHT)
        return r

    # ── Act 3 · Learner ────────────────────────────────────────────
    def _act_learner(self) -> None:
        header = self._header(
            3, "Learner", "REINFORCE-with-baseline nudges the logits", LEARNER_COLOR
        )
        self.play(FadeIn(header), run_time=0.5)

        rule = Text(
            "Δz_a  =  α · (R − b) · ( 1[a] − π(a) )",
            font=MONO, font_size=26, color=WHITE,
        ).move_to([1.6, 3.0, 0])
        rule_note = Text(
            "tiny CPU gradient step · b = EMA value baseline",
            font=SANS, font_size=18, color=MUTED,
        ).next_to(rule, DOWN, buff=0.12)
        self.play(FadeIn(rule), FadeIn(rule_note), run_time=0.6)

        rows_y = [1.5, 0.4, -0.7]
        bars_left = -6.2
        bars_w = 2.6
        bars = self._make_bars(bars_left, rows_y, width=bars_w, show_short=True)
        self.play(FadeIn(bars), run_time=0.6)

        readout = always_redraw(
            lambda: Text(
                "  ".join(
                    f"z{ACTION_SHORT[i]}={self.logit_trackers[i].get_value():+.2f}"
                    for i in range(len(ACTIONS))
                )
                + f"    b={self.baseline_tracker.get_value():+.2f}",
                font=MONO, font_size=18, color=MUTED,
            ).move_to([-4.0, -1.7, 0])
        )
        self.redraws.append(readout)
        self.play(FadeIn(readout), run_time=0.3)

        # Reward curve frame (right side).
        ax_x0, ax_x1 = 0.4, 6.4
        ax_y0, ax_y1 = -1.2, 1.7
        frame = Rectangle(width=ax_x1 - ax_x0, height=ax_y1 - ax_y0,
                          stroke_color=MUTED, stroke_width=1.5)
        frame.move_to([(ax_x0 + ax_x1) / 2, (ax_y0 + ax_y1) / 2, 0])
        ymid = (ax_y0 + ax_y1) / 2
        halfspan = (ax_y1 - ax_y0) / 2 - 0.2
        zero_line = DashedLine([ax_x0, ymid, 0], [ax_x1, ymid, 0],
                               color=MUTED, stroke_width=1.5, dash_length=0.1)
        y_top = Text("mean reward", font=SANS, font_size=17, color=REWARD_COLOR)
        y_top.next_to(frame, UP, buff=0.12).align_to(frame, LEFT)
        x_lbl = Text("offline batches  →", font=SANS, font_size=17, color=MUTED)
        x_lbl.next_to(frame, DOWN, buff=0.12)
        self.play(Create(frame), Create(zero_line), FadeIn(y_top), FadeIn(x_lbl), run_time=0.6)

        # First learner step uses the exact episode judged in Act 2.
        step_lbl = Text(
            "learner.update(policy, episodes, rewards)",
            font=MONO, font_size=20, color=LEARNER_COLOR,
        ).move_to([-4.0, -2.5, 0])
        self.play(FadeIn(step_lbl), run_time=0.4)
        self.sim.update_batch([(self.chosen, self.reward_value)])
        self.play(
            *[self.logit_trackers[i].animate.set_value(self.sim.logits[i]) for i in range(len(ACTIONS))],
            self.baseline_tracker.animate.set_value(self.sim.baseline),
            run_time=1.0,
        )
        self.wait(0.4)

        # Roll the loop forward over several offline batches.
        loop_lbl = Text(
            "runner.run_offline_batch(agent_id)  ·  repeating…",
            font=MONO, font_size=20, color=ACCENT,
        ).move_to([-4.0, -2.5, 0])
        self.play(Transform(step_lbl, loop_lbl), run_time=0.4)

        n_batches = 10
        ep_per_batch = 24
        keyframes = []
        for _ in range(n_batches):
            episodes = []
            for _ in range(ep_per_batch):
                a = self.sim.choose()
                sc = self.sim.judge(a)
                episodes.append((a, self.sim.shape(sc)))
            mean_r = self.sim.update_batch(episodes)
            keyframes.append((mean_r, list(self.sim.logits), self.sim.baseline))

        def cx(b: int) -> float:
            return ax_x0 + 0.35 + (b / max(n_batches - 1, 1)) * (ax_x1 - ax_x0 - 0.7)

        def cy(r: float) -> float:
            return ymid + max(-1.0, min(1.0, r)) * halfspan

        prev_point = None
        for b, (mean_r, logits, baseline) in enumerate(keyframes):
            point = [cx(b), cy(mean_r), 0]
            dot = Dot(point, radius=0.05, color=REWARD_COLOR)
            anims = [
                *[self.logit_trackers[i].animate.set_value(logits[i]) for i in range(len(ACTIONS))],
                self.baseline_tracker.animate.set_value(baseline),
                FadeIn(dot),
            ]
            if prev_point is not None:
                anims.append(Create(Line(prev_point, point, color=REWARD_COLOR, stroke_width=3)))
            self.play(*anims, run_time=0.36)
            prev_point = point

        # Persist to the pluggable store.
        store = RoundedRectangle(
            width=6.6, height=0.8, corner_radius=0.14,
            stroke_color=ACCENT, stroke_width=2, fill_color=ACCENT, fill_opacity=0.08,
        ).move_to([0, -3.35, 0])
        store_lbl = Text(
            "persist policy + lineage  →  store  (in-memory · local files · Cosmos DB)",
            font=SANS, font_size=19, color=WHITE,
        ).move_to(store.get_center())
        self.play(FadeIn(VGroup(store, store_lbl), shift=UP * 0.15), run_time=0.6)

        best = max(range(len(ACTIONS)), key=lambda i: self.sim.logits[i])
        verdict = Text(
            f"policy now favours {ACTIONS[best]} — the highest-reward template.",
            font=SANS, font_size=20, color=ACTION_COLORS[best], weight=BOLD,
        ).move_to([-4.0, -2.5, 0])
        self.play(Transform(step_lbl, verdict), run_time=0.6)
        self.wait(1.3)
        self._transition()

    # ── Closing ────────────────────────────────────────────────────
    def _closing(self) -> None:
        def mini_card(title: str, sub: str, color: str) -> VGroup:
            box = RoundedRectangle(
                width=3.1, height=1.5, corner_radius=0.15,
                stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.1,
            )
            t = Text(title, font=SANS, font_size=28, color=color, weight=BOLD)
            s = Text(sub, font=SANS, font_size=16, color=MUTED)
            VGroup(t, s).arrange(DOWN, buff=0.14).move_to(box.get_center())
            return VGroup(box, t, s)

        policy = mini_card("Policy", "softmax over N actions", ACTION_COLORS[0])
        judges = mini_card("Judges", "3 evaluators → reward", JUDGE_COLORS[2])
        learner = mini_card("Learner", "REINFORCE + baseline", LEARNER_COLOR)
        row = VGroup(policy, judges, learner).arrange(RIGHT, buff=1.3).move_to(UP * 0.6)

        a1 = Arrow(policy.get_right(), judges.get_left(), buff=0.15, color=MUTED, stroke_width=4)
        a2 = Arrow(judges.get_right(), learner.get_left(), buff=0.15, color=MUTED, stroke_width=4)
        back = CurvedArrow(
            learner.get_bottom() + DOWN * 0.15, policy.get_bottom() + DOWN * 0.15,
            angle=-TAU / 6, color=ACCENT, stroke_width=4,
        )
        back_lbl = Text("improve every batch", font=SANS, font_size=18, color=ACCENT)
        back_lbl.next_to(back, DOWN, buff=0.1)

        self.play(FadeIn(policy), run_time=0.4)
        self.play(GrowArrow(a1), FadeIn(judges), run_time=0.5)
        self.play(GrowArrow(a2), FadeIn(learner), run_time=0.5)
        self.play(Create(back), FadeIn(back_lbl), run_time=0.7)

        tagline = Text(
            "No weight fine-tuning.  No GPUs.  Just tiny CPU gradient steps.",
            font=SANS, font_size=26, color=WHITE, weight=BOLD,
        ).move_to(DOWN * 2.3)
        name = Text("agents-learning-sdk", font=MONO, font_size=22, color=ACCENT)
        name.next_to(tagline, DOWN, buff=0.3)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(name), run_time=0.5)
        self.wait(2.0)
