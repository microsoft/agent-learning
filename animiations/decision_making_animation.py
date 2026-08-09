#!/usr/bin/env python3
"""Animate an explicit agent decision improving from measured outcomes.

The scene uses a small softmax policy and a REINFORCE-with-baseline update to
show the product workflow: choose, execute, score, and improve. The foundation
model is never trained.

Usage:
    manim -pql decision_making_animation.py AgentDecisionMaking
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from manim import *

BG = "#18213a"
PANEL = "#10192c"
PANEL_2 = "#0b1220"
WHITE_SOFT = "#d8e2f1"
MUTED = "#aebbd2"
BLUE = "#4cc2ff"
GREEN = "#7fba00"
YELLOW = "#ffb900"
ORANGE = "#f25022"
TRACK = "#293750"

ACTION_COLORS = ["#8aa0be", BLUE, GREEN]
ACTIONS = ["Text search", "Semantic search", "Symbol search"]
ACTION_IDS = ["text", "semantic", "symbol"]
TRUE_QUALITY = [0.52, 0.91, 0.68]
SCORE_NAMES = ["Intent", "Adherence", "Completion"]
SCORE_WEIGHTS = [0.10, 0.20, 0.50]

SANS = "Segoe UI"
MONO = "Consolas"


class DecisionSim:
    """Small deterministic mirror of the decision policy and learner."""

    def __init__(self, seed: int = 11) -> None:
        self.logits: List[float] = [0.0] * len(ACTIONS)
        self.baseline = 0.0
        self.rng = random.Random(seed)

    def probabilities(self) -> List[float]:
        maximum = max(self.logits)
        exponentials = [math.exp(value - maximum) for value in self.logits]
        total = sum(exponentials) or 1.0
        return [value / total for value in exponentials]

    def choose(self) -> int:
        roll = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(self.probabilities()):
            cumulative += probability
            if roll <= cumulative:
                return index
        return len(ACTIONS) - 1

    def score(self, action: int) -> Tuple[float, float, float]:
        quality = TRUE_QUALITY[action]
        return tuple(
            max(0.0, min(1.0, quality + self.rng.uniform(-0.035, 0.035)))
            for _ in SCORE_NAMES
        )

    @staticmethod
    def reward(scores: Tuple[float, float, float]) -> float:
        shaped = sum(
            weight * (2.0 * score - 1.0)
            for weight, score in zip(SCORE_WEIGHTS, scores)
        )
        return max(-1.0, min(1.0, shaped))

    def update(self, episodes: List[Tuple[int, float]], learning_rate: float = 0.75) -> None:
        probabilities = self.probabilities()
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
        )
        deltas = [0.0] * len(ACTIONS)

        for action, reward in episodes:
            advantage = reward - self.baseline
            for index, probability in enumerate(probabilities):
                selected = 1.0 if index == action else 0.0
                deltas[index] += advantage * (selected - probability)
                deltas[index] += 0.01 * (
                    -math.log(max(probability, 1e-12)) - entropy
                )

        count = max(len(episodes), 1)
        for index in range(len(ACTIONS)):
            self.logits[index] += learning_rate * deltas[index] / count
            self.logits[index] = max(-10.0, min(10.0, self.logits[index]))

        mean_reward = sum(reward for _, reward in episodes) / count
        self.baseline = 0.9 * self.baseline + 0.1 * mean_reward


class AgentDecisionMaking(Scene):
    """Four-act explainer for agentic decision making with feedback."""

    def construct(self) -> None:
        self.camera.background_color = BG
        self.sim = DecisionSim(seed=11)
        self.before_probabilities = self.sim.probabilities()
        self.selected_action = 0
        self.scores: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.reward = 0.0

        self._intro()
        self._choose()
        self._execute()
        self._score()
        self._improve()
        self._closing()

    def _stage_header(self, number: str, title: str, subtitle: str, color: str) -> VGroup:
        badge = RoundedRectangle(
            width=0.72,
            height=0.72,
            corner_radius=0.1,
            stroke_color=color,
            stroke_width=2.5,
            fill_color=color,
            fill_opacity=0.14,
        )
        number_text = Text(number, font=MONO, font_size=21, color=color, weight=BOLD)
        number_text.move_to(badge)
        title_text = Text(title, font=SANS, font_size=31, color=WHITE, weight=BOLD)
        subtitle_text = Text(subtitle, font=SANS, font_size=18, color=MUTED)
        words = VGroup(title_text, subtitle_text).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        header = VGroup(VGroup(badge, number_text), words).arrange(RIGHT, buff=0.3)
        header.to_corner(UL, buff=0.45)
        return header

    def _clear(self) -> None:
        if self.mobjects:
            self.play(*[FadeOut(item) for item in self.mobjects], run_time=0.55)

    def _probability_rows(
        self,
        probabilities: List[float],
        left: float,
        top: float,
        width: float,
        selected: int | None = None,
    ) -> VGroup:
        rows = VGroup()
        for index, (name, probability) in enumerate(zip(ACTIONS, probabilities)):
            y = top - index * 0.92
            is_selected = index == selected
            label = Text(
                name,
                font=SANS,
                font_size=20,
                color=WHITE if is_selected else WHITE_SOFT,
                weight=BOLD if is_selected else NORMAL,
            ).move_to([left, y + 0.22, 0], aligned_edge=LEFT)
            track = RoundedRectangle(
                width=width,
                height=0.27,
                corner_radius=0.06,
                stroke_width=0,
                fill_color=TRACK,
                fill_opacity=1.0,
            ).move_to([left + width / 2.0, y - 0.17, 0])
            fill = RoundedRectangle(
                width=max(width * probability, 0.04),
                height=0.27,
                corner_radius=0.06,
                stroke_width=0,
                fill_color=ACTION_COLORS[index],
                fill_opacity=1.0,
            ).move_to([left, y - 0.17, 0], aligned_edge=LEFT)
            percent = Text(
                f"{probability * 100:4.1f}%",
                font=MONO,
                font_size=18,
                color=ACTION_COLORS[index],
            ).move_to([left + width + 0.65, y - 0.17, 0])
            row = VGroup(label, track, fill, percent)
            if is_selected:
                outline = RoundedRectangle(
                    width=width + 1.55,
                    height=0.78,
                    corner_radius=0.12,
                    stroke_color=YELLOW,
                    stroke_width=2.2,
                ).move_to([left + (width + 0.55) / 2.0, y, 0])
                row.add(outline)
            rows.add(row)
        return rows

    def _intro(self) -> None:
        eyebrow = Text(
            "HELP AGENTS MAKE BETTER REPEATABLE CHOICES",
            font=SANS,
            font_size=18,
            color=BLUE,
            weight=BOLD,
        )
        title = Text(
            "Agentic Decision Making",
            font=SANS,
            font_size=56,
            color=WHITE,
            weight=BOLD,
        )
        subtitle = Text(
            "Explicit choices. Measured outcomes. Better next decisions.",
            font=SANS,
            font_size=25,
            color=WHITE_SOFT,
        )
        heading = VGroup(eyebrow, title, subtitle).arrange(DOWN, buff=0.25)
        heading.move_to(UP * 1.05)

        chips = VGroup(
            self._chip("No model fine-tuning", BLUE),
            self._chip("No GPU jobs", GREEN),
            self._chip("Small CPU updates", YELLOW),
        ).arrange(RIGHT, buff=0.35).next_to(heading, DOWN, buff=0.75)

        self.play(FadeIn(eyebrow, shift=UP * 0.15), run_time=0.5)
        self.play(Write(title), run_time=0.9)
        self.play(FadeIn(subtitle), FadeIn(chips, shift=UP * 0.15), run_time=0.7)
        self.wait(1.1)
        self._clear()

    def _chip(self, label: str, color: str) -> VGroup:
        box = RoundedRectangle(
            width=3.2,
            height=0.62,
            corner_radius=0.12,
            stroke_color=color,
            stroke_width=1.8,
            fill_color=color,
            fill_opacity=0.09,
        )
        text = Text(label, font=SANS, font_size=17, color=WHITE_SOFT)
        text.move_to(box)
        return VGroup(box, text)

    def _choose(self) -> None:
        header = self._stage_header(
            "01",
            "Choose",
            "sample one executable action from a bounded policy",
            BLUE,
        )
        self.play(FadeIn(header), run_time=0.45)

        context_box = RoundedRectangle(
            width=5.2,
            height=1.2,
            corner_radius=0.12,
            stroke_color=BLUE,
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1.0,
        ).move_to([-3.65, 1.55, 0])
        context_label = Text(
            "DECISION CONTEXT",
            font=SANS,
            font_size=14,
            color=BLUE,
            weight=BOLD,
        ).move_to([-5.85, 1.85, 0], aligned_edge=LEFT)
        context = Text(
            "Choose a repository search strategy",
            font=SANS,
            font_size=22,
            color=WHITE,
            weight=BOLD,
        ).move_to(context_box)
        self.play(FadeIn(VGroup(context_box, context_label, context), shift=RIGHT * 0.15))

        policy_label = Text(
            "TaskPolicy v0  |  softmax",
            font=MONO,
            font_size=17,
            color=MUTED,
        ).move_to([1.15, 2.1, 0], aligned_edge=LEFT)
        rows = self._probability_rows(
            self.before_probabilities,
            left=0.2,
            top=1.45,
            width=3.5,
        )
        self.play(FadeIn(policy_label), FadeIn(rows, shift=LEFT * 0.15), run_time=0.7)

        self.selected_action = self.sim.choose()
        selector = RoundedRectangle(
            width=5.05,
            height=0.78,
            corner_radius=0.12,
            stroke_color=YELLOW,
            stroke_width=2.5,
        ).move_to([2.23, 1.45, 0])
        self.play(Create(selector), run_time=0.25)
        for y in (0.53, -0.39, 1.45, 0.53):
            self.play(selector.animate.move_to([2.23, y, 0]), run_time=0.2)

        selected = Text(
            f"selected_action: {ACTION_IDS[self.selected_action]}",
            font=MONO,
            font_size=22,
            color=YELLOW,
            weight=BOLD,
        ).move_to([0, -2.55, 0])
        note = Text(
            "The agent must execute the returned action so the outcome stays attributable.",
            font=SANS,
            font_size=18,
            color=MUTED,
        ).next_to(selected, DOWN, buff=0.2)
        self.play(FadeIn(selected, shift=UP * 0.15), FadeIn(note), run_time=0.55)
        self.wait(1.0)
        self._clear()

    def _execute(self) -> None:
        header = self._stage_header(
            "02",
            "Execute",
            "run the selected strategy and preserve attributable evidence",
            GREEN,
        )
        self.play(FadeIn(header), run_time=0.45)

        action_box = RoundedRectangle(
            width=4.2,
            height=1.6,
            corner_radius=0.14,
            stroke_color=GREEN,
            stroke_width=2.5,
            fill_color=GREEN,
            fill_opacity=0.08,
        ).move_to([-4.25, 0.85, 0])
        action_eyebrow = Text(
            "SELECTED ACTION",
            font=SANS,
            font_size=14,
            color=GREEN,
            weight=BOLD,
        ).move_to([-5.9, 1.25, 0], aligned_edge=LEFT)
        action_name = Text(
            ACTIONS[self.selected_action],
            font=SANS,
            font_size=28,
            color=WHITE,
            weight=BOLD,
        ).move_to(action_box)
        self.play(FadeIn(VGroup(action_box, action_eyebrow, action_name)), run_time=0.55)

        terminal = RoundedRectangle(
            width=6.0,
            height=2.55,
            corner_radius=0.12,
            stroke_color="#42536e",
            stroke_width=1.8,
            fill_color=PANEL_2,
            fill_opacity=1.0,
        ).move_to([3.15, 0.45, 0])
        terminal_title = Text(
            "repository agent",
            font=MONO,
            font_size=15,
            color=MUTED,
        ).move_to([0.55, 1.42, 0], aligned_edge=LEFT)
        arrow = Arrow(
            action_box.get_right(),
            terminal.get_left(),
            buff=0.22,
            color=GREEN,
            stroke_width=3.5,
        )
        self.play(GrowArrow(arrow), FadeIn(terminal), FadeIn(terminal_title), run_time=0.55)

        lines = VGroup(
            Text("> semantic_search(query)", font=MONO, font_size=18, color=BLUE),
            Text("  matched refresh-token controller", font=MONO, font_size=17, color=WHITE_SOFT),
            Text("  verified definition + call site", font=MONO, font_size=17, color=WHITE_SOFT),
            Text("  completed in 182 ms", font=MONO, font_size=17, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        lines.move_to([0.55, 0.35, 0], aligned_edge=LEFT)
        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.1), run_time=0.3)

        evidence = VGroup(
            self._chip("intent", BLUE),
            self._chip("action + policy v0", YELLOW),
            self._chip("result + completion", GREEN),
        ).arrange(RIGHT, buff=0.3).scale(0.82).move_to([0, -2.1, 0])
        evidence_label = Text(
            "one durable decision episode",
            font=SANS,
            font_size=18,
            color=MUTED,
        ).next_to(evidence, DOWN, buff=0.2)
        self.play(FadeIn(evidence, shift=UP * 0.15), FadeIn(evidence_label), run_time=0.6)
        self.wait(1.0)
        self._clear()

    def _score(self) -> None:
        header = self._stage_header(
            "03",
            "Score",
            "turn the user-visible outcome into measurable feedback",
            YELLOW,
        )
        self.play(FadeIn(header), run_time=0.45)

        self.scores = self.sim.score(self.selected_action)
        self.reward = self.sim.reward(self.scores)
        centers = [-4.3, 0.0, 4.3]
        cards = VGroup()
        fills = []

        for index, (name, score, center) in enumerate(
            zip(SCORE_NAMES, self.scores, centers)
        ):
            card = RoundedRectangle(
                width=3.55,
                height=2.25,
                corner_radius=0.14,
                stroke_color=(BLUE, GREEN, YELLOW)[index],
                stroke_width=2.2,
                fill_color=PANEL,
                fill_opacity=1.0,
            ).move_to([center, 0.55, 0])
            label = Text(
                name,
                font=SANS,
                font_size=24,
                color=(BLUE, GREEN, YELLOW)[index],
                weight=BOLD,
            ).move_to([center, 1.2, 0])
            track = RoundedRectangle(
                width=2.75,
                height=0.32,
                corner_radius=0.07,
                stroke_width=0,
                fill_color=TRACK,
                fill_opacity=1.0,
            ).move_to([center, 0.55, 0])
            fill = RoundedRectangle(
                width=2.75 * score,
                height=0.32,
                corner_radius=0.07,
                stroke_width=0,
                fill_color=(BLUE, GREEN, YELLOW)[index],
                fill_opacity=1.0,
            ).move_to([center - 1.375, 0.55, 0], aligned_edge=LEFT)
            value = Text(
                f"{score:.2f}",
                font=MONO,
                font_size=24,
                color=WHITE,
                weight=BOLD,
            ).move_to([center, -0.05, 0])
            weight = Text(
                f"weight {SCORE_WEIGHTS[index]:.2f}",
                font=MONO,
                font_size=15,
                color=MUTED,
            ).move_to([center, -0.42, 0])
            cards.add(VGroup(card, label, track, value, weight))
            fills.append(fill)

        self.play(FadeIn(cards, shift=UP * 0.15), run_time=0.65)
        self.play(*[GrowFromEdge(fill, LEFT) for fill in fills], run_time=0.85)

        reward_box = RoundedRectangle(
            width=5.2,
            height=1.0,
            corner_radius=0.14,
            stroke_color=YELLOW,
            stroke_width=2.5,
            fill_color=YELLOW,
            fill_opacity=0.1,
        ).move_to([0, -1.55, 0])
        reward_text = Text(
            f"shaped reward  R = {self.reward:+.2f}",
            font=MONO,
            font_size=26,
            color=YELLOW,
            weight=BOLD,
        ).move_to(reward_box)
        note = Text(
            "Local scoring by default. Azure evaluators remain optional.",
            font=SANS,
            font_size=18,
            color=MUTED,
        ).next_to(reward_box, DOWN, buff=0.28)
        self.play(FadeIn(VGroup(reward_box, reward_text), shift=UP * 0.15), FadeIn(note), run_time=0.65)
        self.wait(1.1)
        self._clear()

    def _improve(self) -> None:
        header = self._stage_header(
            "04",
            "Improve",
            "use completed evidence to update the next decision",
            ORANGE,
        )
        self.play(FadeIn(header), run_time=0.45)

        before_title = Text(
            "BEFORE  |  policy v0",
            font=MONO,
            font_size=18,
            color=MUTED,
        ).move_to([-5.7, 2.0, 0], aligned_edge=LEFT)
        before_rows = self._probability_rows(
            self.before_probabilities,
            left=-5.7,
            top=1.25,
            width=3.0,
        )
        self.play(FadeIn(before_title), FadeIn(before_rows), run_time=0.65)

        episodes: List[Tuple[int, float]] = []
        for action in range(len(ACTIONS)):
            for _ in range(10):
                scores = self.sim.score(action)
                episodes.append((action, self.sim.reward(scores)))
        self.sim.update(episodes)
        after_probabilities = self.sim.probabilities()

        update_box = RoundedRectangle(
            width=3.15,
            height=2.0,
            corner_radius=0.14,
            stroke_color=ORANGE,
            stroke_width=2.4,
            fill_color=ORANGE,
            fill_opacity=0.08,
        ).move_to([0, 0.15, 0])
        update_title = Text(
            "CPU UPDATE",
            font=SANS,
            font_size=16,
            color=ORANGE,
            weight=BOLD,
        ).move_to([0, 0.72, 0])
        update_rule = Text(
            "reward - baseline",
            font=MONO,
            font_size=17,
            color=WHITE,
        ).move_to([0, 0.25, 0])
        update_detail = Text(
            "REINFORCE + entropy",
            font=MONO,
            font_size=15,
            color=MUTED,
        ).move_to([0, -0.17, 0])
        batch = Text(
            "30 scored episodes",
            font=SANS,
            font_size=16,
            color=YELLOW,
        ).move_to([0, -0.57, 0])
        arrows = VGroup(
            Arrow([-2.15, 0.15, 0], [-1.65, 0.15, 0], buff=0, color=ORANGE),
            Arrow([1.65, 0.15, 0], [2.15, 0.15, 0], buff=0, color=ORANGE),
        )
        self.play(
            GrowArrow(arrows[0]),
            FadeIn(VGroup(update_box, update_title, update_rule, update_detail, batch)),
            run_time=0.65,
        )

        after_title = Text(
            "AFTER  |  active policy",
            font=MONO,
            font_size=18,
            color=ORANGE,
        ).move_to([2.2, 2.0, 0], aligned_edge=LEFT)
        best_action = max(range(len(ACTIONS)), key=lambda index: after_probabilities[index])
        after_rows = self._probability_rows(
            after_probabilities,
            left=2.2,
            top=1.25,
            width=3.0,
            selected=best_action,
        )
        self.play(GrowArrow(arrows[1]), FadeIn(after_title), FadeIn(after_rows), run_time=0.8)

        change = Text(
            f"{ACTIONS[best_action]}: {self.before_probabilities[best_action] * 100:.1f}%  ->  "
            f"{after_probabilities[best_action] * 100:.1f}%",
            font=MONO,
            font_size=23,
            color=ORANGE,
            weight=BOLD,
        ).move_to([0, -2.15, 0])
        next_step = Text(
            "The next task-policy-decide call uses this snapshot and prior feedback.",
            font=SANS,
            font_size=18,
            color=MUTED,
        ).next_to(change, DOWN, buff=0.22)
        self.play(FadeIn(change, shift=UP * 0.15), FadeIn(next_step), run_time=0.6)
        self.wait(1.2)
        self._clear()

    def _closing(self) -> None:
        def stage_card(number: str, name: str, detail: str, color: str) -> VGroup:
            box = RoundedRectangle(
                width=2.6,
                height=1.45,
                corner_radius=0.14,
                stroke_color=color,
                stroke_width=2.2,
                fill_color=color,
                fill_opacity=0.08,
            )
            number_text = Text(number, font=MONO, font_size=14, color=color, weight=BOLD)
            name_text = Text(name, font=SANS, font_size=24, color=WHITE, weight=BOLD)
            detail_text = Text(detail, font=SANS, font_size=14, color=MUTED)
            content = VGroup(number_text, name_text, detail_text).arrange(DOWN, buff=0.12)
            content.move_to(box)
            return VGroup(box, content)

        title = Text(
            "Make the next decision better",
            font=SANS,
            font_size=38,
            color=WHITE,
            weight=BOLD,
        ).move_to(UP * 2.65)
        cards = VGroup(
            stage_card("01", "Choose", "bounded actions", BLUE),
            stage_card("02", "Execute", "attributable result", GREEN),
            stage_card("03", "Score", "measured outcome", YELLOW),
            stage_card("04", "Improve", "new snapshot", ORANGE),
        ).arrange(RIGHT, buff=0.55).move_to(UP * 0.55)

        self.play(FadeIn(title), run_time=0.45)
        for card in cards:
            self.play(FadeIn(card, shift=RIGHT * 0.1), run_time=0.3)

        arrows = VGroup(
            *[
                Arrow(
                    cards[index].get_right(),
                    cards[index + 1].get_left(),
                    buff=0.08,
                    color=MUTED,
                    stroke_width=3,
                )
                for index in range(3)
            ]
        )
        self.play(*[GrowArrow(arrow) for arrow in arrows], run_time=0.6)

        feedback = CurvedArrow(
            cards[3].get_bottom() + DOWN * 0.08,
            cards[0].get_bottom() + DOWN * 0.08,
            angle=-TAU / 7,
            color=GREEN,
            stroke_width=3.5,
        )
        feedback_label = Text(
            "evidence improves the next choice",
            font=SANS,
            font_size=17,
            color=GREEN,
        ).next_to(feedback, DOWN, buff=0.08)
        self.play(Create(feedback), FadeIn(feedback_label), run_time=0.7)

        tagline = Text(
            "No model fine-tuning. No GPU jobs. Small, inspectable policy updates.",
            font=SANS,
            font_size=23,
            color=WHITE_SOFT,
            weight=BOLD,
        ).move_to(DOWN * 2.65)
        wordmark = Text("agent-learn", font=MONO, font_size=18, color=BLUE)
        wordmark.next_to(tagline, DOWN, buff=0.25)
        self.play(FadeIn(tagline, shift=UP * 0.12), FadeIn(wordmark), run_time=0.7)
        self.wait(2.0)