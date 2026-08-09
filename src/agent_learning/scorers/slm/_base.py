"""Phi-4-mini-instruct ONNX runner + shared scorer base for Tier 3 SLM.

The runner lazily loads ``onnxruntime-genai`` and the Phi-4-mini-instruct
INT4 model from a local directory the first time :meth:`SlmRunner.generate`
is called. Subsequent calls reuse the cached model and tokenizer instances.
The runner abstraction also makes the per-scorer prompt building and JSON
parsing trivially mockable in unit tests via :meth:`set_runner`.

Prompt format
~~~~~~~~~~~~~

The runner applies the Phi-3.5 / Phi-4 chat template:

.. code-block:: text

    <|system|>
    {system}<|end|>
    <|user|>
    {user}<|end|>
    <|assistant|>

Each subclass of :class:`SlmScorerBase` supplies its own ``system`` and
``user`` strings; the assistant reply is parsed by
:func:`parse_verdict` which expects a single JSON object of the form
``{"verdict": "pass"|"fail", "confidence": <number in [0,1]>}``. The
parser is permissive about leading or trailing prose because instruction-
tuned small models occasionally emit a sentence in front of the JSON.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..base import ScoreResult


_GENAI_IMPORT_ERROR = (
    "Tier 3 SLM scorers require the [slm] extra. Install with: "
    "pip install agent-learning[slm]"
)


def _require_genai() -> Any:
    """Lazy-import ``onnxruntime_genai``. Raises a friendly error if missing."""
    try:
        import onnxruntime_genai as og  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via tests/mocks
        raise ImportError(_GENAI_IMPORT_ERROR) from exc
    return og


# ----- model runner --------------------------------------------------------


class SlmRunner:
    """Thin wrapper around onnxruntime-genai's ``Model``+``Tokenizer``+``Generator``.

    The wrapper hides the version-dependent generator loop behind a single
    :meth:`generate` method. A test harness can replace the runner with any
    object that implements ``generate(prompt: str, *, max_new_tokens: int,
    temperature: float) -> str``.
    """

    _shared_lock = threading.Lock()
    _shared_runners: Dict[str, "SlmRunner"] = {}

    def __init__(self, model_dir: str) -> None:
        if not model_dir:
            raise ValueError("SlmRunner requires a non-empty model_dir.")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"SLM model directory not found: {model_dir!r}. "
                "Download Phi-4-mini-instruct ONNX (INT4) to this path or "
                "set AGENT_LEARNING_SLM_MODEL_DIR to its location."
            )
        og = _require_genai()
        self._og = og
        self._model = og.Model(model_dir)
        self._tokenizer = og.Tokenizer(self._model)
        self.model_dir = model_dir

    @classmethod
    def get_shared(cls, model_dir: str) -> "SlmRunner":
        """Return a process-wide cached runner for ``model_dir``.

        Loading the 3.8 B Phi-4-mini-instruct ONNX model from cold can take
        several seconds and consume ~2.5 GB of RAM. Sharing one runner across
        the three Tier 3 scorers keeps inference hot and avoids the
        re-loading cost.
        """
        with cls._shared_lock:
            runner = cls._shared_runners.get(model_dir)
            if runner is None:
                runner = cls(model_dir)
                cls._shared_runners[model_dir] = runner
            return runner

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        """Run greedy decoding on ``prompt`` and return the assistant text."""
        og = self._og
        tokenizer = self._tokenizer
        input_tokens = tokenizer.encode(prompt)
        input_length = len(input_tokens)
        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=input_length + int(max_new_tokens),
            do_sample=False,
            temperature=float(max(temperature, 0.0)),
        )
        # API surface drift across onnxruntime-genai versions: 0.4 wants
        # params.input_ids, 0.5+ uses generator.append_tokens. Try both.
        try:
            params.input_ids = input_tokens
        except (AttributeError, TypeError):  # pragma: no cover - version drift
            pass
        generator = og.Generator(self._model, params)
        append = getattr(generator, "append_tokens", None)
        if callable(append):  # pragma: no cover - version drift
            try:
                append(input_tokens)
            except Exception:
                pass
        # Greedy decoding loop.
        while not generator.is_done():
            compute_logits = getattr(generator, "compute_logits", None)
            if callable(compute_logits):  # pragma: no cover - version drift
                compute_logits()
            generator.generate_next_token()
        get_sequence = getattr(generator, "get_sequence", None)
        if callable(get_sequence):
            full = get_sequence(0)
        else:  # pragma: no cover - very old version
            full = generator.get_next_tokens()
        # Strip the prompt prefix.
        output_tokens = list(full)[input_length:]
        text = tokenizer.decode(output_tokens)
        return text


# ----- prompt rendering ----------------------------------------------------


_CHAT_TEMPLATE = (
    "<|system|>\n{system}<|end|>\n"
    "<|user|>\n{user}<|end|>\n"
    "<|assistant|>\n"
)


def render_chat_prompt(*, system: str, user: str) -> str:
    """Render a Phi-3.5 / Phi-4 chat-style prompt."""
    return _CHAT_TEMPLATE.format(system=system.strip(), user=user.strip())


# ----- verdict parsing -----------------------------------------------------


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_verdict(text: str) -> Tuple[float, Dict[str, Any]]:
    """Extract ``(probability_pass, raw)`` from the model's reply.

    Looks for the first JSON object in ``text``. Falls back to a token
    heuristic if no JSON is recoverable. The function never raises on a
    malformed reply; an unrecoverable parse degrades to a neutral 0.5
    probability so the caller can still emit a score.
    """
    raw: Dict[str, Any] = {"text": text}
    if not text:
        return 0.5, raw
    # First, try to parse as raw JSON.
    candidate: Optional[Dict[str, Any]] = None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            candidate = loaded
    except (ValueError, TypeError):
        match = _JSON_RE.search(text)
        if match is not None:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    candidate = loaded
            except (ValueError, TypeError):
                candidate = None
    if candidate is not None:
        verdict = str(candidate.get("verdict", "")).strip().lower()
        try:
            confidence = float(candidate.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        raw["verdict"] = verdict
        raw["confidence"] = confidence
        if verdict.startswith("pass"):
            return confidence, raw
        if verdict.startswith("fail"):
            return 1.0 - confidence, raw
    # Token-level fallback. Look for an isolated "pass" or "fail" word.
    lowered = text.lower()
    pass_hit = re.search(r"\bpass\b", lowered)
    fail_hit = re.search(r"\bfail\b", lowered)
    if pass_hit and (not fail_hit or pass_hit.start() < fail_hit.start()):
        raw["heuristic"] = "pass"
        return 0.75, raw
    if fail_hit:
        raw["heuristic"] = "fail"
        return 0.25, raw
    raw["heuristic"] = "neutral"
    return 0.5, raw


# ----- shared base ---------------------------------------------------------


@dataclass
class SlmScorerBase:
    """Shared scoring scaffolding for Tier 3 small-language-model scorers."""

    name: str = "slm"
    pass_threshold: float = 0.5
    max_new_tokens: int = 64
    temperature: float = 0.0
    model_dir: str = ""
    runner: Optional[SlmRunner] = None
    system_prompt: str = field(default="", repr=False)

    def _get_runner(self) -> SlmRunner:
        if self.runner is not None:
            return self.runner
        if not self.model_dir:
            raise ValueError(
                f"{self.name}: model_dir must be set to load the SLM"
            )
        self.runner = SlmRunner.get_shared(self.model_dir)
        return self.runner

    def _build_user_prompt(self, **inputs: Any) -> str:  # pragma: no cover - override
        raise NotImplementedError

    def _validate(self, **inputs: Any) -> None:  # pragma: no cover - override
        return

    def _score_from_inputs(self, **inputs: Any) -> ScoreResult:
        self._validate(**inputs)
        prompt = render_chat_prompt(
            system=self.system_prompt,
            user=self._build_user_prompt(**inputs),
        )
        runner = self._get_runner()
        text = runner.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        probability, raw = parse_verdict(text)
        if probability >= self.pass_threshold:
            label = "pass"
            confidence = probability
        else:
            label = "fail"
            confidence = 1.0 - probability
        return ScoreResult(
            label=label,
            confidence=confidence,
            normalized=probability,
            features={
                "probability": probability,
                "model_dir": self.model_dir,
                "raw": raw,
            },
        )


__all__ = [
    "SlmScorerBase",
    "SlmRunner",
    "parse_verdict",
    "render_chat_prompt",
]
