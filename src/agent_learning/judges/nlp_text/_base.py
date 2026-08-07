"""Shared TF-IDF + scikit-learn logistic-regression pipeline for the
Tier 2 NLP text judges.

The class wraps three concrete sub-classes (intent / adherence /
completion). It defers the scikit-learn import to first use so the
package stays importable without the ``[nlp]`` extra installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..base import JudgeScore


_SKLEARN_IMPORT_ERROR = (
    "Tier 2 NLP text judges require the [nlp] extra. Install with: "
    "pip install agents-learning-sdk[nlp]"
)


def _require_sklearn() -> Tuple[Any, Any]:
    """Lazy-import scikit-learn. Raises a friendly error if missing."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - exercised via tests/mocks
        raise ImportError(_SKLEARN_IMPORT_ERROR) from exc
    return TfidfVectorizer, LogisticRegression


def _require_joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_SKLEARN_IMPORT_ERROR) from exc
    return joblib


def _join_text(*parts: Optional[str]) -> str:
    """Join non-empty text parts with ``" [SEP] "``."""
    cleaned = [p.strip() for p in parts if p]
    return " [SEP] ".join(cleaned)


@dataclass
class _NlpTextJudgeBase:
    """Base class for the three Tier 2 NLP text judges.

    Sub-classes override :attr:`name` and :meth:`_pair_text` to control
    which inputs feed the vectorizer (intent uses query+response;
    adherence and completion use response only).
    """

    name: str = "nlp_text"
    pass_threshold: float = 0.5
    max_features: int = 20000
    ngram_min: int = 1
    ngram_max: int = 2
    C: float = 1.0
    min_df: int = 1
    vectorizer: Any = None  # sklearn TfidfVectorizer or None
    classifier: Any = None  # sklearn LogisticRegression or None

    # ----- inputs -----
    def _pair_text(self, *, query: Optional[str], response: str) -> str:
        return _join_text(query, response)

    # ----- training -----
    def fit(self, training_rows: Sequence[Dict[str, Any]]) -> "_NlpTextJudgeBase":
        """Fit TF-IDF + logistic regression on labeled rows.

        Each row must look like ``{"query": str?, "response": str,
        "label": 0|1}``. Rows missing ``response`` or ``label`` are
        skipped. When fewer than two distinct labels are seen the
        judge stays unfitted.
        """
        TfidfVectorizer, LogisticRegression = _require_sklearn()
        texts: List[str] = []
        labels: List[int] = []
        for row in training_rows:
            if not isinstance(row, dict):
                continue
            response = row.get("response")
            label = row.get("label")
            if not response or label is None:
                continue
            try:
                lbl = int(label)
            except (TypeError, ValueError):
                continue
            texts.append(self._pair_text(query=row.get("query"), response=str(response)))
            labels.append(1 if lbl > 0 else 0)
        if len(set(labels)) < 2:
            self.vectorizer = None
            self.classifier = None
            return self
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(self.ngram_min, self.ngram_max),
            min_df=self.min_df,
            lowercase=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="liblinear",
        )
        self.classifier.fit(X, labels)
        return self

    # ----- prediction -----
    @property
    def fitted(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    def _predict_probability(
        self, *, query: Optional[str], response: str
    ) -> float:
        if not self.fitted:
            return float(self.pass_threshold)
        text = self._pair_text(query=query, response=response)
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        # Find the index of the positive class (label == 1).
        classes = list(self.classifier.classes_)
        if 1 in classes:
            idx = classes.index(1)
        else:
            # Degenerate single-class fit; fall back to the last column.
            idx = len(classes) - 1
        return float(proba[idx])

    def _to_score(
        self,
        probability: float,
        *,
        features: Optional[Dict[str, Any]] = None,
    ) -> JudgeScore:
        if probability >= self.pass_threshold:
            label = "pass"
            confidence = probability
        else:
            label = "fail"
            confidence = 1.0 - probability
        feat: Dict[str, Any] = {
            "probability": probability,
            "fitted": self.fitted,
        }
        if features:
            feat.update(features)
        return JudgeScore(
            label=label,
            confidence=confidence,
            normalized=probability,
            features=feat,
        )

    # ----- snapshots -----
    def to_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable header. The fitted sklearn state
        is dumped separately to a joblib file.
        """
        return {
            "name": self.name,
            "pass_threshold": float(self.pass_threshold),
            "max_features": int(self.max_features),
            "ngram_min": int(self.ngram_min),
            "ngram_max": int(self.ngram_max),
            "C": float(self.C),
            "min_df": int(self.min_df),
            "fitted": self.fitted,
        }

    def save(self, snapshot_dir: str) -> str:
        os.makedirs(snapshot_dir, exist_ok=True)
        header_path = os.path.join(snapshot_dir, f"{self.name}.nlp_text.json")
        with open(header_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_snapshot(), fh)
        if self.fitted:
            joblib = _require_joblib()
            blob_path = os.path.join(
                snapshot_dir, f"{self.name}.nlp_text.joblib"
            )
            joblib.dump(
                {
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                },
                blob_path,
            )
        return header_path

    @classmethod
    def _load_into(
        cls,
        instance: "_NlpTextJudgeBase",
        snapshot_dir: Optional[str],
    ) -> "_NlpTextJudgeBase":
        if not snapshot_dir:
            return instance
        header_path = os.path.join(
            snapshot_dir, f"{instance.name}.nlp_text.json"
        )
        blob_path = os.path.join(
            snapshot_dir, f"{instance.name}.nlp_text.joblib"
        )
        if not os.path.isfile(header_path):
            return instance
        with open(header_path, "r", encoding="utf-8") as fh:
            header = json.load(fh)
        instance.pass_threshold = float(
            header.get("pass_threshold", instance.pass_threshold)
        )
        instance.max_features = int(
            header.get("max_features", instance.max_features)
        )
        instance.ngram_min = int(header.get("ngram_min", instance.ngram_min))
        instance.ngram_max = int(header.get("ngram_max", instance.ngram_max))
        instance.C = float(header.get("C", instance.C))
        instance.min_df = int(header.get("min_df", instance.min_df))
        if header.get("fitted") and os.path.isfile(blob_path):
            joblib = _require_joblib()
            blob = joblib.load(blob_path)
            instance.vectorizer = blob["vectorizer"]
            instance.classifier = blob["classifier"]
        return instance


__all__ = ["_NlpTextJudgeBase", "_join_text", "_require_sklearn"]
