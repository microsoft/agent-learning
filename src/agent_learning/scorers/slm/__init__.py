"""Tier 3 small-language-model scorers.

The package wraps a locally-hosted instance of
`Microsoft Phi-4-mini-instruct
<https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx>`_ (3.8 B
parameters, 4-bit ONNX) and exposes three pass/fail evaluators with the
same surface as the Tier 1 and Tier 2 scorers. The model is loaded once
per process via :class:`SlmRunner` and reused across calls.

Install:

.. code-block:: bash

    pip install agents-learning-sdk[slm]
"""

from .adherence import SlmAdherenceScorer
from .completion import SlmCompletionScorer
from .intent import SlmIntentScorer

__all__ = [
    "SlmAdherenceScorer",
    "SlmCompletionScorer",
    "SlmIntentScorer",
]
