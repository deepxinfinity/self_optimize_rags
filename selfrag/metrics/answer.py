from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1, the official QASPER answer metric."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)
