"""Легкий локальний пошук (BM25) без векторної БД.

Для української використовується грубий «стемінг» — обрізання слова до 6 символів.
Цього достатньо, щоб «казино/казиноо», «гравці/гравців/гравцям» зводились до одного ключа.
"""
from __future__ import annotations

import math
from collections import Counter

from app.utils.text import STOPWORDS, words_lower


def stem(word: str) -> str:
    return word[:6] if len(word) > 6 else word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in words_lower(text) if w not in STOPWORDS and len(w) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            dl = len(d) or 1
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top(self, query: str, k: int = 10, min_score: float = 0.0) -> list[tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > min_score]


def token_overlap(a: str, b: str) -> float:
    """Частка токенів `a`, що є в `b` (0..1)."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)
