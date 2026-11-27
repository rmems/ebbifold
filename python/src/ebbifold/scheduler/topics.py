"""Topic normalization helpers (src/lib/topics.ts, curriculum pacing)."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping

_SMALL_WORDS = {
    "and", "or", "the", "a", "an", "of", "to", "for", "in", "on", "with", "without", "vs", "versus",
}
_ACRONYM_RE = re.compile(r"^[a-z0-9]{2,5}$")
_CURRICULUM_RE = re.compile(r"[^a-z0-9]+")

# JavaScript String.prototype.trim / \s whitespace set.
_JS_WS = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def js_trim(value: str) -> str:
    return value.strip(_JS_WS)


def _collapse_ws(value: str) -> str:
    out: list[str] = []
    in_ws = False
    for ch in value:
        if ch in _JS_WS:
            if not in_ws:
                out.append(" ")
            in_ws = True
        else:
            out.append(ch)
            in_ws = False
    return "".join(out)


def normalize_topic(topic: str) -> str:
    lowered = re.sub(r"[_-]+", " ", topic.lower())
    kept = "".join(
        ch for ch in lowered
        if ch in _JS_WS or unicodedata.category(ch)[0] in ("L", "N")
    )
    return js_trim(_collapse_ws(kept))


def to_title_case_topic(topic: str) -> str:
    normalized = normalize_topic(topic)
    if not normalized:
        return ""
    words = [w for w in normalized.split(" ") if w]
    return " ".join(
        w.upper() if len(w) <= 3 and w not in _SMALL_WORDS else w[0].upper() + w[1:]
        for w in words
    )


def expand_topic_variants(topic: str) -> list[str]:
    trimmed = js_trim(topic)
    normalized = normalize_topic(trimmed)
    title = to_title_case_topic(normalized) if normalized else ""
    smart_caps = (
        " ".join(
            w.upper() if len(w) <= 3 and w not in ("and", "or", "the") else w
            for w in normalized.split(" ")
            if w
        )
        if normalized
        else ""
    )
    variants: dict[str, None] = {}
    for value in (trimmed, normalized, title, smart_caps):
        if value:
            variants[value] = None
    if _ACRONYM_RE.fullmatch(normalized) and not normalized.endswith("\n"):
        variants[normalized.upper()] = None
    return list(variants)


def expand_topic_set(topics: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for topic in topics:
        for raw in expand_topic_variants(topic):
            t = js_trim(raw)
            if t and t not in seen:
                seen[t] = None
    return list(seen)


def normalize_curriculum_topic(value: str) -> str:
    lowered = js_trim(value).lower().replace("&", " and ")
    return _CURRICULUM_RE.sub("-", lowered).removeprefix("-").removesuffix("-")


# curriculum-pacing.ts / teaching-pace.ts
BLOCK_WEEKS = 7
PACING_CURRENT_WEEK_BOOST = 4
PACING_DATED_RECENCY_BOOST = PACING_CURRENT_WEEK_BOOST
CADENCE_CURRENT_WEEK_BOOST = 0.2
CADENCE_LAST_WEEK_BOOST = 0.1
TAUGHT_TOPIC_WINDOW_DAYS = 14


def teaching_week_for_topics(
    topics: Iterable[str] | None, topic_weeks: Mapping[str, int] | None
) -> int | None:
    if not topics or not topic_weeks:
        return None
    earliest: int | None = None
    for topic in topics:
        week = topic_weeks.get(normalize_curriculum_topic(topic))
        if week is None:
            continue
        if earliest is None or week < earliest:
            earliest = week
    return earliest


def teaching_cadence_concept_boost(concept_week: int | None, current_week: int | None) -> float:
    if concept_week is None or current_week is None:
        return 0
    if current_week > BLOCK_WEEKS:
        return 0
    if concept_week == current_week:
        return CADENCE_CURRENT_WEEK_BOOST
    if concept_week == current_week - 1:
        return CADENCE_LAST_WEEK_BOOST
    return 0


def dated_topic_recency_boost(
    topics: Iterable[str], topic_recency_days: Mapping[str, float] | None
) -> float:
    if not topic_recency_days:
        return 0
    strongest = 0.0
    for topic in topics:
        normalized = normalize_curriculum_topic(topic)
        if not normalized:
            continue
        age = topic_recency_days.get(normalized)
        if age is None:
            continue
        if not math.isfinite(age) or age < 0 or age >= TAUGHT_TOPIC_WINDOW_DAYS:
            continue
        boost = -PACING_DATED_RECENCY_BOOST * (TAUGHT_TOPIC_WINDOW_DAYS - age) / TAUGHT_TOPIC_WINDOW_DAYS
        strongest = min(strongest, boost)
    return strongest


def dated_topic_recency_concept_boost(
    topics: Iterable[str], topic_recency_days: Mapping[str, float] | None
) -> float:
    item = dated_topic_recency_boost(topics, topic_recency_days)
    return (-item / PACING_DATED_RECENCY_BOOST) * CADENCE_CURRENT_WEEK_BOOST if item < 0 else 0
