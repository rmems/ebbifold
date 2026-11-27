"""Pure clinical concept-thread follow-up matching.

Port of concept-thread-policy.ts: a mature successful anchor may promote a
same-condition, different-facet candidate inside its existing difficulty and
freshness stratum. No storage access happens here; anchors and candidates are
plain snapshot rows.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

CONCEPT_THREAD_POLICY_VERSION = "clinical-thread-facets-v1"
CONCEPT_THREAD_MIN_AGE_MS = 15 * 60 * 1000
CONCEPT_THREAD_NATURAL_BREAK_MS = 12 * 60 * 60 * 1000
CONCEPT_THREAD_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000
CONCEPT_THREAD_MIN_INTERVENING_EXPOSURES = 5

GENERIC_TOPICS = {
    "acute", "adolescent", "adult", "anatomy", "basic science", "boy", "boys", "cah",
    "cardiology", "child", "children", "classification", "clinical", "clinical features",
    "critical care", "diagnosis", "differential", "differential diagnosis", "emergency",
    "epidemiology", "girl", "girls", "health", "infant", "infants", "infection", "investigation",
    "management", "mechanism", "medicine", "microbiology", "musculoskeletal", "neonatal", "neonate",
    "newborn", "orthopaedics", "paam", "paediatric", "paediatrics", "pathology", "pathophysiology",
    "presentation", "pwh", "recall", "rheumatology", "surgical", "toddler", "toddlers", "treatment",
    "usmle", "usmle step1",
}

GENERIC_TEXT_TOKENS = {
    "a", "about", "acute", "adult", "after", "an", "and", "are", "best", "by", "case", "cause",
    "caused", "causes", "causative", "child", "children", "chronic", "clinical", "common", "complication",
    "condition", "definitive", "diagnosis", "diagnosed", "does", "disease", "disorder", "emergency", "female",
    "features", "following", "for", "how", "in", "investigation", "is", "like", "look", "management", "male",
    "man", "medical", "mechanism", "most", "next", "of", "old", "organism", "patient", "presentation",
    "presents", "sign", "signs", "symptom", "symptoms", "syndrome", "the", "therapy", "to", "treat",
    "treated", "treatment", "what", "which", "with", "woman", "year",
}

FACET_TARGETS = {
    "cause_mechanism": ["presentation", "diagnosis_investigation", "management"],
    "presentation": ["diagnosis_investigation", "management", "complications_prognosis"],
    "diagnosis_investigation": ["management", "complications_prognosis", "cause_mechanism"],
    "management": ["complications_prognosis", "diagnosis_investigation", "cause_mechanism"],
    "complications_prognosis": ["management", "diagnosis_investigation", "cause_mechanism"],
}


def normalize_clinical_thread_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", stripped)).strip()


def _normalized_topic(topic: str) -> str | None:
    normalized = normalize_clinical_thread_text(topic)
    if len(normalized) < 4 or normalized in GENERIC_TOPICS:
        return None
    return normalized


def _concepts_compatible(anchor: set[str], candidate: set[str]) -> bool:
    if not anchor or not candidate:
        return True
    return bool(anchor & candidate)


def _infer_facet(question_type: str, text: str) -> str:
    if question_type in {"management", "next step", "treatment", "therapy"}:
        return "management"
    if question_type in {"complication", "complications", "prognosis", "adverse effect", "outcome"}:
        return "complications_prognosis"
    if question_type in {"clinical features", "presentation"}:
        return "presentation"
    if question_type in {"diagnosis", "investigation", "interpretation", "image interpretation", "clinical reasoning", "discrimination"}:
        return "diagnosis_investigation"
    if question_type in {"mechanism", "basic science", "pathophysiology", "risk factors", "epidemiology", "physiology", "aetiology", "etiology"}:
        return "cause_mechanism"
    if re.search(r"\b(causative|caused by|cause of|causes of|organism|aetiolog|etiolog|mechanism|pathophysiolog|risk factor)\b", text):
        return "cause_mechanism"
    if re.search(r"\b(clinical features|presents with|presentation|signs and symptoms|what does .* look like)\b", text):
        return "presentation"
    if re.search(r"\b(diagnos\w*|investigat\w*|differentiat\w*|criteria|test confirms)\b", text):
        return "diagnosis_investigation"
    if re.search(r"\b(complication|complications|prognosis|outcome|sequela|sequelae|adverse effect)\b", text):
        return "complications_prognosis"
    if re.search(r"\b(manage|management|treat|treatment|therapy|first line|next best step|definitive management)\b", text):
        return "management"
    return "unknown"


def infer_clinical_facet(item: dict[str, Any]) -> str:
    normalized_text = normalize_clinical_thread_text(item.get("text") or "")
    normalized_format = normalize_clinical_thread_text(item.get("format") or "")
    return _infer_facet(
        normalize_clinical_thread_text(item.get("questionType") or ""),
        " ".join(part for part in (normalized_text, normalized_format) if part),
    )


def concept_thread_anchor_is_mature(anchor: dict[str, Any], now_ms: float) -> bool:
    age = now_ms - anchor["answeredAtMs"]
    if not math.isfinite(age) or age < CONCEPT_THREAD_MIN_AGE_MS or age > CONCEPT_THREAD_MAX_AGE_MS:
        return False
    return age >= CONCEPT_THREAD_NATURAL_BREAK_MS or anchor["interveningExposures"] >= CONCEPT_THREAD_MIN_INTERVENING_EXPOSURES


def _text_contains_topic(text: str, topic: str) -> bool:
    return f" {topic} " in f" {text} "


def _text_phrases(normalized_text: str) -> set[str]:
    tokens = [t for t in normalized_text.split(" ") if len(t) >= 3 and t not in GENERIC_TEXT_TOKENS]
    phrases: set[str] = set()
    for size in range(2, min(4, len(tokens)) + 1):
        for index in range(0, len(tokens) - size + 1):
            phrases.add(" ".join(tokens[index:index + size]))
    return phrases


def _prepare_item(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("text") or ""
    normalized_text = normalize_clinical_thread_text(text)
    normalized_format = normalize_clinical_thread_text(item.get("format") or "")
    combined = " ".join(part for part in (normalized_text, normalized_format) if part)
    return {
        "id": item["id"],
        "variantGroupId": item.get("variantGroupId"),
        "facet": _infer_facet(normalize_clinical_thread_text(item.get("questionType") or ""), combined),
        "conceptIds": {str(x).strip() for x in (item.get("conceptIds") or []) if str(x).strip()},
        "specificTopics": {t for t in (_normalized_topic(x) for x in (item.get("topics") or [])) if t},
        "normalizedText": normalized_text,
        "textPhrases": _text_phrases(normalized_text),
    }


def _shared_topic(anchor: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    if not _concepts_compatible(anchor["conceptIds"], candidate["conceptIds"]):
        return None
    matches: set[str] = set()
    for topic in candidate["specificTopics"]:
        if topic in anchor["specificTopics"] or (" " in topic and _text_contains_topic(anchor["normalizedText"], topic)):
            matches.add(topic)
    for topic in anchor["specificTopics"]:
        if topic in candidate["specificTopics"] or (" " in topic and _text_contains_topic(candidate["normalizedText"], topic)):
            matches.add(topic)
    if not matches:
        matches = candidate["textPhrases"] & anchor["textPhrases"]
    if not matches:
        return None
    return sorted(matches, key=lambda x: (-len(x.split(" ")), -len(x), x))[0]


def _prepare_anchor(anchor: dict[str, Any], now_ms: float, input_rank: int) -> dict[str, Any] | None:
    if not concept_thread_anchor_is_mature(anchor, now_ms):
        return None
    prepared = _prepare_item(anchor)
    if prepared["facet"] == "unknown":
        return None
    return {
        **prepared,
        "anchorEventId": anchor["anchorEventId"],
        "ageMs": now_ms - anchor["answeredAtMs"],
        "interveningExposures": anchor["interveningExposures"],
        "inputRank": input_rank,
    }


def _anchor_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    anchor = entry["anchor"]
    return (
        entry["targetPreference"], anchor["ageMs"], -anchor["interveningExposures"],
        anchor["anchorEventId"], anchor["inputRank"],
    )


class PreparedConceptThreadMatcher:
    def __init__(self, anchors: list[dict[str, Any]], now_ms: float):
        prepared = [p for p in (_prepare_anchor(a, now_ms, i) for i, a in enumerate(anchors)) if p]
        self.anchor_count = len(prepared)
        self._by_target: dict[str, list[dict[str, Any]]] = {}
        for anchor in prepared:
            for preference, target in enumerate(FACET_TARGETS[anchor["facet"]]):
                self._by_target.setdefault(target, []).append({"anchor": anchor, "targetPreference": preference})
        for entries in self._by_target.values():
            entries.sort(key=_anchor_key)

    def find_match(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        prepared = _prepare_item(candidate)
        if prepared["facet"] == "unknown":
            return None
        for entry in self._by_target.get(prepared["facet"], []):
            anchor = entry["anchor"]
            if prepared["id"] == anchor["id"] or prepared["facet"] == anchor["facet"]:
                continue
            if prepared["variantGroupId"] and anchor["variantGroupId"] and prepared["variantGroupId"] == anchor["variantGroupId"]:
                continue
            shared = _shared_topic(anchor, prepared)
            if not shared:
                continue
            return {
                "policyVersion": CONCEPT_THREAD_POLICY_VERSION,
                "anchorEventId": anchor["anchorEventId"],
                "anchorItemId": anchor["id"],
                "anchorFacet": anchor["facet"],
                "targetFacet": prepared["facet"],
                "sharedTopic": shared,
                "ageMs": anchor["ageMs"],
                "interveningExposures": anchor["interveningExposures"],
                "targetPreference": entry["targetPreference"],
            }
        return None


def prepare_concept_thread_matcher(anchors: list[dict[str, Any]], now_ms: float) -> PreparedConceptThreadMatcher:
    return PreparedConceptThreadMatcher(anchors, now_ms)


def find_concept_thread_match(candidate: dict[str, Any], anchors: list[dict[str, Any]], now_ms: float) -> dict[str, Any] | None:
    return prepare_concept_thread_matcher(anchors, now_ms).find_match(candidate)
