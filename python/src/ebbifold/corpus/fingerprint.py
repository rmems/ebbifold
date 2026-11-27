"""Deterministic corpus fingerprints ported from the TypeScript release gates.

JavaScript's JSON.stringify is the upstream canonicalisation boundary. These
helpers reproduce its compact tuple/object output closely enough for the
checked-in release receipts; they are not general-purpose serializers.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

BASELINE_MODULE = "usmle/step1/baseline/v1"


def js_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(js_json_dumps(value).encode("utf-8")).hexdigest()


def canonicalize(value: Any) -> Any:
    if isinstance(value, list):
        return [canonicalize(child) for child in value]
    if isinstance(value, dict):
        # Python JSON has no counterpart to JavaScript's undefined; every key
        # present here is part of the canonical payload, including nulls.
        return {key: canonicalize(value[key]) for key in sorted(value)}
    return value


_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("NaHCO3", "NaHCO₃"),
    ("H2CO3", "H₂CO₃"),
    ("PaCO2", "PaCO₂"),
    ("PACO2", "PACO₂"),
    ("EtCO2", "EtCO₂"),
    ("ETCO2", "ETCO₂"),
    ("PaO2", "PaO₂"),
    ("PAO2", "PAO₂"),
    ("FiO2", "FiO₂"),
    ("FIO2", "FIO₂"),
    ("SpO2", "SpO₂"),
    ("SPO2", "SPO₂"),
    ("SaO2", "SaO₂"),
    ("cmH2O", "cmH₂O"),
    ("HCO3-", "HCO₃⁻"),
    ("HCO3", "HCO₃⁻"),
    ("H2O", "H₂O"),
    ("N2O", "N₂O"),
    ("CO2", "CO₂"),
    ("SO42-", "SO₄²⁻"),
    ("SO4^2-", "SO₄²⁻"),
    ("NH4+", "NH₄⁺"),
    ("Fe3+", "Fe³⁺"),
    ("Fe2+", "Fe²⁺"),
    ("Ca2+", "Ca²⁺"),
    ("Mg2+", "Mg²⁺"),
    ("Na+", "Na⁺"),
    ("K+", "K⁺"),
)
_WORD_CHARS = re.compile(r"[a-zA-Z0-9₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]")
_NOTATION_PATTERN = re.compile("|".join(re.escape(key) for key, _ in _REPLACEMENTS))
_NOTATION_MAP = dict(_REPLACEMENTS)


def normalize_medical_notation_text(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        start, end = match.span()
        if start > 0 and _WORD_CHARS.search(text[start - 1]):
            return match.group(0)
        if end < len(text) and _WORD_CHARS.search(text[end]):
            return match.group(0)
        return _NOTATION_MAP[match.group(0)]

    return _NOTATION_PATTERN.sub(replace, text)


def normalize_question_payload(question: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(question)
    normalized["stem"] = normalize_medical_notation_text(question["stem"])
    normalized["context"] = normalize_medical_notation_text(question.get("context") or "")
    normalized_options = []
    for option in question["options"]:
        next_option = dict(option)
        next_option["text"] = normalize_medical_notation_text(option["text"])
        if isinstance(option.get("explanation"), str):
            next_option["explanation"] = normalize_medical_notation_text(option["explanation"])
        normalized_options.append(next_option)
    normalized["options"] = normalized_options
    return normalized


_CLOZE_RE = re.compile(r"\[_+\]")
_TRAP_RE = re.compile(
    r"\bappears? normal\b|\bwould be reassuring\b|\bdeceptively normal\b|'normal'|\"normal\"|‘normal’|'reassuring'|\blooks normal\b|\bactually represents\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\bdistinguishe?s?\b|\bdifferentiate\b|\bcompared to\b|\bcompared with\b|\b versus \b|\b vs\.? \b|\b(?:more|less) likely than\b|\bbetter discriminat(?:es|ing)\b|\bbest discriminator\b",
    re.IGNORECASE,
)
def _compile_all(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


_MECHANISM_RES = _compile_all(
    (
        r"\bwhat is the (?:underlying )?mechanism\b|\bwhat causes\b",
        r"\bwhy does\b|\bwhy is\b|\bwhy are\b|^why\b|\bwhy\??\s*$",
        r"\bwhich (?:mediator|cytokine|receptor|channel|enzyme|pathway)\b|\bvia which receptor\b",
        r"\bbest explains? the (?:cause|mechanism|pathophysiology)\b",
        r"\bwhat (?:is|are) responsible for\b|\bwhat underlies\b|\bwhat drives the\b|\bprimary driver\b",
    )
)
_LAB_TABLE_RE = re.compile(r"\|\s*[^|\n]+\|[^|\n]+\|")
_LAB_VALUES_RES = _compile_all(
    (
        r"\bABG\b|\bVBG\b|\bblood gas\b|\banion gap\b|\blactate\b",
        r"\bPaCO[₂2]?\b|\bPaO[₂2]?\b|\bHCO[₃3]?\b",
        r"\bspo[₂2]\b|\bsao[₂2]\b|\bfio[₂2]\b|\betco[₂2]\b",
    )
)
_INLINE_LAB_RE = re.compile(r"\bpH\s*(?:\d+\.\d+|\d\.?\d?)\b", re.IGNORECASE)
_INTERPRET_CUE_RE = re.compile(
    r"\binterpret(?:ation)?\b|\bmost likely interpretation\b|\bspirometry results\b|\bflow[/\\-]volume loop\b|\bECG (?:shows|demonstrates)\b|\bCXR (?:shows|demonstrates)\b",
    re.IGNORECASE,
)
_AGE_VIGNETTE_RE = re.compile(
    r"^\s*(?:a |an )?\d+[\-\s/]*(?:year|month|week|day)[\s-]*old\b", re.IGNORECASE
)
_PATIENT_VIGNETTE_RES = _compile_all(
    (
        r"^\s*(?:a |an )\s*(?:[\w-]+\s+){0,3}(?:woman|man|child|patient)\b",
        r"^\s*(?:a |an )\s*(?:[\w-]+\s+){0,3}(?:infant|baby|newborn|girl|boy)\b",
        r"^\s*(?:a |an )\s*(?:[\w-]+\s+){0,3}(?:adolescent|teenager|elderly)\b",
        r"^\s*(?:a |an )\s*(?:[\w-]+\s+){0,3}(?:pregnant|gravida|primigravida|multigravida)\b",
    )
)
_PROFESSIONAL_NARRATIVE_RE = re.compile(
    r"^\s*you are (?:a |an |the )?(?:gp|doctor|nurse|consultant|registrar|intern|resident|midwife|paramedic)\b",
    re.IGNORECASE,
)
_RECALL_OPENER_RE = re.compile(
    r"^\s*(?:what|which|where|when|who|how (?:much|many|long|often)|by what|at what|in (?:what|which)|name |list )",
    re.IGNORECASE,
)


def _is_interpretation(stripped: str, lower: str) -> bool:
    return bool(
        _LAB_TABLE_RE.search(stripped)
        or _INTERPRET_CUE_RE.search(lower)
        or (_INLINE_LAB_RE.search(stripped) and _matches_any(_LAB_VALUES_RES, lower))
    )


def _is_scenario(stripped: str) -> bool:
    return bool(
        _AGE_VIGNETTE_RE.search(stripped)
        or _matches_any(_PATIENT_VIGNETTE_RES, stripped)
        or _PROFESSIONAL_NARRATIVE_RE.search(stripped)
    )


def tag_format(stem: str, topics: list[str] | None, question_type: str | None) -> str:
    """Port of upstream's deterministic question-format heuristic."""
    stripped = (stem or "").strip()
    if not stripped:
        return "unknown"
    lower = stripped.lower()
    topic_values = [topic.lower() for topic in topics or []]
    rules = (
        ("cloze", lambda: bool(_CLOZE_RE.search(stripped))),
        ("trap", lambda: "trap" in topic_values or bool(_TRAP_RE.search(lower))),
        ("comparison", lambda: bool(_COMPARISON_RE.search(lower))),
        ("mechanism", lambda: _matches_any(_MECHANISM_RES, lower)),
        ("interpretation", lambda: _is_interpretation(stripped, lower)),
        ("scenario", lambda: _is_scenario(stripped)),
        ("recall", lambda: bool(_RECALL_OPENER_RE.search(stripped))),
    )
    for tag, matches in rules:
        if matches():
            return tag
    return "recall" if question_type == "recall" else "unknown"


def citation_envelope(question: dict[str, Any]) -> dict[str, Any] | None:
    cite = question.get("cite")
    has_public_usmle = "publicUsmle" in question
    if not cite and not has_public_usmle:
        return None
    envelope: dict[str, Any] = {}
    if cite:
        envelope["cite"] = cite
    if has_public_usmle:
        envelope["publicUsmle"] = question.get("publicUsmle")
    return envelope


def public_serving_fingerprint(
    question: dict[str, Any],
    *,
    source_file: str,
    baseline_question_ids: set[str] | frozenset[str] | None = None,
) -> str:
    """Fingerprint the source projection exactly as the TypeScript seed does."""
    normalized = normalize_question_payload(question)
    module_nodes = list(normalized.get("moduleNodes") or [])
    if baseline_question_ids and normalized["id"] in baseline_question_ids:
        if BASELINE_MODULE not in module_nodes:
            module_nodes.append(BASELINE_MODULE)
    payload = {
        "id": normalized["id"],
        "stem": normalized["stem"],
        "options": normalized["options"],
        "context": normalized["context"],
        "rotation": normalized["rotation"],
        "week": normalized.get("week") if isinstance(normalized.get("week"), int) else None,
        "topics": normalized["topics"],
        "moduleNodes": module_nodes,
        "source": "bank",
        "sourceFile": source_file,
        "questionType": normalized["questionType"],
        "difficulty": normalized["difficulty"],
        "format": tag_format(
            normalized["stem"], normalized.get("topics"), normalized.get("questionType")
        ),
        "variantGroupId": normalized.get("variantGroupId"),
        "variantType": normalized.get("variantType"),
        "imageUrl": normalized.get("imageUrl"),
        "imageCaption": normalized.get("imageCaption"),
        "citations": citation_envelope(normalized),
        "crosslinks": normalized.get("crosslinks"),
        "annotations": normalized.get("annotations"),
        "abbreviations": normalized.get("abbreviations"),
        "combinations": normalized.get("combinations"),
        "correctVariants": normalized.get("correctVariants"),
    }
    return sha256_json(canonicalize(payload))


def module_item_content_hash(question: dict[str, Any]) -> str:
    read = {
        "stem": question["stem"],
        "context": question.get("context"),
        "options": [
            {
                "label": option["label"],
                "text": option["text"],
                "isCorrect": option["isCorrect"],
                "explanation": option["explanation"],
            }
            for option in question["options"]
        ],
        "evidence": question["publicModule"]["evidence"],
    }
    return sha256_json(read)


def module_source_fingerprint(source_id: str, source: dict[str, Any]) -> str:
    return sha256_json(
        [
            source_id,
            source["title"],
            source.get("publisher"),
            source["url"],
            source["verifiedAt"],
            source["licence"]["cls"],
        ]
    )


def cohort_card_serving_fingerprint(card: dict[str, Any]) -> str:
    projection = [
        card["id"],
        "cohort-open",
        card["cardType"],
        card["front"],
        card["back"],
        card.get("context"),
        card.get("imageUrl"),
        card.get("imageCaption"),
        sorted([f"cohort/{card['discipline']}"]),
        card.get("variantGroup"),
        card.get("variantIndex"),
    ]
    return sha256_json(projection)


def open_usmle_quote_set_sha256(sources: list[dict[str, Any]]) -> str:
    quote_set = [
        [source["id"], passage["id"], passage["quote"]]
        for source in sources
        for passage in source["passages"]
        if passage.get("quote")
    ]
    return sha256_json(quote_set)


def open_usmle_registry_sha256(verified_at: str, sources: list[dict[str, Any]]) -> str:
    receipt = [
        verified_at,
        [
            [
                source["id"],
                source["title"],
                source["publisher"],
                source["canonicalUrl"],
                source["attribution"],
                source["licence"]["cls"],
                source["licence"]["id"],
                source["licence"]["url"],
                [
                    [passage["id"], passage["locator"], passage.get("quote")]
                    for passage in source["passages"]
                ],
            ]
            for source in sources
        ],
    ]
    return sha256_json(receipt)
