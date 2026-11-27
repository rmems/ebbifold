"""Pure ports of the small policy modules candidate ranking depends on.

Mirrors specific-topic-overlap, session-filters, media-presence,
challenge-policy, recent-neighbor-penalty, review-learning-policy,
review-challenge, curriculum-pacing, figure-cooldown, teaching-pace and
question-retirement.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .scoring import normalize_review_challenge_level
from .topics import (
    BLOCK_WEEKS,
    normalize_curriculum_topic,
    normalize_topic,
    teaching_week_for_topics,
)

BROAD_TOPIC_WORDS = frozenset("""
a an and of in the with acute adult adults adolescent adolescents child children
infant infants neonate neonates neonatal newborn newborns paediatric paediatrics
pediatric pediatrics toddler toddlers cah paam pwh toc usmle critical care anatomy
cardiac cardiology community dental dentistry dermatology developmental emergency
emergencies endocrine endocrinology ent urology psychiatry radiology nephrology
neonatology allergy obstetrics anaesthesia anesthesia trauma gastroenterology general
genetics haematology hematology immunology infectious infection infections disease
diseases medical medicine microbiology musculoskeletal neurology oncology
ophthalmology orthopaedics orthopedics pathology physiology renal respiratory
rheumatology surgery surgical assessment basic classification clinical complications
diagnosis differential disorder disorders epidemiology examination features health
imaging interpretation investigation investigations management mechanism
pathophysiology presentation prevention prognosis recall recognition science signs
status symptoms syndrome syndromes therapy treatment additional essentials pearls
principles sick
""".split())

_SPECIFIC_PUNCT = re.compile(r"[()&/]")


def specific_clinical_topics(topics: Iterable[str]) -> list[str]:
    """Insertion-ordered set of normalized topics that are not purely broad words."""
    result: dict[str, None] = {}
    for topic in topics:
        normalized = normalize_topic(_SPECIFIC_PUNCT.sub(" ", topic))
        if not normalized:
            continue
        if all(word in BROAD_TOPIC_WORDS for word in normalized.split(" ")):
            continue
        result[normalized] = None
    return list(result)


def has_specific_clinical_topic_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    specific_left = set(specific_clinical_topics(left))
    if not specific_left:
        return False
    return any(t in specific_left for t in specific_clinical_topics(right))


COOLDOWN_HOURS = 24
_HOUR_MS = 60 * 60 * 1000


def compute_topic_cooldown_penalty(topics: list[str], exposures: Mapping[str, Mapping[str, float]], now_ms: float) -> float:
    if not topics:
        return 0
    max_penalty = 0.0
    for topic in topics:
        exposure = exposures.get(topic)
        if not exposure:
            continue
        hours_ago = (now_ms - exposure["mostRecentMs"]) / _HOUR_MS
        penalty = exposure["count"] * max(0, 1 - hours_ago / COOLDOWN_HOURS)
        if penalty > max_penalty:
            max_penalty = penalty
    return min(1, max_penalty)


_BANK_CARD_PENALTIES = {0: 0, 1: 0.3, 2: 0.7}


def compute_bank_card_penalty(source_file: str | None, total_reviews: int) -> float:
    if not source_file or not source_file.endswith("-cards"):
        return 0
    return _BANK_CARD_PENALTIES.get(total_reviews, 0.95)


PACING_AHEAD_PENALTY = 3
PACING_LAST_WEEK_BOOST = 2
_PACING_MAX_STEPS = 3
MEDIA_PRESENCE_BOOST = -(PACING_AHEAD_PENALTY / 3)


def media_presence_boost(image_url: Any, clip_id: Any) -> float:
    has = (isinstance(image_url, str) and image_url != "") or (isinstance(clip_id, str) and clip_id != "")
    return MEDIA_PRESENCE_BOOST if has else 0


STANDARD_CHALLENGE_RECALL = 0.6
STRETCH_CHALLENGE_RECALL = 0.8
MAX_COMPLEXITY = 5
LADDER_STEP = 4


def complexity_ladder_boost(complexity: float | None, mode: str = "frontier") -> float:
    bounded = max(1, min(MAX_COMPLEXITY, 2 if complexity is None else complexity))
    if mode == "step-down":
        return (bounded - 1) * LADDER_STEP
    return (MAX_COMPLEXITY - bounded) * LADDER_STEP


RECENT_NEIGHBOR_SIMILARITY_THRESHOLD = 0.85
MAX_RECENT_NEIGHBOR_PENALTY = 0.95
RECENT_NEIGHBOR_MAX_RANK_SINK = 3


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def recent_neighbor_signal(similar_cards: Any, recent_card_ids: Any) -> dict[str, Any]:
    if not recent_card_ids or not isinstance(similar_cards, list):
        return {"maxSimilarity": None, "penalty": 0}
    max_sim: float | None = None
    for entry in similar_cards:
        if not isinstance(entry, dict):
            continue
        cid, sim = entry.get("cardId"), entry.get("similarity")
        if not isinstance(cid, str) or cid not in recent_card_ids or not _is_finite_number(sim):
            continue
        max_sim = sim if max_sim is None else max(max_sim, sim)
    if max_sim is None or max_sim <= RECENT_NEIGHBOR_SIMILARITY_THRESHOLD:
        return {"maxSimilarity": max_sim, "penalty": 0}
    normalized = (max_sim - RECENT_NEIGHBOR_SIMILARITY_THRESHOLD) / (1 - RECENT_NEIGHBOR_SIMILARITY_THRESHOLD)
    return {"maxSimilarity": max_sim, "penalty": min(MAX_RECENT_NEIGHBOR_PENALTY, max(0, normalized))}


# --- review-learning-policy / review-challenge -----------------------------


def review_learning_policy(level: Any) -> dict[str, Any]:
    n = normalize_review_challenge_level(level)
    return {
        "level": n,
        "questionsOnly": n == 2,
        "scaffoldsOnly": n == -2,
        "preferredQuestionShare": {2: 1, -2: 0, 1: 0.7, -1: 0.3}.get(n, 0.4),
        "preferredCardShare": {2: 0, -2: 1, 1: 0.3, -1: 0.7}.get(n, 0.6),
    }


def card_eligible_for_review_learning(complexity: Any, level: Any) -> bool:
    policy = review_learning_policy(level)
    if policy["questionsOnly"]:
        return False
    if not policy["scaffoldsOnly"]:
        return True
    return complexity == 1


def question_eligible_for_review_learning(
    level: Any, *, is_scaffold: bool | None = None, difficulty: str | None = None, demonstrated_gap: bool | None = None
) -> bool:
    policy = review_learning_policy(level)
    if policy["scaffoldsOnly"]:
        return False
    if policy["questionsOnly"]:
        return demonstrated_gap is True and difficulty == "hard" and is_scaffold is not True
    return True


def modality_eligible_for_review_learning(modality: str, level: Any) -> bool:
    policy = review_learning_policy(level)
    if policy["questionsOnly"]:
        return modality == "question"
    if policy["scaffoldsOnly"]:
        return modality == "card"
    return True


def has_demonstrated_learning_gap(
    *, recall_probability: Any = None, recent_failure: Any = None, recent_fail_rate: Any = None,
    last_probe_at_ms: float | None = None, now_ms: float, probe_count: Any = None,
) -> bool:
    if not _is_finite_number(probe_count) or probe_count <= 0:
        return False
    if recent_failure is True:
        return True
    if _is_finite_number(recall_probability) and recall_probability < 0.6:
        return True
    rate = isinstance(recent_fail_rate, (int, float)) and not isinstance(recent_fail_rate, bool) and recent_fail_rate > 0
    return bool(rate and last_probe_at_ms is not None and now_ms - last_probe_at_ms <= 24 * _HOUR_MS)


EMPIRICAL_FACILITY_SAMPLE_FLOOR = 10


def shrunk_facility(facility: Any, sample_size: Any) -> float | None:
    if not _is_finite_number(facility) or not _is_finite_number(sample_size) or sample_size < EMPIRICAL_FACILITY_SAMPLE_FLOOR:
        return None
    bounded = max(0, min(1, facility))
    weight = sample_size / (sample_size + EMPIRICAL_FACILITY_SAMPLE_FLOOR)
    return 0.5 + weight * (bounded - 0.5)


def shift_question_difficulty(difficulty: str, level: Any, slot_index: int = 0) -> str:
    index = 0 if difficulty == "easy" else 2 if difficulty == "hard" else 1
    n = normalize_review_challenge_level(level)
    if n == 0:
        return difficulty
    wave = math.sin((2 * math.pi * slot_index) / 6 - math.pi / 2)
    magnitude = 2 if abs(n) == 2 else 1
    shift = (magnitude if wave >= 0.45 else 0) if n > 0 else (-magnitude if wave <= -0.45 else 0)
    shifted = max(0, min(2, index + shift))
    return "easy" if shifted == 0 else "hard" if shifted == 2 else "medium"


_PROTECTED_WAVE_REASONS = frozenset({
    "failure_escalation", "stuck_intervention", "pre_teach", "pre_teach_naive",
    "chronic_stuck_mcq", "mcq_bridge_card", "preemptive_scaffold", "statement_scaffold",
})


def _content_challenge_rank(item: Mapping[str, Any]) -> int:
    c = item.get("complexity")
    if _is_finite_number(c):
        c = max(1, c)
        return 0 if c <= 1 else 1 if c == 2 else 2
    d = item.get("difficulty")
    return 2 if d == "hard" else 0 if d == "easy" else 1


def apply_review_challenge_wave(items: list[dict[str, Any]], level: Any) -> list[dict[str, Any]]:
    n = normalize_review_challenge_level(level)
    if n == 0 or len(items) < 3:
        return list(items)
    scaffold_targets = {
        (i.get("struggleIntervention") or {}).get("targetCardId")
        for i in items
        if (i.get("struggleIntervention") or {}).get("isScaffold") and (i.get("struggleIntervention") or {}).get("targetCardId")
    }

    def is_fixed(item: Mapping[str, Any]) -> bool:
        si = item.get("struggleIntervention") or {}
        return (item.get("interventionReason") or "") in _PROTECTED_WAVE_REASONS or si.get("isScaffold") is True or (
            item.get("id") is not None and item.get("id") in scaffold_targets
        )

    def wave_chunk(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = list(group)
        remaining = [i for i in group if not is_fixed(i)]
        hard_run = 0
        for position, fixed in enumerate(group):
            if is_fixed(fixed):
                hard_run = hard_run + 1 if _content_challenge_rank(fixed) == 2 else 0
                continue
            desired = max(-1, min(1, 0.85 * math.sin((2 * math.pi * position) / 6 - math.pi / 2) + n * 0.08))
            same_type = [i for i in remaining if i.get("type") == fixed.get("type")]
            candidates = [i for i in same_type if hard_run < 2 or _content_challenge_rank(i) < 2]
            pool = candidates or same_type
            best_index, best_score = 0, math.inf
            for cand in pool:
                idx = next(k for k, r in enumerate(remaining) if r is cand)
                score = abs(_content_challenge_rank(cand) - 1 - desired)
                if score < best_score:
                    best_index, best_score = idx, score
            picked = remaining.pop(best_index)
            result[position] = picked
            hard_run = hard_run + 1 if _content_challenge_rank(picked) == 2 else 0
        return result

    output: list[dict[str, Any]] = []
    for batch_start in range(0, len(items), 15):
        batch = items[batch_start:batch_start + 15]
        start = 0
        while start < len(batch):
            stage = batch[start].get("examTargetMasteryStage") or "__ordinary__"
            end = start + 1
            while end < len(batch) and (batch[end].get("examTargetMasteryStage") or "__ordinary__") == stage:
                end += 1
            output.extend(wave_chunk(batch[start:end]))
            start = end
    return output


# --- curriculum pacing / figure cooldown / teaching pace ------------------


def curriculum_pacing_boost(item_week: int | None, current_week: int | None) -> float:
    if item_week is None or current_week is None or current_week > BLOCK_WEEKS:
        return 0
    steps = item_week - current_week
    return 0 if steps <= 0 else min(steps, _PACING_MAX_STEPS) * PACING_AHEAD_PENALTY


def curriculum_recency_boost(item_week: int | None, current_week: int | None) -> float:
    if item_week is None or current_week is None or current_week > BLOCK_WEEKS:
        return 0
    if item_week == current_week:
        return -4
    if item_week == current_week - 1:
        return -PACING_LAST_WEEK_BOOST
    return 0


def figure_cooldown_hours(count: float) -> float:
    if count <= 0:
        return 0
    return min(24 * 3 ** (count - 1), 24 * 60)


def figure_cooldown_boost(image_url: Any, exposures: Mapping[str, Mapping[str, float]] | None, now_ms: float) -> float:
    if not image_url or not exposures:
        return 0
    exposure = exposures.get(image_url)
    if not exposure or exposure["count"] <= 0:
        return 0
    elapsed = (now_ms - exposure["mostRecentMs"]) / _HOUR_MS
    if elapsed < 0:
        return 0
    window = figure_cooldown_hours(exposure["count"])
    if elapsed >= window:
        return 0
    return 3 * 3 * (1 - elapsed / window)


_WEEK_RE = re.compile(r"^week(\d+)(?:[-.]|$)")
_CONTENT_TO_TEACHING = {"cah": {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4}}
_HAS_TEACHING_WEEKS = frozenset({"critical-care", "paam", "cah"})


def teaching_week_for_source_file(rotation: str, source_file: str | None) -> int | None:
    if rotation not in _HAS_TEACHING_WEEKS or not source_file:
        return None
    match = _WEEK_RE.match(source_file)
    if not match:
        return None
    week = int(match.group(1))
    if week <= 0:
        return None
    mapped = _CONTENT_TO_TEACHING.get(rotation)
    return mapped.get(week) if mapped else week



def item_teaching_week(rotation: str, source_file: str | None, topics: Any, topic_weeks: Any) -> int | None:
    week = teaching_week_for_topics(topics, topic_weeks)
    return week if week is not None else teaching_week_for_source_file(rotation, source_file)


# --- question retirement / contrast-set siblings ---------------------------

MASTERY_CORRECT_THRESHOLD = 1
MASTERED_REENTRY_CAP_PER_SESSION = 1
CONTRAST_SET_VARIANT_TYPE = "contrast-set"


def freshness_tier(familiarity: Mapping[str, Any] | None) -> int:
    if familiarity is None:
        return 0
    return 2 if familiarity.get("corrects", 0) >= MASTERY_CORRECT_THRESHOLD else 1


def take_with_reentry_cap(question_id: str, familiarity: Mapping[str, Any], counter: dict[str, int]) -> bool:
    if freshness_tier(familiarity.get(question_id)) != 2:
        return True
    if counter["masteredServed"] >= MASTERED_REENTRY_CAP_PER_SESSION:
        return False
    counter["masteredServed"] += 1
    return True


def partition_by_freshness(items: list[Any], get_id: Any, familiarity: Mapping[str, Any]) -> list[Any]:
    if len(items) <= 1 or not familiarity:
        return items
    tiers: list[list[Any]] = [[], [], []]
    for item in items:
        tiers[freshness_tier(familiarity.get(get_id(item)))].append(item)
    for tier in (1, 2):
        indexed = list(enumerate(tiers[tier]))
        indexed.sort(key=lambda p: (familiarity[get_id(p[1])].get("lastSeenAtMs", 0), p[0]))
        tiers[tier] = [x for _, x in indexed]
    return tiers[0] + tiers[1] + tiers[2]


def prioritize_least_recently_served_contrast_siblings(
    candidates: list[Any], get_id: Any, familiarity: Mapping[str, Any]
) -> list[Any]:
    if len(candidates) <= 1:
        return candidates
    families: dict[str, list[int]] = {}
    for index, cand in enumerate(candidates):
        if cand.get("variantType") != CONTRAST_SET_VARIANT_TYPE or not cand.get("variantGroupId"):
            continue
        families.setdefault(cand["variantGroupId"], []).append(index)
    result = list(candidates)

    def served(pos: int) -> float:
        fam = familiarity.get(get_id(candidates[pos]))
        return fam.get("lastSeenAtMs", -math.inf) if fam is not None else -math.inf

    for positions in families.values():
        if len(positions) <= 1:
            continue
        ordered = sorted(enumerate(positions), key=lambda x: (served(x[1]), x[0]))
        for k, pos in enumerate(positions):
            result[pos] = candidates[ordered[k][1]]
    return result
