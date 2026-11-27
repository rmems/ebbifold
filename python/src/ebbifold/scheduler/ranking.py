"""Pure port of candidate-ranking.ts candidate pools used by scheduler lanes.

``bulk`` is the explicit BulkCandidates snapshot: card/question/video rows,
precomputed vector scores (``{conceptId: {itemId: score}}`` in insertion
order), exam-relevance maps and familiarity. Sets arrive as lists and are
normalized by :func:`normalize_bulk`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .difficulty import QUESTION_DIFFICULTIES, build_difficulty_plan, get_effective_question_difficulty
from .errors import UnsupportedSchedulerInput
from .policies import (
    MAX_RECENT_NEIGHBOR_PENALTY,
    RECENT_NEIGHBOR_MAX_RANK_SINK,
    card_eligible_for_review_learning,
    complexity_ladder_boost,
    compute_bank_card_penalty,
    compute_topic_cooldown_penalty,
    curriculum_pacing_boost,
    curriculum_recency_boost,
    figure_cooldown_boost,
    item_teaching_week,
    media_presence_boost,
    partition_by_freshness,
    prioritize_least_recently_served_contrast_siblings,
    question_eligible_for_review_learning,
    recent_neighbor_signal,
    shift_question_difficulty,
    shrunk_facility,
    specific_clinical_topics,
    take_with_reentry_cap,
)
from .topics import dated_topic_recency_boost, expand_topic_set

EXAM_RELEVANCE_BOOST_FACTOR = 5
_SET_FIELDS = ("passSetDeferredCardIds", "failedStretchCardIds", "passedScaffoldCardIds", "fragileSeenCards")
CARD_FIELDS = (
    "id", "clusterId", "similarCards", "topics", "sourceFile", "imageUrl", "clipId",
    "importance", "complexity", "variantGroupId", "variantIndex", "variantType",
)


def normalize_bulk(raw: Mapping[str, Any]) -> dict[str, Any]:
    bulk = dict(raw)
    for key in ("unseenCards", "seenCards", "rotationQuestions", "rotationVideos", "questionConceptLinks", "videoConceptLinks"):
        bulk[key] = list(bulk.get(key) or [])
    for key in _SET_FIELDS:
        bulk[key] = set(bulk.get(key) or [])
    for key in ("cardScores", "questionScores", "videoScores"):
        bulk[key] = dict(bulk.get(key) or {})
    for key in ("cardExamRelevance", "questionExamRelevance", "seenCardReviewCounts", "questionFamiliarity"):
        bulk[key] = dict(bulk.get(key) or {})
    bulk["cardVariantGroupHistory"] = {k: set(v) for k, v in (bulk.get("cardVariantGroupHistory") or {}).items()}
    bulk["questionMap"] = {q["id"]: q for q in bulk["rotationQuestions"]} | dict(bulk.get("questionMap") or {})
    bulk["videoMap"] = {v["id"]: v for v in bulk["rotationVideos"]} | dict(bulk.get("videoMap") or {})
    bulk.setdefault("rotation", "")
    return bulk


def build_rank_from_scores(scores: Mapping[str, float] | None, limit: int) -> dict[str, int] | None:
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
    return {item_id: i for i, (item_id, _) in enumerate(ordered)}


def _zero_nan(value: float) -> float:
    return 0 if value != value or value == 0 else value


def exam_target_rank_boost(score: float | None, factor: float = EXAM_RELEVANCE_BOOST_FACTOR) -> float:
    if score is None:
        return 0
    if score != score:
        return 0
    return _zero_nan(-(max(0, min(1, score)) * factor))


def apply_bounded_exam_target_nudge(ordered: list[Any], get_score: Any, max_move: float = EXAM_RELEVANCE_BOOST_FACTOR) -> list[Any]:
    if max_move is None:
        max_move = EXAM_RELEVANCE_BOOST_FACTOR
    keyed = [(rank + exam_target_rank_boost(get_score(c), max_move), rank, c) for rank, c in enumerate(ordered)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in keyed]


def _card_candidate(card: Mapping[str, Any]) -> dict[str, Any]:
    return {k: card.get(k) for k in CARD_FIELDS}


def get_cards_from_bulk(
    concept_id: str,
    concept: Mapping[str, Any],
    max_candidates: int,
    bulk: Mapping[str, Any],
    selected_card_ids: set[str],
    penalty_context: Mapping[str, Any] | None = None,
    concept_recall: float | None = None,
    current_teaching_week: int | None = None,
    exam_target_scores: Mapping[str, float] | None = None,
    exam_target_max_rank_move: float = EXAM_RELEVANCE_BOOST_FACTOR,
    topic_teaching_weeks: Mapping[str, int] | None = None,
    recent_figure_exposures: Mapping[str, Any] | None = None,
    figure_now_ms: float = 0,
    review_challenge: int = 0,
    topic_recency_days: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    topic_set = set(expand_topic_set(concept["topics"]))
    if not topic_set:
        return []
    matching_unseen: list[dict[str, Any]] = []
    matching_seen: list[dict[str, Any]] = []
    for card in bulk["unseenCards"]:
        if card["id"] in selected_card_ids or card["id"] in bulk["passSetDeferredCardIds"]:
            continue
        if card_eligible_for_review_learning(card.get("complexity"), review_challenge) and any(t in topic_set for t in card["topics"]):
            matching_unseen.append(_card_candidate(card))
    for card in bulk["seenCards"]:
        if card["id"] in selected_card_ids:
            continue
        if card_eligible_for_review_learning(card.get("complexity"), review_challenge) and any(t in topic_set for t in card["topics"]):
            matching_seen.append(_card_candidate(card))

    vector_rank = build_rank_from_scores(bulk["cardScores"].get(concept_id), 200)
    exam_scores = exam_target_scores if exam_target_scores is not None else bulk["cardExamRelevance"]
    history = bulk["cardVariantGroupHistory"]
    failed = bulk["failedStretchCardIds"]
    step_down = bool(failed) and any(c["id"] in failed for c in matching_seen)
    recent_ids = (penalty_context or {}).get("recentCardIds") or set()
    recent_failures = (penalty_context or {}).get("recentFailureConceptIds") or set()

    def importance_boost(importance: Any) -> float:
        return -10 if importance == 3 else -3 if importance == 2 else 0

    def variant_boost(card: Mapping[str, Any]) -> float:
        group = card.get("variantGroupId")
        if group is None:
            return 0
        seen = history.get(group)
        if not seen:
            return 0
        return 4 if card["id"] in seen else -4

    def ladder(card: Mapping[str, Any]) -> float:
        return complexity_ladder_boost(card.get("complexity"), "step-down" if step_down else "frontier")

    def pacing(card: Mapping[str, Any]) -> float:
        week = item_teaching_week(bulk["rotation"], card.get("sourceFile"), card["topics"], topic_teaching_weeks)
        return curriculum_pacing_boost(week, current_teaching_week) + min(
            curriculum_recency_boost(week, current_teaching_week),
            dated_topic_recency_boost(card["topics"], topic_recency_days),
        )

    def figure(card: Mapping[str, Any]) -> float:
        return figure_cooldown_boost(card.get("imageUrl"), recent_figure_exposures, figure_now_ms)

    def media(card: Mapping[str, Any]) -> float:
        return media_presence_boost(card.get("imageUrl"), card.get("clipId"))

    def soft(card: Mapping[str, Any]) -> float:
        return pacing(card) + figure(card) + media(card)

    def fallback_key(card: Mapping[str, Any]) -> tuple[float, float, float, float]:
        importance = card.get("importance")
        return (variant_boost(card), ladder(card), soft(card), -(1 if importance is None else importance))

    def recent_neighbor(ordered: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
        if not enabled or not recent_ids or concept_id in recent_failures:
            return ordered
        rows = []
        for i, card in enumerate(ordered):
            penalty = recent_neighbor_signal(card.get("similarCards"), recent_ids)["penalty"]
            rows.append((i + (penalty / MAX_RECENT_NEIGHBOR_PENALTY) * RECENT_NEIGHBOR_MAX_RANK_SINK, penalty, i, card))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        return [r[3] for r in rows]

    def nudge(ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return apply_bounded_exam_target_nudge(ordered, lambda c: exam_scores.get(c["id"]), exam_target_max_rank_move)

    def rank_candidates(candidates: list[dict[str, Any]], novelty: bool) -> list[dict[str, Any]]:
        if vector_rank is None:
            return recent_neighbor(nudge(sorted(candidates, key=fallback_key)), novelty)
        ranked, unranked = [], []
        for card in candidates:
            base = vector_rank.get(card["id"])
            if base is None:
                unranked.append(card)
            else:
                importance = card.get("importance")
                rank = (base + importance_boost(1 if importance is None else importance) + variant_boost(card)
                        + ladder(card) + pacing(card) + figure(card) + media(card))
                ranked.append((rank, card))
        ranked.sort(key=lambda r: r[0])
        unranked.sort(key=fallback_key)
        return recent_neighbor(nudge([c for _, c in ranked]), novelty) + recent_neighbor(nudge(unranked), novelty)

    passed_ids = bulk["passedScaffoldCardIds"]
    passed = [c for c in matching_seen if c["id"] in passed_ids] if passed_ids else []

    def is_block(card: Mapping[str, Any]) -> bool:
        return (2 if card.get("complexity") is None else card["complexity"]) <= 2

    if step_down:
        both = matching_unseen + matching_seen
        combined = rank_candidates([c for c in both if is_block(c)], False) + rank_candidates(
            [c for c in both if not is_block(c)], False
        )
    else:
        passed_ref = {id(c) for c in passed}
        frontier = [c for c in matching_unseen if not is_block(c)] + [c for c in matching_seen if not is_block(c)]
        combined = (
            rank_candidates(frontier, True)
            + rank_candidates([c for c in matching_unseen if is_block(c)], True)
            + rank_candidates([c for c in matching_seen if is_block(c) and id(c) not in passed_ref], False)
            + rank_candidates(passed, False)
        )

    if penalty_context is not None:
        rows = []
        for i, card in enumerate(combined):
            topic_penalty = compute_topic_cooldown_penalty(
                card["topics"], penalty_context.get("recentTopicExposures") or {}, penalty_context["nowMs"]
            )
            bank_penalty = compute_bank_card_penalty(card.get("sourceFile"), bulk["seenCardReviewCounts"].get(card["id"], 0))
            penalty = max(topic_penalty, bank_penalty)
            if card["id"] in bulk["fragileSeenCards"]:
                penalty -= 1.0
            rows.append((penalty, i, card))
        rows.sort(key=lambda r: (r[0], r[1]))
        combined = [r[2] for r in rows]

    if review_challenge != 0 and not step_down:
        seen_groups = {c.get("variantGroupId") for c in matching_seen if c.get("variantGroupId")}
        unseen_by_id = {
            c["id"]: c for c in bulk["unseenCards"] if not c.get("variantGroupId") or c["variantGroupId"] not in seen_groups
        }
        target = {-2: 1, -1: 2, 1: 4}.get(review_challenge, 5)

        def cx(card: Mapping[str, Any]) -> float:
            return 2 if card.get("complexity") is None else card["complexity"]

        unseen = [c for c in combined if c["id"] in unseen_by_id]
        authored = sorted(enumerate(unseen), key=lambda p: (abs(cx(p[1]) - target), p[0]))
        preferred = []
        for index, (_, card) in enumerate(authored):
            meta = unseen_by_id.get(card["id"]) or {}
            quality = shrunk_facility(meta.get("facilityIndex"), meta.get("sampleSize"))
            rank = index + (0 if quality is None else 2 * review_challenge * (quality - 0.5))
            preferred.append((abs(cx(card) - target), rank, index, card))
        preferred.sort(key=lambda r: (r[0], r[1], r[2]))
        it = iter(preferred)
        combined = [next(it)[3] if c["id"] in unseen_by_id else c for c in combined]

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for card in combined:
        if len(out) >= max_candidates:
            break
        if card["id"] in seen_ids:
            continue
        seen_ids.add(card["id"])
        out.append(card)
    return out


def get_videos_from_bulk(
    concept_id: str, concept: Mapping[str, Any], limit: int, bulk: Mapping[str, Any], exclude_video_ids: set[str]
) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    for link in bulk["videoConceptLinks"]:
        if link["conceptId"] != concept_id or link["videoId"] in exclude_video_ids:
            continue
        video = bulk["videoMap"].get(link["videoId"])
        if video:
            linked.append(video)
    topic_matched: list[dict[str, Any]] = []
    if expand_topic_set(concept["topics"]):
        linked_ids = {v["id"] for v in linked}
        for video in bulk["rotationVideos"]:
            if video["id"] in exclude_video_ids or video["id"] in linked_ids:
                continue
            if concept_id in video["id"]:
                topic_matched.append(video)
    vector_rank = build_rank_from_scores(bulk["videoScores"].get(concept_id), 100)

    def by_vector(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if vector_rank is None or len(videos) <= 1:
            return videos
        ranked = sorted((r for r in ((vector_rank.get(v["id"]), v) for v in videos) if r[0] is not None), key=lambda r: r[0])
        return [v for _, v in ranked] + [v for v in videos if vector_rank.get(v["id"]) is None]

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for video in by_vector(linked) + by_vector(topic_matched):
        if len(out) >= limit:
            break
        if video["id"] in seen:
            continue
        seen.add(video["id"])
        out.append(video)
    return out


def _suppression_key(q: Mapping[str, Any]) -> str | None:
    from .build_state import question_suppression_key

    return question_suppression_key(q)


def _question_pool(concept_id: str, concept: Mapping[str, Any], bulk: Mapping[str, Any], selected: set[str]):
    used = set(selected)
    primary: list[dict[str, Any]] = []
    for link in bulk["questionConceptLinks"]:
        if link["conceptId"] != concept_id or not link.get("isPrimary") or link["questionId"] in used:
            continue
        q = bulk["questionMap"].get(link["questionId"])
        if q:
            used.add(link["questionId"])
            primary.append(q)
    linked: list[dict[str, Any]] = []
    if not primary:
        for link in bulk["questionConceptLinks"]:
            if link["conceptId"] != concept_id or link["questionId"] in used:
                continue
            q = bulk["questionMap"].get(link["questionId"])
            if q:
                used.add(link["questionId"])
                linked.append(q)
    topic_set = set(specific_clinical_topics(concept["topics"]))
    topic_matched: list[dict[str, Any]] = []
    if topic_set:
        for q in bulk["rotationQuestions"]:
            if q["id"] in used:
                continue
            if any(t in topic_set for t in specific_clinical_topics(q["topics"])):
                used.add(q["id"])
                topic_matched.append(q)
    return primary, linked, topic_matched


def get_questions_from_bulk(
    concept_id: str,
    concept: Mapping[str, Any],
    limit: int,
    bulk: Mapping[str, Any],
    *,
    selected_question_ids: set[str],
    selected_question_variant_groups: set[str],
    difficulty_plan: list[str] | None = None,
    recent_formats_by_concept: Mapping[str, Any] | None = None,
    concept_thread_anchors: list[Any] | None = None,
    exam_target_scores: Mapping[str, float] | None = None,
    exam_target_max_rank_move: float | None = None,
    mastered_reentry_counter: dict[str, int] | None = None,
    current_teaching_week: int | None = None,
    topic_teaching_weeks: Mapping[str, int] | None = None,
    review_challenge: int | None = None,
    topic_recency_days: Mapping[str, float] | None = None,
    review_challenge_gap: bool | None = None,
    challenge_slot_index: int = 0,
) -> list[dict[str, Any]]:
    if concept_thread_anchors:
        raise UnsupportedSchedulerInput("concept-thread question matching is not ported in Phase 3a")
    if not question_eligible_for_review_learning(
        review_challenge, demonstrated_gap=review_challenge_gap, difficulty="hard" if review_challenge == 2 else None
    ):
        return []
    plan = build_difficulty_plan(difficulty_plan if difficulty_plan is not None else ["medium"], limit)
    endpoint_plan = build_difficulty_plan(["hard"], limit) if review_challenge == 2 else plan
    primary, linked, topic_matched = _question_pool(concept_id, concept, bulk, selected_question_ids)
    vector_rank = build_rank_from_scores(bulk["questionScores"].get(concept_id), 200)
    exam_scores = exam_target_scores if exam_target_scores is not None else bulk["questionExamRelevance"]
    move = EXAM_RELEVANCE_BOOST_FACTOR if exam_target_max_rank_move is None else exam_target_max_rank_move

    def q_pacing(row: Mapping[str, Any]) -> float:
        week = item_teaching_week(bulk["rotation"], None, row["topics"], topic_teaching_weeks)
        return curriculum_pacing_boost(week, current_teaching_week) + min(
            curriculum_recency_boost(week, current_teaching_week),
            dated_topic_recency_boost(row["topics"], topic_recency_days),
        )

    def nudge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return apply_bounded_exam_target_nudge(rows, lambda c: exam_scores.get(c["id"]), move)

    def by_vector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_pacing = sorted(rows, key=q_pacing)
        if vector_rank is None or len(rows) <= 1:
            return nudge(by_pacing)
        ranked = [(vector_rank[q["id"]] + q_pacing(q), q) for q in rows if q["id"] in vector_rank]
        ranked.sort(key=lambda r: r[0])
        return nudge([q for _, q in ranked]) + nudge(sorted((q for q in rows if q["id"] not in vector_rank), key=q_pacing))

    ordered = by_vector(primary) + by_vector(linked) + by_vector(topic_matched)
    bank_preferred = [c for c in ordered if c.get("source") == "bank"] + [c for c in ordered if c.get("source") != "bank"]
    by_difficulty: dict[str, list[dict[str, Any]]] = {d: [] for d in QUESTION_DIFFICULTIES}
    for cand in bank_preferred:
        if review_challenge == 2 and cand.get("difficulty") == "hard":
            effective = "hard"
        else:
            effective = get_effective_question_difficulty(
                cand.get("difficulty"), cand.get("facilityIndex"), cand.get("totalAttempts")
            )
        by_difficulty[effective].append(cand)

    seen_formats = (recent_formats_by_concept or {}).get(concept_id)
    if seen_formats:
        for d, rows in by_difficulty.items():
            by_difficulty[d] = [c for c in rows if not (c.get("format") and c["format"] in seen_formats)] + [
                c for c in rows if c.get("format") and c["format"] in seen_formats
            ]
    familiarity = bulk["questionFamiliarity"]
    if familiarity:
        for d, rows in by_difficulty.items():
            fresh = partition_by_freshness(rows, lambda c: c["id"], familiarity)
            by_difficulty[d] = prioritize_least_recently_served_contrast_siblings(fresh, lambda c: c["id"], familiarity)
    if review_challenge:
        for d, rows in by_difficulty.items():
            unseen = [c for c in rows if c["id"] not in familiarity]
            pref = []
            for index, cand in enumerate(unseen):
                quality = shrunk_facility(cand.get("facilityIndex"), cand.get("totalAttempts"))
                pref.append((index + (0 if quality is None else 2 * review_challenge * (quality - 0.5)), index, cand))
            pref.sort(key=lambda r: (r[0], r[1]))
            it = iter(pref)
            by_difficulty[d] = [c if c["id"] in familiarity else next(it)[2] for c in rows]
    return _take_questions(
        by_difficulty, endpoint_plan, limit, familiarity, selected_question_ids, selected_question_variant_groups,
        mastered_reentry_counter, review_challenge, challenge_slot_index,
    )


def _take_questions(by_difficulty, endpoint_plan, limit, familiarity, selected_ids, selected_groups, counter, level, slot):
    def take_next(difficulty: str, unseen_only: bool = False) -> dict[str, Any] | None:
        rows = by_difficulty.get(difficulty)
        if not rows:
            return None
        while rows:
            if unseen_only:
                idx = next((k for k, c in enumerate(rows) if c["id"] not in familiarity), -1)
                if idx < 0:
                    break
            else:
                idx = 0
            cand = rows.pop(idx)
            if cand["id"] in selected_ids:
                continue
            key = _suppression_key(cand)
            if key and key in selected_groups:
                continue
            if counter is not None and not take_with_reentry_cap(cand["id"], familiarity, counter):
                continue
            selected_ids.add(cand["id"])
            if key:
                selected_groups.add(key)
            return {"id": cand["id"], "variantGroupId": cand.get("variantGroupId"), "variantType": cand.get("variantType")}
        return None

    selected: list[dict[str, Any]] = []
    for difficulty in endpoint_plan:
        if len(selected) >= limit:
            break
        baseline = next((
            c for c in by_difficulty.get(difficulty, [])
            if c["id"] not in selected_ids and not ((g := _suppression_key(c)) and g in selected_groups)
        ), None)
        shifted = shift_question_difficulty(difficulty, level or 0, slot + len(selected))
        may_shift = baseline is not None and baseline["id"] not in familiarity
        if level == 2:
            picked = take_next(shifted, True) or take_next(shifted)
        elif may_shift and shifted != difficulty:
            picked = take_next(shifted, True) or take_next(difficulty)
        else:
            picked = take_next(difficulty)
        if picked:
            selected.append(picked)
    if len(selected) < limit:
        for difficulty in (["hard"] if level == 2 else ["easy", "medium", "hard"]):
            while len(selected) < limit:
                picked = take_next(difficulty)
                if not picked:
                    break
                selected.append(picked)
            if len(selected) >= limit:
                break
    return selected
