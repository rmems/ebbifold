"""Pure port of candidate-ranking.ts candidate pools used by scheduler lanes.

``bulk`` is the explicit BulkCandidates snapshot: card/question/video rows,
precomputed vector scores (``{conceptId: {itemId: score}}`` in insertion
order), exam-relevance maps and familiarity. Sets arrive as lists and are
normalized by :func:`normalize_bulk`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .concept_thread_ranking import attach_thread_matches, find_thread_matches, promote_thread_matches
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
    return 0 if math.isnan(value) or value == 0 else value


def exam_target_rank_boost(score: float | None, factor: float = EXAM_RELEVANCE_BOOST_FACTOR) -> float:
    if score is None:
        return 0
    if math.isnan(score):
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


class _CardRanker:
    def __init__(self, query: CardPoolQuery, matching_seen: list[dict[str, Any]]) -> None:
        options = query.options
        concept_id, bulk = query.concept_id, query.bulk
        self.concept_id = concept_id
        self.bulk = bulk
        self.current_teaching_week = options.current_teaching_week
        self.exam_target_max_rank_move = options.exam_target_max_rank_move
        self.topic_teaching_weeks = options.topic_teaching_weeks
        self.recent_figure_exposures = options.recent_figure_exposures
        self.figure_now_ms = options.figure_now_ms
        self.topic_recency_days = options.topic_recency_days
        self.vector_rank = build_rank_from_scores(bulk["cardScores"].get(concept_id), 200)
        self.exam_scores = options.exam_target_scores if options.exam_target_scores is not None else bulk["cardExamRelevance"]
        self.history = bulk["cardVariantGroupHistory"]
        failed = bulk["failedStretchCardIds"]
        self.step_down = bool(failed) and any(card["id"] in failed for card in matching_seen)
        penalty_context = options.penalty_context or {}
        self.recent_ids = penalty_context.get("recentCardIds") or set()
        self.recent_failures = penalty_context.get("recentFailureConceptIds") or set()

    @staticmethod
    def importance_boost(importance: Any) -> float:
        return -10 if importance == 3 else -3 if importance == 2 else 0

    def variant_boost(self, card: Mapping[str, Any]) -> float:
        group = card.get("variantGroupId")
        if group is None:
            return 0
        seen = self.history.get(group)
        if not seen:
            return 0
        return 4 if card["id"] in seen else -4

    def ladder(self, card: Mapping[str, Any]) -> float:
        return complexity_ladder_boost(card.get("complexity"), "step-down" if self.step_down else "frontier")

    def pacing(self, card: Mapping[str, Any]) -> float:
        week = item_teaching_week(self.bulk["rotation"], card.get("sourceFile"), card["topics"], self.topic_teaching_weeks)
        return curriculum_pacing_boost(week, self.current_teaching_week) + min(
            curriculum_recency_boost(week, self.current_teaching_week),
            dated_topic_recency_boost(card["topics"], self.topic_recency_days),
        )

    def figure(self, card: Mapping[str, Any]) -> float:
        return figure_cooldown_boost(card.get("imageUrl"), self.recent_figure_exposures, self.figure_now_ms)

    @staticmethod
    def media(card: Mapping[str, Any]) -> float:
        return media_presence_boost(card.get("imageUrl"), card.get("clipId"))

    def soft(self, card: Mapping[str, Any]) -> float:
        return self.pacing(card) + self.figure(card) + self.media(card)

    def fallback_key(self, card: Mapping[str, Any]) -> tuple[float, float, float, float]:
        importance = card.get("importance")
        return (self.variant_boost(card), self.ladder(card), self.soft(card), -(1 if importance is None else importance))

    def recent_neighbor(self, ordered: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
        if not enabled or not self.recent_ids or self.concept_id in self.recent_failures:
            return ordered
        rows = []
        for i, card in enumerate(ordered):
            penalty = recent_neighbor_signal(card.get("similarCards"), self.recent_ids)["penalty"]
            rows.append((i + (penalty / MAX_RECENT_NEIGHBOR_PENALTY) * RECENT_NEIGHBOR_MAX_RANK_SINK, penalty, i, card))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        return [r[3] for r in rows]

    def nudge(self, ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return apply_bounded_exam_target_nudge(
            ordered, lambda c: self.exam_scores.get(c["id"]), self.exam_target_max_rank_move
        )

    def vector_key(self, card: Mapping[str, Any], base: float) -> float:
        importance = card.get("importance")
        return (
            base + self.importance_boost(1 if importance is None else importance) + self.variant_boost(card)
            + self.ladder(card) + self.pacing(card) + self.figure(card) + self.media(card)
        )

    def rank_candidates(self, candidates: list[dict[str, Any]], novelty: bool) -> list[dict[str, Any]]:
        if self.vector_rank is None:
            return self.recent_neighbor(self.nudge(sorted(candidates, key=self.fallback_key)), novelty)
        ranked, unranked = [], []
        for card in candidates:
            base = self.vector_rank.get(card["id"])
            if base is None:
                unranked.append(card)
            else:
                ranked.append((self.vector_key(card, base), card))
        ranked.sort(key=lambda r: r[0])
        unranked.sort(key=self.fallback_key)
        return self.recent_neighbor(self.nudge([c for _, c in ranked]), novelty) + self.recent_neighbor(
            self.nudge(unranked), novelty
        )


def _matching_cards(
    bulk: Mapping[str, Any], topic_set: set[str], selected_card_ids: set[str], review_challenge: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    return matching_unseen, matching_seen


def _apply_card_penalties(
    combined: list[dict[str, Any]], penalty_context: Mapping[str, Any], bulk: Mapping[str, Any]
) -> list[dict[str, Any]]:
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
    return [r[2] for r in rows]


def _apply_review_challenge_card_order(
    combined: list[dict[str, Any]], matching_seen: list[dict[str, Any]], bulk: Mapping[str, Any], review_challenge: int
) -> list[dict[str, Any]]:
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
    return [next(it)[3] if c["id"] in unseen_by_id else c for c in combined]


def _unique_limited(cards: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for card in cards:
        if len(out) >= max_candidates:
            break
        if card["id"] in seen_ids:
            continue
        seen_ids.add(card["id"])
        out.append(card)
    return out


@dataclass(frozen=True)
class CardRankingOptions:
    selected_card_ids: set[str] = field(default_factory=set)
    penalty_context: Mapping[str, Any] | None = None
    concept_recall: float | None = None
    current_teaching_week: int | None = None
    exam_target_scores: Mapping[str, float] | None = None
    exam_target_max_rank_move: float = EXAM_RELEVANCE_BOOST_FACTOR
    topic_teaching_weeks: Mapping[str, int] | None = None
    recent_figure_exposures: Mapping[str, Any] | None = None
    figure_now_ms: float = 0
    review_challenge: int = 0
    topic_recency_days: Mapping[str, float] | None = None


@dataclass(frozen=True)
class CardPoolQuery:
    concept_id: str
    concept: Mapping[str, Any]
    max_candidates: int
    bulk: Mapping[str, Any]
    options: CardRankingOptions = field(default_factory=CardRankingOptions)


def get_cards_from_bulk(query: CardPoolQuery) -> list[dict[str, Any]]:
    concept, max_candidates = query.concept, query.max_candidates
    bulk, options = query.bulk, query.options
    selected_card_ids = options.selected_card_ids
    penalty_context = options.penalty_context
    review_challenge = options.review_challenge
    topic_set = set(expand_topic_set(concept["topics"]))
    if not topic_set:
        return []
    matching_unseen, matching_seen = _matching_cards(bulk, topic_set, selected_card_ids, review_challenge)
    ranker = _CardRanker(query, matching_seen)
    passed_ids = bulk["passedScaffoldCardIds"]
    passed = [c for c in matching_seen if c["id"] in passed_ids] if passed_ids else []

    def is_block(card: Mapping[str, Any]) -> bool:
        return (2 if card.get("complexity") is None else card["complexity"]) <= 2

    if ranker.step_down:
        both = matching_unseen + matching_seen
        combined = ranker.rank_candidates([c for c in both if is_block(c)], False) + ranker.rank_candidates(
            [c for c in both if not is_block(c)], False
        )
    else:
        passed_ref = {id(c) for c in passed}
        frontier = [c for c in matching_unseen if not is_block(c)] + [c for c in matching_seen if not is_block(c)]
        combined = (
            ranker.rank_candidates(frontier, True)
            + ranker.rank_candidates([c for c in matching_unseen if is_block(c)], True)
            + ranker.rank_candidates([c for c in matching_seen if is_block(c) and id(c) not in passed_ref], False)
            + ranker.rank_candidates(passed, False)
        )
    if penalty_context is not None:
        combined = _apply_card_penalties(combined, penalty_context, bulk)
    if review_challenge != 0 and not ranker.step_down:
        combined = _apply_review_challenge_card_order(combined, matching_seen, bulk, review_challenge)
    return _unique_limited(combined, max_candidates)

@dataclass(frozen=True)
class VideoPoolQuery:
    concept_id: str
    concept: Mapping[str, Any]
    limit: int
    bulk: Mapping[str, Any]
    exclude_video_ids: set[str] = field(default_factory=set)


def _linked_videos(bulk: Mapping[str, Any], concept_id: str, exclude_video_ids: set[str]) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    for link in bulk["videoConceptLinks"]:
        if link["conceptId"] != concept_id or link["videoId"] in exclude_video_ids:
            continue
        video = bulk["videoMap"].get(link["videoId"])
        if video:
            linked.append(video)
    return linked


def _topic_matched_videos(query: VideoPoolQuery, linked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not expand_topic_set(query.concept["topics"]):
        return []
    linked_ids = {v["id"] for v in linked}
    topic_matched: list[dict[str, Any]] = []
    for video in query.bulk["rotationVideos"]:
        if video["id"] in query.exclude_video_ids or video["id"] in linked_ids:
            continue
        if query.concept_id in video["id"]:
            topic_matched.append(video)
    return topic_matched


def _ordered_video_candidates(query: VideoPoolQuery) -> list[dict[str, Any]]:
    linked = _linked_videos(query.bulk, query.concept_id, query.exclude_video_ids)
    topic_matched = _topic_matched_videos(query, linked)
    vector_rank = build_rank_from_scores(query.bulk["videoScores"].get(query.concept_id), 100)

    def by_vector(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if vector_rank is None or len(videos) <= 1:
            return videos
        ranked = sorted((r for r in ((vector_rank.get(v["id"]), v) for v in videos) if r[0] is not None), key=lambda r: r[0])
        return [v for _, v in ranked] + [v for v in videos if vector_rank.get(v["id"]) is None]

    return by_vector(linked) + by_vector(topic_matched)


def get_videos_from_bulk(query: VideoPoolQuery) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for video in _ordered_video_candidates(query):
        if len(out) >= query.limit:
            break
        if video["id"] in seen:
            continue
        seen.add(video["id"])
        out.append(video)
    return out


@dataclass(frozen=True)
class QuestionRankingOptions:
    selected_question_ids: set[str] = field(default_factory=set)
    selected_question_variant_groups: set[str] = field(default_factory=set)
    difficulty_plan: list[str] | None = None
    recent_formats_by_concept: Mapping[str, Any] | None = None
    concept_thread_anchors: list[Any] | None = None
    exam_target_scores: Mapping[str, float] | None = None
    exam_target_max_rank_move: float | None = None
    mastered_reentry_counter: dict[str, int] | None = None
    current_teaching_week: int | None = None
    topic_teaching_weeks: Mapping[str, int] | None = None
    review_challenge: int | None = None
    topic_recency_days: Mapping[str, float] | None = None
    review_challenge_gap: bool | None = None
    challenge_slot_index: int = 0


@dataclass(frozen=True)
class QuestionPoolQuery:
    concept_id: str
    concept: Mapping[str, Any]
    limit: int
    bulk: Mapping[str, Any]
    options: QuestionRankingOptions = field(default_factory=QuestionRankingOptions)


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


@dataclass(frozen=True)
class QuestionOrderContext:
    bulk: Mapping[str, Any]
    vector_rank: dict[str, int] | None
    exam_scores: Mapping[str, float]
    max_rank_move: float
    current_teaching_week: int | None
    topic_teaching_weeks: Mapping[str, int] | None
    topic_recency_days: Mapping[str, float] | None


def _prepared_question_bulk(bulk: Mapping[str, Any], anchors: list[Any] | None) -> Mapping[str, Any]:
    if not anchors or bulk.get("conceptThreadMatcher"):
        return bulk
    now_ms = bulk.get("conceptThreadNowMs")
    if now_ms is None:
        raise UnsupportedSchedulerInput("raw concept-thread anchors require a prepared matcher or conceptThreadNowMs")
    from .concept_thread import prepare_concept_thread_matcher

    return {**bulk, "conceptThreadMatcher": prepare_concept_thread_matcher(anchors, now_ms)}


def _question_order_context(query: QuestionPoolQuery, bulk: Mapping[str, Any]) -> QuestionOrderContext:
    options = query.options
    return QuestionOrderContext(
        bulk=bulk,
        vector_rank=build_rank_from_scores(bulk["questionScores"].get(query.concept_id), 200),
        exam_scores=options.exam_target_scores if options.exam_target_scores is not None else bulk["questionExamRelevance"],
        max_rank_move=EXAM_RELEVANCE_BOOST_FACTOR if options.exam_target_max_rank_move is None else options.exam_target_max_rank_move,
        current_teaching_week=options.current_teaching_week,
        topic_teaching_weeks=options.topic_teaching_weeks,
        topic_recency_days=options.topic_recency_days,
    )


def _question_pacing(context: QuestionOrderContext, row: Mapping[str, Any]) -> float:
    week = item_teaching_week(context.bulk["rotation"], None, row["topics"], context.topic_teaching_weeks)
    return curriculum_pacing_boost(week, context.current_teaching_week) + min(
        curriculum_recency_boost(week, context.current_teaching_week),
        dated_topic_recency_boost(row["topics"], context.topic_recency_days),
    )


def _nudge_questions(context: QuestionOrderContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return apply_bounded_exam_target_nudge(rows, lambda candidate: context.exam_scores.get(candidate["id"]), context.max_rank_move)


def _questions_by_vector(context: QuestionOrderContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pacing = sorted(rows, key=lambda row: _question_pacing(context, row))
    if context.vector_rank is None or len(rows) <= 1:
        return _nudge_questions(context, by_pacing)
    ranked = [
        (context.vector_rank[row["id"]] + _question_pacing(context, row), row)
        for row in rows if row["id"] in context.vector_rank
    ]
    ranked.sort(key=lambda pair: pair[0])
    unranked = sorted((row for row in rows if row["id"] not in context.vector_rank), key=lambda row: _question_pacing(context, row))
    return _nudge_questions(context, [row for _, row in ranked]) + _nudge_questions(context, unranked)


def _ordered_question_rows(context: QuestionOrderContext, pools: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    primary, linked, topic_matched = pools
    ordered = _questions_by_vector(context, primary) + _questions_by_vector(context, linked) + _questions_by_vector(context, topic_matched)
    return [row for row in ordered if row.get("source") == "bank"] + [row for row in ordered if row.get("source") != "bank"]


def _effective_difficulty(row: Mapping[str, Any], review_challenge: int | None) -> str:
    if review_challenge == 2 and row.get("difficulty") == "hard":
        return "hard"
    return get_effective_question_difficulty(row.get("difficulty"), row.get("facilityIndex"), row.get("totalAttempts"))


def _group_questions_by_difficulty(rows: list[dict[str, Any]], review_challenge: int | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {difficulty: [] for difficulty in QUESTION_DIFFICULTIES}
    for row in rows:
        grouped[_effective_difficulty(row, review_challenge)].append(row)
    return grouped


def _format_seen(row: Mapping[str, Any], seen_formats: Any) -> bool:
    return bool(row.get("format") and row["format"] in seen_formats)


def _deprioritize_seen_formats(grouped: dict[str, list[dict[str, Any]]], seen_formats: Any) -> None:
    if not seen_formats:
        return
    for difficulty, rows in grouped.items():
        grouped[difficulty] = [row for row in rows if not _format_seen(row, seen_formats)] + [
            row for row in rows if _format_seen(row, seen_formats)
        ]


def _apply_familiarity_order(grouped: dict[str, list[dict[str, Any]]], familiarity: Mapping[str, Any]) -> None:
    if not familiarity:
        return
    for difficulty, rows in grouped.items():
        fresh = partition_by_freshness(rows, lambda row: row["id"], familiarity)
        grouped[difficulty] = prioritize_least_recently_served_contrast_siblings(fresh, lambda row: row["id"], familiarity)


def _challenge_quality_rank(index: int, candidate: Mapping[str, Any], review_challenge: int) -> tuple[float, int, Mapping[str, Any]]:
    quality = shrunk_facility(candidate.get("facilityIndex"), candidate.get("totalAttempts"))
    adjustment = 0 if quality is None else 2 * review_challenge * (quality - 0.5)
    return index + adjustment, index, candidate


def _apply_challenge_quality_order(grouped: dict[str, list[dict[str, Any]]], familiarity: Mapping[str, Any], review_challenge: int | None) -> None:
    if not review_challenge:
        return
    for difficulty, rows in grouped.items():
        unseen = [row for row in rows if row["id"] not in familiarity]
        preferred = sorted(
            (_challenge_quality_rank(index, row, review_challenge) for index, row in enumerate(unseen)),
            key=lambda row: (row[0], row[1]),
        )
        replacements = iter(preferred)
        grouped[difficulty] = [row if row["id"] in familiarity else next(replacements)[2] for row in rows]


def get_questions_from_bulk(query: QuestionPoolQuery) -> list[dict[str, Any]]:
    concept_id, concept, limit = query.concept_id, query.concept, query.limit
    bulk, options = query.bulk, query.options
    selected_question_ids = options.selected_question_ids
    selected_question_variant_groups = options.selected_question_variant_groups
    difficulty_plan = options.difficulty_plan
    recent_formats_by_concept = options.recent_formats_by_concept
    concept_thread_anchors = options.concept_thread_anchors
    mastered_reentry_counter = options.mastered_reentry_counter
    review_challenge = options.review_challenge
    review_challenge_gap = options.review_challenge_gap
    challenge_slot_index = options.challenge_slot_index
    bulk = _prepared_question_bulk(bulk, concept_thread_anchors)
    if not question_eligible_for_review_learning(
        review_challenge, demonstrated_gap=review_challenge_gap, difficulty="hard" if review_challenge == 2 else None
    ):
        return []
    plan = build_difficulty_plan(difficulty_plan if difficulty_plan is not None else ["medium"], limit)
    endpoint_plan = build_difficulty_plan(["hard"], limit) if review_challenge == 2 else plan
    primary, linked, topic_matched = _question_pool(concept_id, concept, bulk, selected_question_ids)
    order_context = _question_order_context(query, bulk)
    ordered = _ordered_question_rows(order_context, (primary, linked, topic_matched))
    by_difficulty = _group_questions_by_difficulty(ordered, review_challenge)

    seen_formats = (recent_formats_by_concept or {}).get(concept_id)
    _deprioritize_seen_formats(by_difficulty, seen_formats)
    thread_matches = find_thread_matches(by_difficulty, bulk.get("conceptThreadMatcher"), concept_id)

    familiarity = bulk["questionFamiliarity"]
    _apply_familiarity_order(by_difficulty, familiarity)
    promote_thread_matches(by_difficulty, thread_matches, familiarity)
    _apply_challenge_quality_order(by_difficulty, familiarity, review_challenge)
    take_request = QuestionTakeRequest(
        by_difficulty=by_difficulty, endpoint_plan=endpoint_plan, limit=limit, familiarity=familiarity,
        selected_ids=selected_question_ids, selected_groups=selected_question_variant_groups,
        counter=mastered_reentry_counter, level=review_challenge, slot=challenge_slot_index,
        thread_matches=thread_matches,
    )
    selected = _take_questions(take_request)
    return attach_thread_matches(selected, thread_matches)


@dataclass(frozen=True)
class QuestionTakeRequest:
    by_difficulty: dict[str, list[dict[str, Any]]]
    endpoint_plan: list[str]
    limit: int
    familiarity: Mapping[str, Any]
    selected_ids: set[str]
    selected_groups: set[str]
    counter: dict[str, int] | None
    level: int | None
    slot: int
    thread_matches: Mapping[str, Any] | None = None


class _QuestionTaker:
    def __init__(self, request: QuestionTakeRequest) -> None:
        self.request = request
        self.selected: list[dict[str, Any]] = []

    def rows(self, difficulty: str) -> list[dict[str, Any]]:
        return self.request.by_difficulty.get(difficulty) or []

    @staticmethod
    def _candidate_index(rows: list[dict[str, Any]], familiarity: Mapping[str, Any], unseen_only: bool) -> int:
        if not unseen_only:
            return 0
        return next((index for index, candidate in enumerate(rows) if candidate["id"] not in familiarity), -1)

    def _blocked(self, candidate: Mapping[str, Any]) -> bool:
        if candidate["id"] in self.request.selected_ids:
            return True
        key = _suppression_key(candidate)
        if key and key in self.request.selected_groups:
            return True
        counter = self.request.counter
        return counter is not None and not take_with_reentry_cap(candidate["id"], self.request.familiarity, counter)

    def _mark_selected(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        self.request.selected_ids.add(candidate["id"])
        key = _suppression_key(candidate)
        if key:
            self.request.selected_groups.add(key)
        return {
            "id": candidate["id"], "variantGroupId": candidate.get("variantGroupId"),
            "variantType": candidate.get("variantType"),
        }

    def take_next(self, difficulty: str, unseen_only: bool = False) -> dict[str, Any] | None:
        rows = self.rows(difficulty)
        while rows:
            index = self._candidate_index(rows, self.request.familiarity, unseen_only)
            if index < 0:
                break
            candidate = rows.pop(index)
            if self._blocked(candidate):
                continue
            return self._mark_selected(candidate)
        return None

    def baseline(self, difficulty: str) -> dict[str, Any] | None:
        return next((candidate for candidate in self.rows(difficulty) if not self._already_reserved(candidate)), None)

    def _already_reserved(self, candidate: Mapping[str, Any]) -> bool:
        key = _suppression_key(candidate)
        return candidate["id"] in self.request.selected_ids or bool(key and key in self.request.selected_groups)

    def _may_shift(self, baseline: dict[str, Any] | None) -> bool:
        return (
            baseline is not None and baseline["id"] not in self.request.familiarity
            and baseline["id"] not in (self.request.thread_matches or {})
        )

    def _pick_planned(self, difficulty: str) -> dict[str, Any] | None:
        level = self.request.level
        shifted = shift_question_difficulty(difficulty, level or 0, self.request.slot + len(self.selected))
        if level == 2:
            return self.take_next(shifted, True) or self.take_next(shifted)
        if self._may_shift(self.baseline(difficulty)) and shifted != difficulty:
            return self.take_next(shifted, True) or self.take_next(difficulty)
        return self.take_next(difficulty)

    def take_planned(self) -> None:
        for difficulty in self.request.endpoint_plan:
            if len(self.selected) >= self.request.limit:
                break
            picked = self._pick_planned(difficulty)
            if picked:
                self.selected.append(picked)

    def fallback_difficulties(self) -> list[str]:
        return ["hard"] if self.request.level == 2 else ["easy", "medium", "hard"]

    def take_fallback(self) -> None:
        for difficulty in self.fallback_difficulties():
            while len(self.selected) < self.request.limit:
                picked = self.take_next(difficulty)
                if not picked:
                    break
                self.selected.append(picked)
            if len(self.selected) >= self.request.limit:
                break

    def take(self) -> list[dict[str, Any]]:
        self.take_planned()
        if len(self.selected) < self.request.limit:
            self.take_fallback()
        return self.selected


def _take_questions(request: QuestionTakeRequest):
    return _QuestionTaker(request).take()
