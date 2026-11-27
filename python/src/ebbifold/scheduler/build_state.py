"""Mutable selection bookkeeping (unified-scheduler-phase-state.ts).

Rejections are recorded as explicit reason codes so parity failures can be
explained; the TypeScript class returns ``false``/``null`` for the same cases.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .errors import UnsupportedSchedulerInput
from .scoring import TEACHING_ARC_TARGET_ITEMS, classify_teaching_state

SUPPRESSIBLE_VARIANT_TYPES = frozenset({"near-duplicate", "contrast-set"})
DEFAULT_TEACHING_CAP = 3


def parse_similar_links(value: Any) -> list[tuple[str, float]]:
    if not isinstance(value, list):
        return []
    links = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        card_id = entry.get("cardId")
        similarity = entry.get("similarity")
        if not isinstance(card_id, str) or not card_id:
            continue
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            continue
        links.append((card_id, similarity))
    return links


def question_suppression_key(item: Mapping[str, Any]) -> str | None:
    group, variant_type = item.get("variantGroupId"), item.get("variantType")
    if not group or not variant_type:
        return None
    return group if variant_type in SUPPRESSIBLE_VARIANT_TYPES else None


class SessionBuildState:
    def __init__(
        self,
        *,
        size: int,
        min_first_sight_items: int,
        max_new_cards: float,
        max_cross_source_items: int,
        unseen_card_ids: Iterable[str],
        cross_source_card_ids: Iterable[str] = (),
        cross_source_question_ids: Iterable[str] = (),
        question_familiarity: Iterable[str] = (),
        excluded_card_ids: Iterable[str] = (),
        excluded_question_ids: Iterable[str] = (),
        excluded_video_ids: Iterable[str] = (),
        weak_concepts: Sequence[Any] = (),
        concept_thread_anchors: Sequence[Any] = (),
        now_ms: float = 0,
    ) -> None:
        from .concept_thread import prepare_concept_thread_matcher

        self.size = size
        self.min_first_sight_items = min_first_sight_items
        self.max_new_cards = max_new_cards
        self.max_cross_source_items = max_cross_source_items
        self.unseen_card_ids = set(unseen_card_ids)
        self.cross_source_card_ids = set(cross_source_card_ids)
        self.cross_source_question_ids = set(cross_source_question_ids)
        self.question_familiarity = set(question_familiarity)
        self.selected_items: list[dict[str, Any]] = []
        self.selected_concept_ids: set[str] = set()
        self.selected_card_ids = set(excluded_card_ids)
        self.selected_question_ids = set(excluded_question_ids)
        self.selected_video_ids = set(excluded_video_ids)
        self.selected_question_variant_groups: set[str] = set()
        self.selected_card_variant_groups: set[str] = set()
        self.cards_per_concept: dict[str, int] = {}
        self.questions_per_concept: dict[str, int] = {}
        self.videos_per_concept: dict[str, int] = {}
        self.cluster_counts: dict[str, int] = {}
        self.similarity_to_selected: dict[str, float] = {}
        self.challenge_recall_at_selection: dict[str, float] = {}
        self.concept_teaching_cap: dict[str, int] = {}
        self.selected_card_count = 0
        self.selected_question_count = 0
        self.new_card_count = 0
        self.selected_cross_source_items = 0
        self.first_sight_selected = 0
        self.enforce_first_sight_reservation = min_first_sight_items > 0
        self.deferred_for_novelty: dict[str, dict[str, Any]] = {}
        self.concept_thread_followup_selected = False
        self.prepared_concept_thread_matcher = prepare_concept_thread_matcher(list(concept_thread_anchors), now_ms)
        self.trace: list[dict[str, Any]] = []
        for cs in weak_concepts:
            state = classify_teaching_state(cs.exposureCount, cs.recallOnExamDay, cs.confidence)
            self.concept_teaching_cap[cs.conceptId] = TEACHING_ARC_TARGET_ITEMS[state]

    def available_concept_thread_matcher(self):
        if self.concept_thread_followup_selected:
            return None
        return self.prepared_concept_thread_matcher

    def is_first_sight_item(self, item: Mapping[str, Any]) -> bool:
        if item["type"] == "card":
            return item["id"] in self.unseen_card_ids
        return item["type"] == "question" and item["id"] not in self.question_familiarity

    def _is_cross_source_item(self, item: Mapping[str, Any]) -> bool:
        if item["type"] == "card":
            return item["id"] in self.cross_source_card_ids
        return item["type"] == "question" and item["id"] in self.cross_source_question_ids

    def record_applied_challenge_recall(self, card_id: str, recall: float | None) -> None:
        if recall is not None and math.isfinite(recall):
            self.challenge_recall_at_selection[card_id] = recall

    def teaching_cap_for(self, concept_id: str) -> int:
        return self.concept_teaching_cap.get(concept_id, DEFAULT_TEACHING_CAP)

    def concept_items_so_far(self, concept_id: str) -> int:
        return self.cards_per_concept.get(concept_id, 0) + self.questions_per_concept.get(concept_id, 0)

    def _reject(self, kind: str, item_id: str, reason: str) -> None:
        self.trace.append({"op": kind, "id": item_id, "result": reason})

    def add_item(self, item: dict[str, Any], *, bypass_novelty_reservation: bool = False) -> bool:
        if len(self.selected_items) >= self.size:
            self._reject("add", item["id"], "session_full")
            return False
        first_sight = self.is_first_sight_item(item)
        seats_remaining = self.size - len(self.selected_items)
        quota_remaining = max(0, self.min_first_sight_items - self.first_sight_selected)
        if (
            not bypass_novelty_reservation
            and not first_sight
            and self.enforce_first_sight_reservation
            and seats_remaining <= quota_remaining
        ):
            self.deferred_for_novelty[f"{item['type']}:{item['id']}"] = item
            self._reject("add", item["id"], "deferred_for_novelty")
            return False
        concept_id = item["conceptId"]
        synthetic = concept_id == "_unattached" or concept_id.startswith("rotation:")
        cross_source = self._is_cross_source_item(item)
        if cross_source and self.selected_cross_source_items >= self.max_cross_source_items:
            self._reject("add", item["id"], "cross_source_cap")
            return False
        if (
            item["type"] != "video"
            and not synthetic
            and self.concept_items_so_far(concept_id) >= self.teaching_cap_for(concept_id)
        ):
            self._reject("add", item["id"], "teaching_cap")
            return False

        self.selected_items.append(item)
        self.trace.append({"op": "add", "id": item["id"], "result": "added"})
        if first_sight:
            self.first_sight_selected += 1
        self.selected_concept_ids.add(concept_id)
        if cross_source:
            self.selected_cross_source_items += 1
        if item["type"] == "card":
            self.selected_card_count += 1
            if item["id"] in self.unseen_card_ids:
                self.new_card_count += 1
            self.cards_per_concept[concept_id] = self.cards_per_concept.get(concept_id, 0) + 1
            self.selected_card_ids.add(item["id"])
            if item.get("variantGroupId"):
                self.selected_card_variant_groups.add(item["variantGroupId"])
            return True
        if item["type"] == "video":
            self.videos_per_concept[concept_id] = self.videos_per_concept.get(concept_id, 0) + 1
            self.selected_video_ids.add(item["id"])
            return True
        self.selected_question_count += 1
        self.questions_per_concept[concept_id] = self.questions_per_concept.get(concept_id, 0) + 1
        self.selected_question_ids.add(item["id"])
        key = question_suppression_key(item)
        if key:
            self.selected_question_variant_groups.add(key)
        return True

    def _would_cause_interference(self, candidate: Mapping[str, Any], threshold: float) -> bool:
        known = self.similarity_to_selected.get(candidate["id"])
        if known is not None and known > threshold:
            return True
        return any(
            card_id in self.selected_card_ids and similarity > threshold
            for card_id, similarity in parse_similar_links(candidate.get("similarCards"))
        )

    def record_card_neighborhood(self, candidate: Mapping[str, Any]) -> None:
        for card_id, similarity in parse_similar_links(candidate.get("similarCards")):
            if similarity > self.similarity_to_selected.get(card_id, 0):
                self.similarity_to_selected[card_id] = similarity

    def record_cluster(self, cluster_id: str | None) -> None:
        if cluster_id:
            self.cluster_counts[cluster_id] = self.cluster_counts.get(cluster_id, 0) + 1

    def pick_card_candidate(
        self, candidates: Sequence[Mapping[str, Any]], *, similarity_threshold: float, max_per_cluster: int
    ) -> Mapping[str, Any] | None:
        for c in candidates:
            cid = c["id"]
            if cid in self.selected_card_ids:
                reason = "already_selected"
            elif cid in self.cross_source_card_ids and self.selected_cross_source_items >= self.max_cross_source_items:
                reason = "cross_source_cap"
            elif c.get("variantGroupId") and c["variantGroupId"] in self.selected_card_variant_groups:
                reason = "variant_group"
            elif cid in self.unseen_card_ids and self.new_card_count >= self.max_new_cards:
                reason = "new_card_budget"
            elif c.get("clusterId") and self.cluster_counts.get(c["clusterId"], 0) >= max_per_cluster:
                reason = "cluster_cap"
            elif self._would_cause_interference(c, similarity_threshold):
                reason = "interference"
            else:
                self.trace.append({"op": "pick", "id": cid, "result": "picked"})
                return c
            self._reject("pick", cid, reason)
        return None

    def release_first_sight_reservation(self) -> None:
        """Phase 8d (releaseFirstSightReservation): backfill with deferred review work."""
        if len(self.selected_items) < self.size and self.deferred_for_novelty:
            self.enforce_first_sight_reservation = False
            for item in list(self.deferred_for_novelty.values()):
                if len(self.selected_items) >= self.size:
                    break
                self.add_item(item)
