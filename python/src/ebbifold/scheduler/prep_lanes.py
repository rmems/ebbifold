"""Session-preparation reservation lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .build_state import SessionBuildState


@dataclass(frozen=True)
class PrepQueueRequest:
    keys: list[str]
    sessions: list[dict[str, Any]]
    weak: set[str]
    unseen: set[str]
    share: float


@dataclass(frozen=True)
class PrepAdditionRecord:
    build: SessionBuildState
    card: dict[str, Any] | None
    question: dict[str, Any] | None
    reentry: dict[str, int]
    added: bool


@dataclass(frozen=True)
class PrepReservationContext:
    inputs: dict[str, Any]
    build: SessionBuildState
    card_by_key: dict[str, dict[str, Any]]
    question_by_key: dict[str, dict[str, Any]]


def _append_at_index(out: list[str], keys: list[str], index: int) -> None:
    if index < len(keys):
        out.append(keys[index])


def _priority_lanes(weak_keys: list[str], unseen_keys: list[str], share: float) -> tuple[list[str], list[str]]:
    if share < 0.25:
        return unseen_keys, weak_keys
    return weak_keys, unseen_keys


def _outside_priority_keys(keys: list[str], weak: set[str], unseen: set[str]) -> list[str]:
    excluded = weak | unseen
    return [key for key in keys if key not in excluded]


def _interleave_prep_keys(keys: list[str], weak: set[str], unseen: set[str], share: float) -> list[str]:
    weak_keys = [key for key in keys if key in weak]
    unseen_keys = [key for key in keys if key in unseen]
    first, second = _priority_lanes(weak_keys, unseen_keys, share)
    out = []
    for index in range(max(len(weak_keys), len(unseen_keys))):
        _append_at_index(out, first, index)
        _append_at_index(out, second, index)
    return out + _outside_priority_keys(keys, weak, unseen)


def _prep_session_queues(request: PrepQueueRequest) -> list[dict[str, Any]]:
    queues = []
    for session in request.sessions:
        ids = set(session.get("itemIds") or [])
        session_keys = [key for key in request.keys if key in ids]
        queues.append({"share": session["share"], "taken": 0,
                       "keys": _interleave_prep_keys(session_keys, request.weak, request.unseen, request.share)})
    return queues


def _drop_used_prep_keys(queues: list[dict[str, Any]], used: set[str]) -> None:
    for queue in queues:
        while queue["keys"]:
            if queue["keys"][0] not in used:
                break
            queue["keys"].pop(0)


def _next_prep_key(queues: list[dict[str, Any]], used: set[str]) -> str | None:
    _drop_used_prep_keys(queues, used)
    available = [queue for queue in queues if queue["keys"]]
    if not available:
        return None
    queue = min(available, key=lambda item: (item["taken"] + 1) / item["share"])
    queue["taken"] += 1
    return queue["keys"].pop(0)


def _planned_list(plan: dict[str, Any], key: str) -> list[Any]:
    value = plan.get(key)
    if value is None:
        return []
    return value


def _numeric_share(value: Any) -> float:
    if value is None:
        return 0
    return value


def _positive_sessions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [session for session in _planned_list(plan, "sessions") if _numeric_share(session.get("share")) > 0]


def order_prep_keys(keys: list[str], plan: dict[str, Any]) -> list[str]:
    weak = set(_planned_list(plan, "weakItemIds"))
    unseen = set(_planned_list(plan, "unseenItemIds"))
    share = _numeric_share(plan.get("share"))
    sessions = _positive_sessions(plan)
    if len(sessions) < 2:
        return _interleave_prep_keys(keys, weak, unseen, share)
    queues = _prep_session_queues(PrepQueueRequest(keys, sessions, weak, unseen, share))
    used, out = set(), []
    while True:
        key = _next_prep_key(queues, used)
        if key is None:
            break
        used.add(key)
        out.append(key)
    return out + _outside_priority_keys(keys, used, set())


def _concept_topics_match(concept: dict[str, Any], topic_set: set[str]) -> bool:
    return any(topic in topic_set for topic in concept.get("topics") or [])


def _matching_prep_concept(inputs: dict[str, Any], topic_set: set[str]) -> dict[str, Any] | None:
    return next((concept for concept in inputs["concepts"] if _concept_topics_match(concept, topic_set)), None)


def _prep_card_concept_id(topics: list[str], inputs: dict[str, Any]) -> str:
    match = _matching_prep_concept(inputs, set(topics))
    if match:
        return match["id"]
    return "_unattached"


def _question_concept_links(inputs: dict[str, Any], question_id: str) -> list[dict[str, Any]]:
    return [link for link in inputs["bulk"]["questionConceptLinks"] if link["questionId"] == question_id]


def _primary_question_link(links: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((link for link in links if link.get("isPrimary")), None)


def _first_question_link(links: list[dict[str, Any]]) -> dict[str, Any] | None:
    if links:
        return links[0]
    return None


def _prep_question_concept_id(question_id: str, inputs: dict[str, Any]) -> str:
    links = _question_concept_links(inputs, question_id)
    link = _primary_question_link(links) or _first_question_link(links)
    if link:
        return link["conceptId"]
    return "_unattached"


def _wanted_prep_cards(inputs: dict[str, Any], wanted: set[str]) -> list[dict[str, Any]]:
    return [card for card in inputs["bulk"]["unseenCards"] + inputs["bulk"]["seenCards"] if f"card:{card['id']}" in wanted]


def _wanted_prep_questions(inputs: dict[str, Any], wanted: set[str]) -> list[dict[str, Any]]:
    return [question for question in inputs["bulk"]["rotationQuestions"] if f"question:{question['id']}" in wanted]


def _prep_maps(inputs: dict[str, Any], item_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    wanted = set(item_ids)
    cards = _wanted_prep_cards(inputs, wanted)
    questions = _wanted_prep_questions(inputs, wanted)
    return ({f"card:{card['id']}": card for card in cards},
            {f"question:{question['id']}": question for question in questions})


def _suppression_group_blocked(build: SessionBuildState, suppression: str | None) -> bool:
    if not suppression:
        return False
    return suppression in build.selected_question_variant_groups


def _prep_question_allowed(build: SessionBuildState, question: dict[str, Any], familiarity: dict[str, Any], reentry: dict[str, int]) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    suppression = question_suppression_key(question)
    if question["id"] in build.selected_question_ids:
        return False
    if _suppression_group_blocked(build, suppression):
        return False
    return take_with_reentry_cap(question["id"], familiarity, reentry)


def _prep_concept_id(inputs: dict[str, Any], card: dict[str, Any] | None, question: dict[str, Any] | None) -> str:
    if card:
        return _prep_card_concept_id(card.get("topics") or [], inputs)
    return _prep_question_concept_id(question["id"], inputs)


def _prep_source(card: dict[str, Any] | None, question: dict[str, Any] | None) -> dict[str, Any]:
    if card:
        return card
    return question


def _prep_concept_name(concept: dict[str, Any] | None) -> str:
    if concept:
        return concept.get("name") or "Class preparation"
    return "Class preparation"


def _prep_item_type(card: dict[str, Any] | None) -> str:
    if card:
        return "card"
    return "question"


def _prep_item(inputs: dict[str, Any], card: dict[str, Any] | None, question: dict[str, Any] | None) -> dict[str, Any]:
    source = _prep_source(card, question)
    concept_id = _prep_concept_id(inputs, card, question)
    concept = inputs["conceptMap"].get(concept_id)
    item = {
        "type": _prep_item_type(card), "id": source["id"], "conceptId": concept_id,
        "conceptName": _prep_concept_name(concept), "priority": 1,
        "interventionReason": "pre_teach", "topics": source.get("topics") or [],
        "variantGroupId": source.get("variantGroupId"), "variantType": source.get("variantType"),
        "rotation": source.get("rotation"), "challengePolicyApplied": False,
    }
    if card:
        item.update({"variantIndex": card.get("variantIndex"), "complexity": card.get("complexity"), "clusterId": card.get("clusterId")})
    return item


def _prep_card_accepted(inputs: dict[str, Any], build: SessionBuildState, card: dict[str, Any] | None) -> bool:
    if not card:
        return True
    picked = build.pick_card_candidate([card], similarity_threshold=inputs["interferenceThreshold"], max_per_cluster=3)
    return bool(picked)


def _prep_question_accepted(inputs: dict[str, Any], build: SessionBuildState,
                            question: dict[str, Any] | None, reentry: dict[str, int]) -> bool:
    if not question:
        return True
    return _prep_question_allowed(build, question, inputs["bulk"]["questionFamiliarity"], reentry)


def _record_prep_addition(record: PrepAdditionRecord) -> None:
    if not record.added:
        return
    if record.card:
        record.build.record_cluster(record.card.get("clusterId"))
        record.build.record_card_neighborhood(record.card)
    if record.question:
        record.build.mastered_reentry_counter["masteredServed"] = record.reentry["masteredServed"]


def _add_prep_candidate(inputs: dict[str, Any], build: SessionBuildState,
                        card: dict[str, Any] | None, question: dict[str, Any] | None) -> bool:
    if not _prep_card_accepted(inputs, build, card):
        return False
    reentry = dict(build.mastered_reentry_counter)
    if not _prep_question_accepted(inputs, build, question, reentry):
        return False
    added = build.add_item(_prep_item(inputs, card, question))
    _record_prep_addition(PrepAdditionRecord(build, card, question, reentry, added))
    return added


def _valid_prep_share(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return value > 0


def _prep_seat_count(size: int, selected_count: int, share: float) -> int:
    discretionary = max(0, size - selected_count)
    return min(discretionary, int(discretionary * min(0.5, max(0.15, share))))


def _has_prep_candidate(card: dict[str, Any] | None, question: dict[str, Any] | None) -> bool:
    if card:
        return True
    return bool(question)


def _reserve_prep_keys(context: PrepReservationContext, ordered_ids: list[str], stop_at: int) -> None:
    for key in ordered_ids:
        if len(context.build.selected_items) >= stop_at:
            break
        card = context.card_by_key.get(key)
        question = context.question_by_key.get(key)
        if _has_prep_candidate(card, question):
            _add_prep_candidate(context.inputs, context.build, card, question)


def _configured_prep_item_ids(prep: dict[str, Any] | None) -> list[str]:
    if not prep:
        return []
    item_ids = prep.get("itemIds")
    if item_ids is None:
        return []
    return list(item_ids)


def _configured_prep_share(prep: dict[str, Any] | None) -> Any:
    if not prep:
        return None
    return prep.get("share")


def _has_prep_source(key: str, card_by_key: dict[str, dict[str, Any]], question_by_key: dict[str, dict[str, Any]]) -> bool:
    if key in card_by_key:
        return True
    return key in question_by_key


def _available_prep_keys(item_ids: list[str], card_by_key: dict[str, dict[str, Any]],
                         question_by_key: dict[str, dict[str, Any]]) -> list[str]:
    return [key for key in item_ids if _has_prep_source(key, card_by_key, question_by_key)]


def reserve_session_prep_blend(inputs: dict[str, Any], _plan: dict[str, Any], build: SessionBuildState) -> None:
    from .queue import shuffle_with_seed

    prep = inputs["options"].get("sessionPrepBlend")
    item_ids = _configured_prep_item_ids(prep)
    share_value = _configured_prep_share(prep)
    if not prep:
        return
    if not item_ids:
        return
    if not _valid_prep_share(share_value):
        return
    seats = _prep_seat_count(inputs["size"], len(build.selected_items), share_value)
    if seats == 0:
        return

    card_by_key, question_by_key = _prep_maps(inputs, item_ids)
    available = _available_prep_keys(item_ids, card_by_key, question_by_key)
    shuffled = shuffle_with_seed(available, f"{inputs['servingRotationSeed']}:class-prep")
    ordered_ids = order_prep_keys(shuffled, prep)
    stop_at = min(inputs["size"], len(build.selected_items) + seats)
    context = PrepReservationContext(inputs, build, card_by_key, question_by_key)
    _reserve_prep_keys(context, ordered_ids, stop_at)
