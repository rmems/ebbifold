"""Session-preparation reservation lane."""

from __future__ import annotations

from typing import Any

from .build_state import SessionBuildState


def _interleave_prep_keys(keys: list[str], weak: set[str], unseen: set[str], share: float) -> list[str]:
    weak_keys = [key for key in keys if key in weak]
    unseen_keys = [key for key in keys if key in unseen]
    out = []
    for index in range(max(len(weak_keys), len(unseen_keys))):
        if share < 0.25 and index < len(unseen_keys):
            out.append(unseen_keys[index])
        if index < len(weak_keys):
            out.append(weak_keys[index])
        if share >= 0.25 and index < len(unseen_keys):
            out.append(unseen_keys[index])
    return out + [key for key in keys if key not in weak and key not in unseen]


def _prep_session_queues(keys: list[str], sessions: list[dict[str, Any]], weak: set[str],
                         unseen: set[str], share: float) -> list[dict[str, Any]]:
    queues = []
    for session in sessions:
        ids = set(session.get("itemIds") or [])
        session_keys = [key for key in keys if key in ids]
        queues.append({"share": session["share"], "taken": 0,
                       "keys": _interleave_prep_keys(session_keys, weak, unseen, share)})
    return queues


def _drop_used_prep_keys(queues: list[dict[str, Any]], used: set[str]) -> None:
    for queue in queues:
        while queue["keys"] and queue["keys"][0] in used:
            queue["keys"].pop(0)


def order_prep_keys(keys: list[str], plan: dict[str, Any]) -> list[str]:
    weak = set(plan.get("weakItemIds") or [])
    unseen = set(plan.get("unseenItemIds") or [])
    share = plan.get("share") or 0
    sessions = [session for session in plan.get("sessions") or [] if (session.get("share") or 0) > 0]
    if len(sessions) < 2:
        return _interleave_prep_keys(keys, weak, unseen, share)
    queues = _prep_session_queues(keys, sessions, weak, unseen, share)
    used, out = set(), []
    while True:
        _drop_used_prep_keys(queues, used)
        available = [queue for queue in queues if queue["keys"]]
        if not available:
            break
        queue = min(available, key=lambda item: (item["taken"] + 1) / item["share"])
        key = queue["keys"].pop(0)
        queue["taken"] += 1
        used.add(key)
        out.append(key)
    return out + [key for key in keys if key not in used]

def _prep_card_concept_id(topics: list[str], inputs: dict[str, Any]) -> str:
    topic_set = set(topics)
    match = next((concept for concept in inputs["concepts"] if any(topic in topic_set for topic in concept.get("topics") or [])), None)
    return match["id"] if match else "_unattached"


def _prep_question_concept_id(question_id: str, inputs: dict[str, Any]) -> str:
    links = [link for link in inputs["bulk"]["questionConceptLinks"] if link["questionId"] == question_id]
    primary = next((link for link in links if link.get("isPrimary")), None)
    link = primary or (links[0] if links else None)
    return link["conceptId"] if link else "_unattached"


def _prep_maps(inputs: dict[str, Any], item_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bulk, wanted = inputs["bulk"], set(item_ids)
    cards = [card for card in bulk["unseenCards"] + bulk["seenCards"] if f"card:{card['id']}" in wanted]
    questions = [question for question in bulk["rotationQuestions"] if f"question:{question['id']}" in wanted]
    return ({f"card:{card['id']}": card for card in cards},
            {f"question:{question['id']}": question for question in questions})


def _prep_question_allowed(build: SessionBuildState, question: dict[str, Any], familiarity: dict[str, Any], reentry: dict[str, int]) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    suppression = question_suppression_key(question)
    if question["id"] in build.selected_question_ids:
        return False
    if suppression and suppression in build.selected_question_variant_groups:
        return False
    return take_with_reentry_cap(question["id"], familiarity, reentry)


def _prep_item(inputs: dict[str, Any], card: dict[str, Any] | None, question: dict[str, Any] | None) -> dict[str, Any]:
    source = card or question
    concept_id = (_prep_card_concept_id(card.get("topics") or [], inputs) if card
                  else _prep_question_concept_id(question["id"], inputs))
    concept = inputs["conceptMap"].get(concept_id)
    item = {
        "type": "card" if card else "question", "id": source["id"], "conceptId": concept_id,
        "conceptName": (concept or {}).get("name") or "Class preparation", "priority": 1,
        "interventionReason": "pre_teach", "topics": source.get("topics") or [],
        "variantGroupId": source.get("variantGroupId"), "variantType": source.get("variantType"),
        "rotation": source.get("rotation"), "challengePolicyApplied": False,
    }
    if card:
        item.update({"variantIndex": card.get("variantIndex"), "complexity": card.get("complexity"), "clusterId": card.get("clusterId")})
    return item


def _add_prep_candidate(inputs: dict[str, Any], build: SessionBuildState,
                        card: dict[str, Any] | None, question: dict[str, Any] | None) -> bool:
    picked = None
    if card:
        picked = build.pick_card_candidate([card], similarity_threshold=inputs["interferenceThreshold"], max_per_cluster=3)
        if not picked:
            return False
    reentry = dict(build.mastered_reentry_counter)
    if question and not _prep_question_allowed(build, question, inputs["bulk"]["questionFamiliarity"], reentry):
        return False
    added = build.add_item(_prep_item(inputs, card, question))
    if added and card:
        build.record_cluster(card.get("clusterId"))
        build.record_card_neighborhood(card)
    if added and question:
        build.mastered_reentry_counter["masteredServed"] = reentry["masteredServed"]
    return added


def reserve_session_prep_blend(inputs: dict[str, Any], _plan: dict[str, Any], build: SessionBuildState) -> None:
    from .queue import shuffle_with_seed

    prep = inputs["options"].get("sessionPrepBlend")
    item_ids = list((prep or {}).get("itemIds") or [])
    share_value = (prep or {}).get("share")
    if not prep or not isinstance(share_value, (int, float)) or share_value <= 0 or not item_ids:
        return
    share = min(0.5, max(0.15, share_value))
    discretionary = max(0, inputs["size"] - len(build.selected_items))
    seats = min(discretionary, int(discretionary * share))
    if seats == 0:
        return

    card_by_key, question_by_key = _prep_maps(inputs, item_ids)
    available = [key for key in item_ids if key in card_by_key or key in question_by_key]
    shuffled = shuffle_with_seed(available, f"{inputs['servingRotationSeed']}:class-prep")
    ordered_ids = order_prep_keys(shuffled, prep)
    stop_at = min(inputs["size"], len(build.selected_items) + seats)

    for key in ordered_ids:
        if len(build.selected_items) >= stop_at:
            break
        card, question = card_by_key.get(key), question_by_key.get(key)
        if card or question:
            _add_prep_candidate(inputs, build, card, question)
