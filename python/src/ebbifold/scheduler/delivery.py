"""Delivery telemetry helpers ported from the TypeScript scheduler.

All history and proximity inputs are explicit snapshot data. The scheduler does
not read committed proximity artifacts or learner memory directly.
"""

from __future__ import annotations

import math
from typing import Any

PROXIMITY_TIGHT_FLOOR = 0.85
PROXIMITY_LOOSE_FLOOR = 0.78
SERVE_CONDITIONING_VERSION = 1
EASY = 5
_STEP_REASON = {1: "reword", 2: "same-fact-mcq", 3: "harder-same-fact-mcq"}


def links_for_card(overlay: dict[str, Any] | None, stable_id: str) -> dict[str, list[dict[str, Any]]]:
    links = ((overlay or {}).get("cards") or {}).get(stable_id) or []
    tight: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    for link in sorted(links, key=lambda row: -(row.get("sim") or 0)):
        sim = link.get("sim")
        if not isinstance(sim, (int, float)) or isinstance(sim, bool) or not math.isfinite(sim):
            continue
        if sim >= PROXIMITY_TIGHT_FLOOR:
            tight.append(link)
        elif sim >= PROXIMITY_LOOSE_FLOOR:
            loose.append(link)
    return {"tight": tight, "loose": loose}


def evidence_for_card(overlay: dict[str, Any] | None, stable_id: str,
                      outcomes: dict[str, dict[str, int]]) -> dict[str, Any] | None:
    links = links_for_card(overlay, stable_id)
    if not links["tight"] and not links["loose"]:
        return None

    def total(rows: list[dict[str, Any]]) -> dict[str, int]:
        n = correct = 0
        for link in rows:
            outcome = outcomes.get(link.get("q"))
            if outcome:
                n += outcome.get("n", 0)
                correct += outcome.get("correct", 0)
        return {"n": n, "correct": correct}

    return {"tight": total(links["tight"]), "loose": total(links["loose"])}


def build_serve_conditioning(record: dict[str, Any] | None, overlay: dict[str, Any] | None,
                             stable_id: str | None) -> dict[str, Any] | None:
    if not record:
        return None
    return {
        "version": SERVE_CONDITIONING_VERSION,
        "verdict": record.get("verdict"),
        "evidence": evidence_for_card(overlay, stable_id, record.get("questionOutcomes") or {}) if stable_id else None,
        "calibration": record.get("calibration"),
        "recordComputedAt": record.get("computedAt"),
    }


def is_reword(card: dict[str, Any]) -> bool:
    group = card.get("variantGroupId")
    return card.get("variantType") is None and group is not None and not str(group).startswith("dup-")


def default_target_step(due: dict[str, Any]) -> int:
    if due.get("targetStep") is not None:
        return due["targetStep"]
    if due.get("correctCount", 0) >= 2 or (due.get("lastQuality") or 0) >= EASY:
        return 2
    return 1


def _search_order(target: int) -> list[int]:
    return list(range(target, 4)) + list(range(target - 1, 0, -1))


def _pick_step(step: int, due: dict[str, Any], pools: dict[str, Any]) -> str | None:
    if step == 1:
        cards = [
            card for card in pools["siblings"]
            if card.get("variantGroupId") is not None
            and card.get("variantGroupId") == due.get("variantGroupId")
            and is_reword(card)
            and card["id"] != due["cardId"]
            and card["id"] not in pools["seenCardIds"]
        ]
        cards.sort(key=lambda c: (abs((c.get("complexity") or 1) - (due.get("complexity") or 1)), c["id"]))
        return cards[0]["id"] if cards else None
    links = [
        link for link in pools["proximity"].get(due["stableId"], [])
        if link.get("sim", 0) >= PROXIMITY_TIGHT_FLOOR and link.get("q") not in pools["seenQuestionIds"]
    ]
    if step == 3:
        links = [link for link in links if pools["questionComplexity"].get(link["q"], -math.inf) > due.get("complexity", 1)]
    links.sort(key=lambda link: (-link.get("sim", 0), link.get("q") or ""))
    return links[0]["q"] if links else None


def substitute_due_slot(due: dict[str, Any], pools: dict[str, Any]) -> dict[str, Any]:
    if due.get("lastQuality") is None or due["lastQuality"] <= 2:
        return {"kind": "keep", "itemType": "card", "id": due["cardId"], "reason": "not-yet-correct"}
    target = default_target_step(due)
    for step in _search_order(target):
        pick = _pick_step(step, due, pools)
        if pick:
            return {
                "kind": "substitute",
                "itemType": "card" if step == 1 else "question",
                "id": pick,
                "step": step,
                "reason": _STEP_REASON[step],
            }
    return {"kind": "empty", "supplyEmpty": True, "stableId": due["stableId"], "reason": "no-unseen-probe"}


def repetition_slot_shadow(due: dict[str, Any], pools: dict[str, Any]) -> dict[str, Any] | None:
    if due.get("lastQuality") is None or due["lastQuality"] <= 2:
        return None
    decision = substitute_due_slot(due, pools)
    if decision["kind"] == "keep":
        return None
    if decision["kind"] == "substitute":
        return {
            "kind": "substitute",
            "wouldServeType": decision["itemType"],
            "wouldServeId": decision["id"],
            "step": decision["step"],
            "reason": decision["reason"],
            "supplyEmpty": False,
            "shadow": True,
        }
    return {
        "kind": "empty",
        "wouldServeType": None,
        "wouldServeId": None,
        "step": None,
        "reason": decision["reason"],
        "supplyEmpty": True,
        "shadow": True,
    }


def build_repetition_slot_context(*, cards: list[dict[str, Any]], grades: dict[str, dict[str, Any]] | None,
                                  seen_card_ids: list[str], variant_group_history: dict[str, set[str]],
                                  seen_question_ids: list[str], overlay: dict[str, Any] | None) -> dict[str, Any] | None:
    if not grades:
        return None
    cards_by_id = {card["id"]: card for card in cards}
    due_cards: dict[str, dict[str, Any]] = {}
    for card_id, grade in grades.items():
        card = cards_by_id.get(card_id)
        if not card or not card.get("stableId"):
            continue
        due_cards[card_id] = {
            "cardId": card_id,
            "stableId": card["stableId"],
            "variantGroupId": card.get("variantGroupId"),
            "variantType": card.get("variantType"),
            "complexity": card.get("complexity") or 1,
            "lastQuality": grade.get("lastQuality"),
            "correctCount": grade.get("correctCount") or 0,
        }
    if not due_cards:
        return None
    seen = set(seen_card_ids)
    for ids in variant_group_history.values():
        seen.update(ids)
    proximity = {stable_id: list(links) for stable_id, links in ((overlay or {}).get("cards") or {}).items()}
    return {
        "grades": due_cards,
        "pools": {
            "siblings": [
                {
                    "id": card["id"],
                    "stableId": card.get("stableId") or "",
                    "variantGroupId": card.get("variantGroupId"),
                    "variantType": card.get("variantType"),
                    "complexity": card.get("complexity") or 1,
                }
                for card in cards
            ],
            "seenCardIds": sorted(seen),
            "proximity": proximity,
            "seenQuestionIds": sorted(set(seen_question_ids)),
            "questionComplexity": {},
        },
    }


def shadow_for_due_id(card_id: str, context: dict[str, Any] | None) -> dict[str, Any] | None:
    due = (context or {}).get("grades", {}).get(card_id)
    if not due or not context:
        return None
    return repetition_slot_shadow(due, context["pools"])
