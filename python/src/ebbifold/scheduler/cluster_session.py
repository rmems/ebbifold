"""Empty-concept fallback (unified-scheduler-cluster-session.ts).

Every storage read is an explicit snapshot adapter: ``clusterMastery`` rows
(getClusterMastery), ``clusterCards`` per cluster (getCardsForCluster, ordered,
with ``firstSightAtSelection``), ``clusterQuestions`` (getQuestionsForRotation),
``clusterScaffolds`` (getUnseenScaffoldsForClusters), ``itemEmbeddings``.
Deck strand mixes, cross-source/media/week cluster restriction and prep blends
are rejected."""

from __future__ import annotations

import math
from typing import Any

from .build_state import TEACHING_ARC_TARGET_ITEMS, question_suppression_key
from .cluster_card_selection import select_cluster_cards
from .cluster_eligibility import eligible_rows, restricted
from .deck_strand_mix import order_by_deck_strands
from .lane_utils import thread_trace
from .prep_allocation import order_session_prep_candidates, prep_modality_budgets
from .queue import break_modality_runs, compute_concept_pairing_rate, order_by_manifold_walk
from .scoring import classify_teaching_state, enforce_concept_teaching_caps


def _check(inputs: dict[str, Any]) -> None:
    mix = inputs.get("deckStrandMix")
    if mix is not None and not isinstance(mix.get("strands"), list):
        raise ValueError("deckStrandMix must be a serializable strand mapping")


def _cluster_states(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states = []
    for c in clusters:
        coverage = c["cardsReviewed"] / c["cardCount"] if c["cardCount"] > 0 else 1
        mastery = c.get("mastery") or 0
        priority = (1 - coverage) * 0.6 + (1 - mastery) * 0.4 + (0.1 if c.get("needsAttention") else 0)
        states.append({**c, "coverage": coverage, "priority": priority})
    states.sort(key=lambda s: -s["priority"])
    return states


def construct_cluster_session(inputs: dict[str, Any]) -> dict[str, Any]:
    _check(inputs)
    size, card_ratio = inputs["size"], inputs["cardRatio"]
    max_new = inputs["effectiveMaxNewCards"]
    required = min(size, max(0, math.floor(inputs["options"].get("minFirstSightItems") or 0)))
    clusters = list(inputs.get("clusterMastery") or [])
    if not clusters and inputs["options"].get("media") is None:
        return {"items": [], "noveltyQuota": {"required": required, "selected": 0},
                "stats": {"totalConcepts": 0, "weakConcepts": 0, "selectedConcepts": 0, "cardCount": 0,
                          "questionCount": 0, "averagePriority": 0}}
    states = _cluster_states(clusters)
    if restricted(inputs):
        states = [s for s in states if eligible_rows(inputs, (inputs.get("clusterCards") or {}).get(s["clusterId"], []))]
    target_cards = math.ceil(size * card_ratio)
    target_questions = size - target_cards
    sel_cards = set(inputs["excludedCardIds"])
    sel_questions = set(inputs["excludedQuestionIds"])
    question_groups: set[str] = set()
    if restricted(inputs) and not states and inputs["options"].get("media") is None:
        return {"items": [], "noveltyQuota": {"required": required, "selected": 0},
                "stats": {"totalConcepts": 0, "weakConcepts": 0, "selectedConcepts": 0, "cardCount": 0,
                          "questionCount": 0, "averagePriority": 0}}
    items, selected_clusters, strand_identities = select_cluster_cards(inputs, states, target_cards)
    ctx = {"states": states, "items": items, "required": required, "selectedClusters": selected_clusters,
           "selCards": sel_cards, "questionGroups": question_groups, "selQuestions": sel_questions,
           "targetQuestions": target_questions, "maxNew": max_new, "size": size,
           "strandIdentities": strand_identities}
    return _finish_cluster_session(inputs, ctx)


def _cluster_anchor(it: dict[str, Any]) -> bool:
    if it.get("interventionReason") == "concept_followup":
        return False
    if it["type"] == "question":
        return it.get("clusterId") is not None
    if it["type"] == "card":
        return it.get("complexity") is not None and it["complexity"] >= 2 and it.get("clusterId") is not None
    return False


def _cluster_pairing_limit(items: list[dict[str, Any]]) -> int:
    return max(2, min(max(2, sum(1 for i in items if _cluster_anchor(i))), max(2, len(items) // 3)))


def _cluster_scaffold_candidate(
    candidates: list[dict[str, Any]], used: set[str], present: set[str], groups: set[str], anchor: dict[str, Any]
) -> dict[str, Any] | None:
    return next(
        (c for c in candidates if c["id"] not in used and c["id"] not in present
         and (not c.get("variantGroupId") or c["variantGroupId"] not in groups)
         and c.get("clusterId") is not None and c["clusterId"] == anchor["clusterId"]),
        None,
    )


def _cluster_scaffold_item(anchor: dict[str, Any], card: dict[str, Any], rotation: str) -> dict[str, Any]:
    return {"type": "card", "id": card["id"], "rotation": anchor["rotation"] if anchor.get("rotation") is not None else rotation,
            "priority": 0.5, "conceptId": anchor.get("conceptId") or "", "conceptName": anchor.get("conceptName") or "",
            "topics": card["topics"], "complexity": card.get("complexity"), "clusterId": card.get("clusterId"),
            "variantGroupId": card.get("variantGroupId"), "interventionReason": "preemptive_scaffold",
            "struggleIntervention": {"strategy": "preemptive", "isScaffold": True, "targetCardId": anchor["id"]}}


def _insert_after_positions(items: list[dict[str, Any]], inserts: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        out.append(it)
        if idx in inserts:
            out.append(inserts[idx])
    return out


def _cluster_scaffolds(items: list[dict[str, Any]], pool: list[dict[str, Any]], rotation: str, seed: str,
                       max_pairings: int | None) -> list[dict[str, Any]]:
    from .queue import shuffle_with_seed

    if max_pairings is None:
        max_pairings = _cluster_pairing_limit(items)
    if max_pairings <= 0 or not items or not pool:
        return items
    scaffolds = [c for c in pool if c.get("complexity") == 1]
    if not scaffolds:
        return items
    candidates = shuffle_with_seed(scaffolds, seed)
    used: set[str] = set()
    present = {i["id"] for i in items}
    groups = {i["variantGroupId"] for i in items if i["type"] == "card" and i.get("variantGroupId")}
    inserts: dict[int, dict[str, Any]] = {}
    for idx, it in enumerate(items):
        if len(inserts) >= max_pairings:
            break
        if not _cluster_anchor(it):
            continue
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        if nxt and nxt["type"] == "card" and nxt.get("complexity") == 1:
            continue
        card = _cluster_scaffold_candidate(candidates, used, present, groups, it)
        if card is None:
            continue
        used.add(card["id"])
        if card.get("variantGroupId"):
            groups.add(card["variantGroupId"])
        inserts[idx] = _cluster_scaffold_item(it, card, rotation)
    if not inserts:
        return items
    return _insert_after_positions(items, inserts)


def _append_cluster_questions(inputs: dict[str, Any], ctx: dict[str, Any], first_sight_q: set[str]) -> None:
    count = max(ctx["size"] if inputs["options"].get("media") else ctx["targetQuestions"], ctx["required"])
    if count <= 0:
        return
    familiarity = inputs.get("questionFamiliarity") or {}
    eligible = eligible_rows(inputs, inputs.get("clusterQuestions") or [])
    rows = [r for r in eligible if r["id"] not in ctx["selQuestions"]][:count]
    for q in rows:
        if q["id"] not in familiarity:
            first_sight_q.add(q["id"])
        ctx["items"].append({"type": "question", "id": q["id"], "conceptId": f"question:{q['id']}",
                             "conceptName": "Rotation probe", "priority": 0.5, "interventionReason": "needs_retest",
                             **thread_trace(), "rotation": q.get("rotation"), "topics": q["topics"],
                             "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType"),
                             "firstSightAtSelection": q["id"] in first_sight_q})
        key = question_suppression_key(q)
        if key:
            ctx["questionGroups"].add(key)


def _trim_to_first_sight_quota(items: list[dict[str, Any]], required: int, size: int) -> None:
    if required <= 0 or len(items) <= size:
        return
    first = [i for i in items if i["firstSightAtSelection"]][:required]
    reserved = {(i["type"], i["id"]) for i in first}
    items[:] = (first + [i for i in items if (i["type"], i["id"]) not in reserved])[:size]


def _copy_scaffold_metadata(paired: list[dict[str, Any]], scaffolds: list[dict[str, Any]]) -> None:
    meta = {c["id"]: c for c in scaffolds}
    for it in paired:
        if it["type"] == "card" and it.get("interventionReason") == "preemptive_scaffold" and it["id"] in meta:
            for key in ("variantGroupId", "variantIndex", "variantType"):
                it[key] = meta[it["id"]].get(key)


def _remove_non_first_sight(reconciled: list[dict[str, Any]], keys: set[tuple[str, str]], per_concept: dict[str, int]) -> bool:
    at = next((k for k in range(len(reconciled) - 1, -1, -1) if not reconciled[k].get("firstSightAtSelection")), -1)
    if at < 0:
        return False
    removed = reconciled.pop(at)
    keys.discard((removed["type"], removed["id"]))
    per_concept[removed["conceptId"]] = max(0, per_concept.get(removed["conceptId"], 1) - 1)
    return True


def _reconcile_first_sight_quota(
    reconciled: list[dict[str, Any]], quota_candidates: list[dict[str, Any]], required: int, size: int,
    caps: dict[str, int]
) -> None:
    keys = {(i["type"], i["id"]) for i in reconciled}
    per_concept: dict[str, int] = {}
    for i in reconciled:
        per_concept[i["conceptId"]] = per_concept.get(i["conceptId"], 0) + 1
    fs = sum(1 for i in reconciled if i.get("firstSightAtSelection"))
    for cand in quota_candidates:
        if fs >= required:
            break
        key = (cand["type"], cand["id"])
        if not cand["firstSightAtSelection"] or key in keys:
            continue
        if per_concept.get(cand["conceptId"], 0) >= caps.get(cand["conceptId"], 3):
            continue
        if len(reconciled) >= size and not _remove_non_first_sight(reconciled, keys, per_concept):
            break
        reconciled.append(cand)
        keys.add(key)
        per_concept[cand["conceptId"]] = per_concept.get(cand["conceptId"], 0) + 1
        fs += 1


def _strand_identity_map(inputs: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    for rows in (inputs.get("clusterCards") or {}).values():
        for row in rows:
            identities[("card", row["id"])] = {"rotation": row.get("rotation"), "sourceFile": row.get("sourceFile")}
    for row in inputs.get("clusterQuestions") or []:
        identities[("question", row["id"])] = {"rotation": row.get("rotation"), "sourceFile": row.get("sourceFile")}
    for row in inputs.get("clusterScaffolds") or []:
        identities[("card", row["id"])] = {"rotation": row.get("rotation"), "sourceFile": row.get("sourceFile")}
    return identities


def _order_cluster_strands(inputs: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mix = inputs.get("deckStrandMix")
    if not mix or not mix.get("strands"):
        return items
    identities = _strand_identity_map(inputs)

    def identify(item: dict[str, Any]) -> dict[str, Any]:
        return identities.get((item["type"], item["id"])) or {
            "rotation": item.get("rotation") or inputs["rotation"], "sourceFile": None,
        }

    return order_by_deck_strands(
        items, mix, seed=inputs["servingRotationSeed"], identify=identify,
        keep_with_previous=lambda item: item.get("interventionReason") == "preemptive_scaffold",
        share_basis="present",
    )


def _finish_cluster_session(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    items, required, size = ctx["items"], ctx["required"], ctx["size"]
    first_sight_q: set[str] = set()
    _append_cluster_questions(inputs, ctx, first_sight_q)
    quota_candidates = list(items)
    _trim_to_first_sight_quota(items, required, size)
    ordered = order_by_manifold_walk(items, inputs.get("itemEmbeddings") or {})
    scaffolds = [] if ctx["maxNew"] == 0 else [
        c for c in eligible_rows(inputs, inputs.get("clusterScaffolds") or [])
        if c["id"] not in ctx["selCards"] and c.get("clusterId") in ctx["selectedClusters"]
    ]
    paired = _cluster_scaffolds(
        ordered, scaffolds, inputs["rotation"], inputs["servingRotationSeed"], 0 if ctx["maxNew"] == 0 else None
    )
    _copy_scaffold_metadata(paired, scaffolds)
    caps = {
        s["clusterId"]: TEACHING_ARC_TARGET_ITEMS[classify_teaching_state(s["cardsReviewed"], s["mastery"], s["mastery"])]
        for s in ctx["states"]
    }
    prep_keys = set((inputs["options"].get("sessionPrepBlend") or {}).get("itemIds") or [])
    reconciled = list(enforce_concept_teaching_caps(
        paired, lambda cid: caps.get(cid, 3),
        preserve=lambda item: f"{item['type']}:{item['id']}" in prep_keys,
    ))
    _reconcile_first_sight_quota(reconciled, quota_candidates, required, size, caps)
    final = break_modality_runs(_order_cluster_strands(inputs, reconciled))
    unseen_scaffolds = {c["id"] for c in scaffolds}
    tagged = [
        {**i, "firstSightAtSelection": i["firstSightAtSelection"] if i.get("firstSightAtSelection") is not None
         else i["id"] in unseen_scaffolds}
        for i in final
    ]
    tagged = order_session_prep_candidates(
        tagged, inputs["options"].get("sessionPrepBlend"), size=size,
        key=lambda item: f"{item['type']}:{item['id']}",
        protected=lambda item: item.get("interventionReason") in {"failure_escalation", "needs_retest"},
    )
    avg = sum(i["priority"] for i in tagged) / len(tagged) if tagged else 0
    return {"items": tagged, "noveltyQuota": {"required": required,
                                              "selected": sum(1 for i in tagged if i["firstSightAtSelection"])},
            "stats": {"totalConcepts": len(ctx["states"]), "weakConcepts": sum(1 for s in ctx["states"] if s["priority"] > 0),
                      "selectedConcepts": len(ctx["selectedClusters"]),
                      "cardCount": sum(1 for i in tagged if i["type"] == "card"),
                      "questionCount": sum(1 for i in tagged if i["type"] == "question"),
                      "averagePriority": math.floor(avg * 1000 + 0.5) / 1000,
                      "conceptPairingRate": compute_concept_pairing_rate(tagged)}}
