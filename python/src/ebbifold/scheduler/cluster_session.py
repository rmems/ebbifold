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
from .errors import UnsupportedSchedulerInput
from .lanes import thread_trace
from .queue import break_modality_runs, compute_concept_pairing_rate, order_by_manifold_walk
from .scoring import classify_teaching_state, enforce_concept_teaching_caps


def _check(inputs: dict[str, Any]) -> None:
    options = inputs["options"]
    if inputs.get("deckStrandMix") or options.get("sessionPrepBlend") or options.get("media") is not None:
        raise UnsupportedSchedulerInput("cluster fallback strand/prep/media lanes are not ported in Phase 3a")
    if options.get("crossSourceRotations") or inputs.get("week") is not None:
        raise UnsupportedSchedulerInput("cluster fallback eligibility restriction is not ported in Phase 3a")


def construct_cluster_session(inputs: dict[str, Any]) -> dict[str, Any]:
    _check(inputs)
    size, card_ratio = inputs["size"], inputs["cardRatio"]
    max_new = inputs["effectiveMaxNewCards"]
    required = min(size, max(0, math.floor(inputs["options"].get("minFirstSightItems") or 0)))
    clusters = list(inputs.get("clusterMastery") or [])
    if not clusters:
        return {"items": [], "noveltyQuota": {"required": required, "selected": 0},
                "stats": {"totalConcepts": 0, "weakConcepts": 0, "selectedConcepts": 0, "cardCount": 0,
                          "questionCount": 0, "averagePriority": 0}}
    states = []
    for c in clusters:
        coverage = c["cardsReviewed"] / c["cardCount"] if c["cardCount"] > 0 else 1
        mastery = c.get("mastery") or 0
        priority = (1 - coverage) * 0.6 + (1 - mastery) * 0.4 + (0.1 if c.get("needsAttention") else 0)
        states.append({**c, "coverage": coverage, "priority": priority})
    states.sort(key=lambda s: -s["priority"])
    target_cards = math.ceil(size * card_ratio)
    target_questions = size - target_cards
    items: list[dict[str, Any]] = []
    sel_cards = set(inputs["excludedCardIds"])
    sel_questions = set(inputs["excludedQuestionIds"])
    card_groups: set[str] = set()
    question_groups: set[str] = set()
    per_cluster: dict[str, int] = {}
    selected_clusters: set[str] = set()
    card_count = 0
    pass_no = 0
    while card_count < target_cards:
        added = 0
        for cluster in states:
            if card_count >= target_cards:
                break
            already = per_cluster.get(cluster["clusterId"], 0)
            if already >= 2 + pass_no:
                continue
            pool = [r for r in (inputs.get("clusterCards") or {}).get(cluster["clusterId"], []) if r["id"] not in sel_cards][:12]
            picked = next((r for r in pool if not r.get("variantGroupId") or r["variantGroupId"] not in card_groups), None)
            if picked is None:
                continue
            items.append({"type": "card", "id": picked["id"], "conceptId": cluster["clusterId"],
                          "conceptName": cluster["clusterName"], "priority": cluster["priority"],
                          "interventionReason": "weak_recall" if cluster["mastery"] < 0.6 else "reinforcement",
                          "rotation": inputs["rotation"], "topics": picked["topics"], "complexity": picked.get("complexity"),
                          "clusterId": picked.get("clusterId"), "variantGroupId": picked.get("variantGroupId"),
                          "variantIndex": picked.get("variantIndex"), "variantType": picked.get("variantType"),
                          "firstSightAtSelection": picked["firstSightAtSelection"]})
            sel_cards.add(picked["id"])
            if picked.get("variantGroupId"):
                card_groups.add(picked["variantGroupId"])
            card_count += 1
            per_cluster[cluster["clusterId"]] = already + 1
            selected_clusters.add(cluster["clusterId"])
            added += 1
        if added == 0:
            break
        pass_no += 1
    ctx = {"states": states, "items": items, "required": required, "selectedClusters": selected_clusters,
           "selCards": sel_cards, "questionGroups": question_groups, "selQuestions": sel_questions,
           "targetQuestions": target_questions, "maxNew": max_new, "size": size}
    return _finish_cluster_session(inputs, ctx)


def _cluster_scaffolds(items: list[dict[str, Any]], pool: list[dict[str, Any]], rotation: str, seed: str,
                       max_pairings: int | None) -> list[dict[str, Any]]:
    from .queue import shuffle_with_seed

    def anchor(it: dict[str, Any]) -> bool:
        if it.get("interventionReason") == "concept_followup":
            return False
        if it["type"] == "question":
            return it.get("clusterId") is not None
        if it["type"] == "card":
            return it.get("complexity") is not None and it["complexity"] >= 2 and it.get("clusterId") is not None
        return False

    if max_pairings is None:
        max_pairings = max(2, min(max(2, sum(1 for i in items if anchor(i))), max(2, len(items) // 3)))
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
        if not anchor(it):
            continue
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        if nxt and nxt["type"] == "card" and nxt.get("complexity") == 1:
            continue
        card = next((c for c in candidates if c["id"] not in used and c["id"] not in present
                     and (not c.get("variantGroupId") or c["variantGroupId"] not in groups)
                     and c.get("clusterId") is not None and c["clusterId"] == it["clusterId"]), None)
        if card is None:
            continue
        used.add(card["id"])
        if card.get("variantGroupId"):
            groups.add(card["variantGroupId"])
        inserts[idx] = {"type": "card", "id": card["id"], "rotation": it["rotation"] if it.get("rotation") is not None else rotation, "priority": 0.5,
                        "conceptId": it.get("conceptId") or "", "conceptName": it.get("conceptName") or "",
                        "topics": card["topics"], "complexity": card.get("complexity"), "clusterId": card.get("clusterId"),
                        "variantGroupId": card.get("variantGroupId"), "interventionReason": "preemptive_scaffold",
                        "struggleIntervention": {"strategy": "preemptive", "isScaffold": True, "targetCardId": it["id"]}}
    if not inserts:
        return items
    out: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        out.append(it)
        if idx in inserts:
            out.append(inserts[idx])
    return out


def _finish_cluster_session(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    items, required, size = ctx["items"], ctx["required"], ctx["size"]
    familiarity = inputs.get("questionFamiliarity") or {}
    first_sight_q: set[str] = set()
    count = max(ctx["targetQuestions"], required)
    if count > 0:
        rows = [r for r in (inputs.get("clusterQuestions") or []) if r["id"] not in ctx["selQuestions"]][:count]
        for q in rows:
            if q["id"] not in familiarity:
                first_sight_q.add(q["id"])
            items.append({"type": "question", "id": q["id"], "conceptId": f"question:{q['id']}",
                          "conceptName": "Rotation probe", "priority": 0.5, "interventionReason": "needs_retest",
                          **thread_trace(), "rotation": q.get("rotation"), "topics": q["topics"],
                          "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType"),
                          "firstSightAtSelection": q["id"] in first_sight_q})
            key = question_suppression_key(q)
            if key:
                ctx["questionGroups"].add(key)
    quota_candidates = list(items)
    if required > 0 and len(items) > size:
        first = [i for i in items if i["firstSightAtSelection"]][:required]
        reserved = {(i["type"], i["id"]) for i in first}
        items[:] = (first + [i for i in items if (i["type"], i["id"]) not in reserved])[:size]
    ordered = order_by_manifold_walk(items, inputs.get("itemEmbeddings") or {})
    scaffolds = [] if ctx["maxNew"] == 0 else [c for c in (inputs.get("clusterScaffolds") or [])
                                               if c["id"] not in ctx["selCards"]
                                               and c.get("clusterId") in ctx["selectedClusters"]]
    paired = _cluster_scaffolds(ordered, scaffolds, inputs["rotation"], inputs["servingRotationSeed"],
                                0 if ctx["maxNew"] == 0 else None)
    meta = {c["id"]: c for c in scaffolds}
    for it in paired:
        if it["type"] == "card" and it.get("interventionReason") == "preemptive_scaffold" and it["id"] in meta:
            for key in ("variantGroupId", "variantIndex", "variantType"):
                it[key] = meta[it["id"]].get(key)
    caps = {s["clusterId"]: TEACHING_ARC_TARGET_ITEMS[classify_teaching_state(s["cardsReviewed"], s["mastery"], s["mastery"])]
            for s in ctx["states"]}
    reconciled = list(enforce_concept_teaching_caps(paired, lambda cid: caps.get(cid, 3)))
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
        if len(reconciled) >= size:
            at = next((k for k in range(len(reconciled) - 1, -1, -1) if not reconciled[k].get("firstSightAtSelection")), -1)
            if at < 0:
                break
            removed = reconciled.pop(at)
            keys.discard((removed["type"], removed["id"]))
            per_concept[removed["conceptId"]] = max(0, per_concept.get(removed["conceptId"], 1) - 1)
        reconciled.append(cand)
        keys.add(key)
        per_concept[cand["conceptId"]] = per_concept.get(cand["conceptId"], 0) + 1
        fs += 1
    final = break_modality_runs(reconciled)
    unseen_scaffolds = {c["id"] for c in scaffolds}
    tagged = [{**i, "firstSightAtSelection": i["firstSightAtSelection"] if i.get("firstSightAtSelection") is not None
               else i["id"] in unseen_scaffolds} for i in final]
    avg = sum(i["priority"] for i in tagged) / len(tagged) if tagged else 0
    return {"items": tagged, "noveltyQuota": {"required": required,
                                              "selected": sum(1 for i in tagged if i["firstSightAtSelection"])},
            "stats": {"totalConcepts": len(ctx["states"]), "weakConcepts": sum(1 for s in ctx["states"] if s["priority"] > 0),
                      "selectedConcepts": len(ctx["selectedClusters"]),
                      "cardCount": sum(1 for i in tagged if i["type"] == "card"),
                      "questionCount": sum(1 for i in tagged if i["type"] == "question"),
                      "averagePriority": math.floor(avg * 1000 + 0.5) / 1000,
                      "conceptPairingRate": compute_concept_pairing_rate(tagged)}}
