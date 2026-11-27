"""Empty-concept fallback (unified-scheduler-cluster-session.ts).

Every storage read is an explicit snapshot adapter: ``clusterMastery`` rows
(getClusterMastery), ``clusterCards`` per cluster (getCardsForCluster, ordered,
with ``firstSightAtSelection``), ``clusterQuestions`` (getQuestionsForRotation),
``clusterScaffolds`` (getUnseenScaffoldsForClusters), ``itemEmbeddings``.
Deck strand mixes, cross-source/media/week cluster restriction and prep blends
are rejected."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .build_state import TEACHING_ARC_TARGET_ITEMS, question_suppression_key
from .cluster_card_selection import select_cluster_cards
from .cluster_eligibility import eligible_rows, restricted
from .deck_strand_mix import DeckStrandOrderOptions, order_by_deck_strands
from .lane_utils import thread_trace
from .prep_allocation import PrepOrderOptions, order_session_prep_candidates
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


@dataclass
class ClusterScaffoldFilter:
    used: set[str]
    present: set[str]
    groups: set[str]


def _cluster_scaffold_candidate(
    filter_state: ClusterScaffoldFilter, candidates: list[dict[str, Any]], anchor: dict[str, Any]
) -> dict[str, Any] | None:
    return next(
        (c for c in candidates if c["id"] not in filter_state.used and c["id"] not in filter_state.present
         and (not c.get("variantGroupId") or c["variantGroupId"] not in filter_state.groups)
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


@dataclass(frozen=True)
class ClusterScaffoldOptions:
    rotation: str
    seed: str
    max_pairings: int | None


def _cluster_scaffold_filter(items: list[dict[str, Any]]) -> ClusterScaffoldFilter:
    return ClusterScaffoldFilter(
        used=set(), present={item["id"] for item in items},
        groups={item["variantGroupId"] for item in items if item["type"] == "card" and item.get("variantGroupId")},
    )


def _next_item_is_scaffold(items: list[dict[str, Any]], index: int) -> bool:
    next_item = items[index + 1] if index + 1 < len(items) else None
    return bool(next_item and next_item["type"] == "card" and next_item.get("complexity") == 1)


def _record_scaffold_pick(filter_state: ClusterScaffoldFilter, card: dict[str, Any]) -> None:
    filter_state.used.add(card["id"])
    if card.get("variantGroupId"):
        filter_state.groups.add(card["variantGroupId"])


def _cluster_scaffolds(
    items: list[dict[str, Any]], pool: list[dict[str, Any]], options: ClusterScaffoldOptions,
) -> list[dict[str, Any]]:
    from .queue import shuffle_with_seed

    max_pairings = options.max_pairings
    if max_pairings is None:
        max_pairings = _cluster_pairing_limit(items)
    if max_pairings <= 0 or not items or not pool:
        return items
    scaffolds = [card for card in pool if card.get("complexity") == 1]
    if not scaffolds:
        return items
    candidates = shuffle_with_seed(scaffolds, options.seed)
    filter_state = _cluster_scaffold_filter(items)
    inserts: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if len(inserts) >= max_pairings:
            break
        if not _cluster_anchor(item) or _next_item_is_scaffold(items, index):
            continue
        card = _cluster_scaffold_candidate(filter_state, candidates, item)
        if card is None:
            continue
        _record_scaffold_pick(filter_state, card)
        inserts[index] = _cluster_scaffold_item(item, card, options.rotation)
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


@dataclass(frozen=True)
class FirstSightQuotaRule:
    required: int
    size: int
    caps: dict[str, int]


def _concept_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item["conceptId"]] = counts.get(item["conceptId"], 0) + 1
    return counts


def _quota_candidate_allowed(candidate: dict[str, Any], key: tuple[str, str], state: dict[str, Any], rule: FirstSightQuotaRule) -> bool:
    if not candidate["firstSightAtSelection"] or key in state["keys"]:
        return False
    return state["per_concept"].get(candidate["conceptId"], 0) < rule.caps.get(candidate["conceptId"], 3)


def _add_quota_candidate(reconciled: list[dict[str, Any]], candidate: dict[str, Any], state: dict[str, Any]) -> None:
    reconciled.append(candidate)
    state["keys"].add((candidate["type"], candidate["id"]))
    state["per_concept"][candidate["conceptId"]] = state["per_concept"].get(candidate["conceptId"], 0) + 1
    state["first_sight"] += 1


def _reconcile_first_sight_quota(
    reconciled: list[dict[str, Any]], quota_candidates: list[dict[str, Any]], rule: FirstSightQuotaRule,
) -> None:
    state = {
        "keys": {(item["type"], item["id"]) for item in reconciled},
        "per_concept": _concept_counts(reconciled),
        "first_sight": sum(1 for item in reconciled if item.get("firstSightAtSelection")),
    }
    for candidate in quota_candidates:
        if state["first_sight"] >= rule.required:
            break
        key = (candidate["type"], candidate["id"])
        if not _quota_candidate_allowed(candidate, key, state, rule):
            continue
        if len(reconciled) >= rule.size and not _remove_non_first_sight(reconciled, state["keys"], state["per_concept"]):
            break
        _add_quota_candidate(reconciled, candidate, state)


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

    options = DeckStrandOrderOptions(
        seed=inputs["servingRotationSeed"], identify=identify,
        keep_with_previous=lambda item: item.get("interventionReason") == "preemptive_scaffold",
        share_basis="present",
    )
    return order_by_deck_strands(items, mix, options)


def _cluster_scaffold_rows(inputs: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    if ctx["maxNew"] == 0:
        return []
    return [
        row for row in eligible_rows(inputs, inputs.get("clusterScaffolds") or [])
        if row["id"] not in ctx["selCards"] and row.get("clusterId") in ctx["selectedClusters"]
    ]


def _cluster_teaching_caps(states: list[dict[str, Any]]) -> dict[str, int]:
    return {
        state["clusterId"]: TEACHING_ARC_TARGET_ITEMS[
            classify_teaching_state(state["cardsReviewed"], state["mastery"], state["mastery"])
        ]
        for state in states
    }


def _apply_cluster_caps_and_quota(inputs: dict[str, Any], ctx: dict[str, Any], paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    caps = _cluster_teaching_caps(ctx["states"])
    prep_keys = set((inputs["options"].get("sessionPrepBlend") or {}).get("itemIds") or [])
    reconciled = list(enforce_concept_teaching_caps(
        paired, lambda concept_id: caps.get(concept_id, 3),
        preserve=lambda item: f"{item['type']}:{item['id']}" in prep_keys,
    ))
    quota_rule = FirstSightQuotaRule(required=ctx["required"], size=ctx["size"], caps=caps)
    _reconcile_first_sight_quota(reconciled, ctx["quotaCandidates"], quota_rule)
    return reconciled


def _tag_cluster_first_sight(items: list[dict[str, Any]], unseen_scaffolds: set[str]) -> list[dict[str, Any]]:
    return [
        {**item, "firstSightAtSelection": item["firstSightAtSelection"] if item.get("firstSightAtSelection") is not None
         else item["id"] in unseen_scaffolds}
        for item in items
    ]


def _order_cluster_prep(inputs: dict[str, Any], items: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    options = PrepOrderOptions(
        size=size, key=lambda item: f"{item['type']}:{item['id']}",
        protected=lambda item: item.get("interventionReason") in {"failure_escalation", "needs_retest"},
    )
    return order_session_prep_candidates(items, inputs["options"].get("sessionPrepBlend"), options)


def _cluster_session_result(ctx: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    average = sum(item["priority"] for item in items) / len(items) if items else 0
    return {
        "items": items,
        "noveltyQuota": {
            "required": ctx["required"],
            "selected": sum(1 for item in items if item["firstSightAtSelection"]),
        },
        "stats": {
            "totalConcepts": len(ctx["states"]),
            "weakConcepts": sum(1 for state in ctx["states"] if state["priority"] > 0),
            "selectedConcepts": len(ctx["selectedClusters"]),
            "cardCount": sum(1 for item in items if item["type"] == "card"),
            "questionCount": sum(1 for item in items if item["type"] == "question"),
            "averagePriority": math.floor(average * 1000 + 0.5) / 1000,
            "conceptPairingRate": compute_concept_pairing_rate(items),
        },
    }


def _finish_cluster_session(inputs: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    items, required, size = ctx["items"], ctx["required"], ctx["size"]
    first_sight_q: set[str] = set()
    _append_cluster_questions(inputs, ctx, first_sight_q)
    ctx["quotaCandidates"] = list(items)
    _trim_to_first_sight_quota(items, required, size)
    ordered = order_by_manifold_walk(items, inputs.get("itemEmbeddings") or {})
    scaffolds = _cluster_scaffold_rows(inputs, ctx)
    scaffold_options = ClusterScaffoldOptions(
        rotation=inputs["rotation"], seed=inputs["servingRotationSeed"],
        max_pairings=0 if ctx["maxNew"] == 0 else None,
    )
    paired = _cluster_scaffolds(ordered, scaffolds, scaffold_options)
    _copy_scaffold_metadata(paired, scaffolds)
    reconciled = _apply_cluster_caps_and_quota(inputs, ctx, paired)
    final = break_modality_runs(_order_cluster_strands(inputs, reconciled))
    tagged = _tag_cluster_first_sight(final, {row["id"] for row in scaffolds})
    tagged = _order_cluster_prep(inputs, tagged, size)
    return _cluster_session_result(ctx, tagged)
