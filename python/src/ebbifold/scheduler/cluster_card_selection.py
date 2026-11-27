"""Cluster fallback card selection, including strand seats and prep targeting."""

from __future__ import annotations

from typing import Any

from .cluster_eligibility import eligible_rows
from .deck_strand_mix import deck_strand_of, strand_seat_targets
from .prep_allocation import prep_modality_budgets


def _prep_cluster_map(rows: list[dict[str, Any]], plan: dict[str, Any], prep_ids: set[str]) -> dict[str, list[str]]:
    configured = plan.get("cardClusters") or {}
    out: dict[str, list[str]] = {}
    for item_id, cluster in configured.items():
        if f"card:{item_id}" in prep_ids:
            out.setdefault(str(cluster), []).append(str(item_id))
    if configured:
        return out
    for row in rows:
        if f"card:{row['id']}" in prep_ids and row.get("clusterId"):
            out.setdefault(str(row["clusterId"]), []).append(str(row["id"]))
    return out


def _prep_rank(row: dict[str, Any], prep_ids: set[str], prep_weak_ids: set[str], wants_prep: bool) -> int:
    if not row.get("firstSightAtSelection", True):
        return 0
    key = f"card:{row['id']}"
    if (key in prep_ids) != wants_prep:
        return 3
    return 1 if wants_prep and key in prep_weak_ids else 2


def _pick_cluster(remaining: list[dict[str, Any]], wants_prep: bool, prep_by_cluster: dict[str, list[str]], selected: set[str]) -> dict[str, Any]:
    index = 0
    if wants_prep:
        index = next(
            (i for i, cluster in enumerate(remaining)
             if any(item_id not in selected for item_id in prep_by_cluster.get(str(cluster["clusterId"]), []))),
            -1,
        )
    return remaining.pop(index if index >= 0 else 0)


def _redistribute_seats(mix: dict[str, Any], available: set[str], remaining: int) -> dict[str, int]:
    strands = [strand for strand in mix.get("strands", []) if str(strand.get("id")) in available]
    total = sum(float(strand.get("share", 0)) for strand in strands)
    if total <= 0:
        return {}
    normalized = {"strands": [{**strand, "share": float(strand.get("share", 0)) / total} for strand in strands]}
    return strand_seat_targets(normalized, remaining)


def _ordered_candidates(rows: list[dict[str, Any]], cluster_id: str, prep_ids: set[str], prep_weak_ids: set[str], wants_prep: bool, prep_target_ids: set[str]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("clusterId")) == cluster_id][:12]
    if wants_prep and prep_target_ids:
        present = {row["id"] for row in candidates}
        targeted = [row for row in rows if row["id"] in prep_target_ids and row["id"] not in present]
        candidates = [*candidates, *targeted]
    if not prep_ids:
        return candidates
    return sorted(candidates, key=lambda row: _prep_rank(row, prep_ids, prep_weak_ids, wants_prep))


def select_cluster_cards(inputs: dict[str, Any], states: list[dict[str, Any]], target_cards: int) -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, Any]]]:
    flat_rows = [row for cluster_rows in (inputs.get("clusterCards") or {}).values() for row in cluster_rows]
    rows = eligible_rows(inputs, flat_rows)
    plan = inputs["options"].get("sessionPrepBlend") or {}
    prep_ids = set(plan.get("itemIds") or [])
    prep_weak_ids = set(plan.get("weakItemIds") or [])
    prep_card_supply = sum(1 for key in prep_ids if key.startswith("card:"))
    prep_seats = prep_modality_budgets(
        float(plan.get("share") or 0), target_cards, inputs["size"] - target_cards, prep_card_supply, 0,
    )["cards"]
    prep_by_cluster = _prep_cluster_map(rows, plan, prep_ids)
    mix = inputs.get("deckStrandMix")
    seats = strand_seat_targets(mix, target_cards) if mix else {}
    strand_counts: dict[str, int] = {}
    identities: dict[str, dict[str, Any]] = {}
    selected_items: list[dict[str, Any]] = []
    selected_ids = set(inputs["excludedCardIds"])
    variant_groups: set[str] = set()
    cluster_counts: dict[str, int] = {}
    selected_clusters: set[str] = set()
    pass_number = 0

    def strand_open(row: dict[str, Any]) -> bool:
        if not mix or not row.get("firstSightAtSelection", True):
            return True
        strand = deck_strand_of(mix, row)
        return strand_counts.get(strand, 0) < seats.get(strand, 0)

    while len(selected_items) < target_cards:
        max_per_cluster = 2 + pass_number
        added = 0
        available_strands: set[str] = set()
        remaining = list(states)
        while remaining and len(selected_items) < target_cards:
            prep_picked = sum(1 for item in selected_items if f"card:{item['id']}" in prep_ids)
            wants_prep = int((len(selected_items) + 1) * prep_seats / max(1, target_cards)) > prep_picked
            cluster = _pick_cluster(remaining, wants_prep, prep_by_cluster, selected_ids)
            cluster_id = str(cluster["clusterId"])
            if cluster_counts.get(cluster_id, 0) >= max_per_cluster:
                continue
            target_ids = set(prep_by_cluster.get(cluster_id, []))
            candidates = _ordered_candidates(rows, cluster_id, prep_ids, prep_weak_ids, wants_prep, target_ids)
            candidates = [row for row in candidates if row["id"] not in selected_ids]
            if mix:
                for row in candidates:
                    if not row.get("variantGroupId") or row["variantGroupId"] not in variant_groups:
                        available_strands.add(deck_strand_of(mix, row))
            picked = next((
                row for row in candidates
                if (not row.get("variantGroupId") or row["variantGroupId"] not in variant_groups) and strand_open(row)
            ), None)
            if picked is None:
                continue
            if mix:
                strand = deck_strand_of(mix, picked)
                strand_counts[strand] = strand_counts.get(strand, 0) + 1
                identities[f"card:{picked['id']}"] = {"rotation": picked.get("rotation"), "sourceFile": picked.get("sourceFile")}
            selected_items.append(picked)
            selected_ids.add(picked["id"])
            if picked.get("variantGroupId"):
                variant_groups.add(picked["variantGroupId"])
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            selected_clusters.add(cluster_id)
            added += 1
        if added:
            pass_number += 1
            continue
        if mix and available_strands:
            extra = _redistribute_seats(mix, available_strands, target_cards - len(selected_items))
            seats = {str(strand.get("id")): strand_counts.get(str(strand.get("id")), 0) + extra.get(str(strand.get("id")), 0) for strand in mix.get("strands", [])}
            pass_number += 1
            continue
        break
    wrapped = []
    for row in selected_items:
        cluster = next(state for state in states if str(state["clusterId"]) == str(row.get("clusterId")))
        wrapped.append({"type": "card", "id": row["id"], "conceptId": cluster["clusterId"],
                        "conceptName": cluster["clusterName"], "priority": cluster["priority"],
                        "interventionReason": "weak_recall" if cluster["mastery"] < 0.6 else "reinforcement",
                        "rotation": inputs["rotation"], "topics": row["topics"], "complexity": row.get("complexity"),
                        "clusterId": row.get("clusterId"), "variantGroupId": row.get("variantGroupId"),
                        "variantIndex": row.get("variantIndex"), "variantType": row.get("variantType"),
                        "firstSightAtSelection": row["firstSightAtSelection"]})
    return wrapped, selected_clusters, identities
