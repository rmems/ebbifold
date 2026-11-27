"""Cluster fallback card selection, including strand seats and prep targeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cluster_eligibility import eligible_rows
from .deck_strand_mix import deck_strand_of, strand_seat_targets
from .prep_allocation import PrepModalityBudgetRequest, prep_modality_budgets


def _configured_prep_clusters(plan: dict[str, Any], prep_ids: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item_id, cluster in (plan.get("cardClusters") or {}).items():
        if f"card:{item_id}" in prep_ids:
            out.setdefault(str(cluster), []).append(str(item_id))
    return out


def _derived_prep_clusters(rows: list[dict[str, Any]], prep_ids: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        key = f"card:{row['id']}"
        if key in prep_ids and row.get("clusterId"):
            out.setdefault(str(row["clusterId"]), []).append(str(row["id"]))
    return out


def _prep_cluster_map(rows: list[dict[str, Any]], plan: dict[str, Any], prep_ids: set[str]) -> dict[str, list[str]]:
    configured = _configured_prep_clusters(plan, prep_ids)
    return configured if plan.get("cardClusters") else _derived_prep_clusters(rows, prep_ids)


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


def _prep_budget(inputs: dict[str, Any], target_cards: int, prep_ids: set[str]) -> int:
    plan = inputs["options"].get("sessionPrepBlend") or {}
    supply = sum(1 for key in prep_ids if key.startswith("card:"))
    request = PrepModalityBudgetRequest(
        share=float(plan.get("share") or 0), card_seats=target_cards,
        question_seats=inputs["size"] - target_cards, card_supply=supply, question_supply=0,
    )
    return prep_modality_budgets(request)["cards"]


@dataclass
class ClusterCardSelection:
    inputs: dict[str, Any]
    states: list[dict[str, Any]]
    target_cards: int
    rows: list[dict[str, Any]]
    prep_ids: set[str]
    prep_weak_ids: set[str]
    prep_by_cluster: dict[str, list[str]]
    prep_seats: int
    mix: dict[str, Any] | None
    seats: dict[str, int]
    strand_counts: dict[str, int] = field(default_factory=dict)
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_items: list[dict[str, Any]] = field(default_factory=list)
    selected_ids: set[str] = field(default_factory=set)
    variant_groups: set[str] = field(default_factory=set)
    cluster_counts: dict[str, int] = field(default_factory=dict)
    selected_clusters: set[str] = field(default_factory=set)
    available_strands: set[str] = field(default_factory=set)

    def strand_open(self, row: dict[str, Any]) -> bool:
        if not self.mix or not row.get("firstSightAtSelection", True):
            return True
        strand = deck_strand_of(self.mix, row)
        return self.strand_counts.get(strand, 0) < self.seats.get(strand, 0)

    def wants_prep(self) -> bool:
        picked = sum(1 for item in self.selected_items if f"card:{item['id']}" in self.prep_ids)
        return int((len(self.selected_items) + 1) * self.prep_seats / max(1, self.target_cards)) > picked

    def _targeted_prep_rows(self, candidates: list[dict[str, Any]], target_ids: set[str]) -> list[dict[str, Any]]:
        present = {row["id"] for row in candidates}
        return [row for row in self.rows if row["id"] in target_ids and row["id"] not in present]

    def ordered_candidates(self, cluster_id: str, wants_prep: bool) -> list[dict[str, Any]]:
        candidates = [
            row for row in self.rows
            if str(row.get("clusterId")) == cluster_id and row["id"] not in self.selected_ids
        ][:12]
        target_ids = set(self.prep_by_cluster.get(cluster_id, []))
        if wants_prep and target_ids:
            candidates += self._targeted_prep_rows(candidates, target_ids)
        candidates = [row for row in candidates if row["id"] not in self.selected_ids]
        if not self.prep_ids:
            return candidates
        return sorted(candidates, key=lambda row: _prep_rank(row, self.prep_ids, self.prep_weak_ids, wants_prep))

    def note_available_strands(self, candidates: list[dict[str, Any]]) -> None:
        if not self.mix:
            return
        for row in candidates:
            if not row.get("variantGroupId") or row["variantGroupId"] not in self.variant_groups:
                self.available_strands.add(deck_strand_of(self.mix, row))

    def candidate_allowed(self, row: dict[str, Any]) -> bool:
        variant_free = not row.get("variantGroupId") or row["variantGroupId"] not in self.variant_groups
        return variant_free and self.strand_open(row)

    def record_pick(self, picked: dict[str, Any], cluster_id: str) -> None:
        if self.mix:
            strand = deck_strand_of(self.mix, picked)
            self.strand_counts[strand] = self.strand_counts.get(strand, 0) + 1
            self.identities[f"card:{picked['id']}"] = {"rotation": picked.get("rotation"), "sourceFile": picked.get("sourceFile")}
        self.selected_items.append(picked)
        self.selected_ids.add(picked["id"])
        if picked.get("variantGroupId"):
            self.variant_groups.add(picked["variantGroupId"])
        self.cluster_counts[cluster_id] = self.cluster_counts.get(cluster_id, 0) + 1
        self.selected_clusters.add(cluster_id)

    def select_from_cluster(self, remaining: list[dict[str, Any]], max_per_cluster: int) -> bool:
        cluster = _pick_cluster(remaining, self.wants_prep(), self.prep_by_cluster, self.selected_ids)
        cluster_id = str(cluster["clusterId"])
        if self.cluster_counts.get(cluster_id, 0) >= max_per_cluster:
            return False
        candidates = self.ordered_candidates(cluster_id, self.wants_prep())
        self.note_available_strands(candidates)
        picked = next((row for row in candidates if self.candidate_allowed(row)), None)
        if picked is None:
            return False
        self.record_pick(picked, cluster_id)
        return True

    def run_pass(self, pass_number: int) -> int:
        added = 0
        remaining = list(self.states)
        while remaining and len(self.selected_items) < self.target_cards:
            added += int(self.select_from_cluster(remaining, 2 + pass_number))
        return added

    def redistribute(self) -> bool:
        if not self.mix or not self.available_strands:
            return False
        extra = _redistribute_seats(self.mix, self.available_strands, self.target_cards - len(self.selected_items))
        seats = {
            str(strand.get("id")): self.strand_counts.get(str(strand.get("id")), 0) + extra.get(str(strand.get("id")), 0)
            for strand in self.mix.get("strands", [])
        }
        if seats == self.seats:
            return False
        self.seats = seats
        return True

    def run(self) -> None:
        pass_number = 0
        while len(self.selected_items) < self.target_cards:
            self.available_strands = set()
            if self.run_pass(pass_number) or self.redistribute():
                pass_number += 1
                continue
            break


def _wrap_card(inputs: dict[str, Any], cluster: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {"type": "card", "id": row["id"], "conceptId": cluster["clusterId"],
            "conceptName": cluster["clusterName"], "priority": cluster["priority"],
            "interventionReason": "weak_recall" if cluster["mastery"] < 0.6 else "reinforcement",
            "rotation": inputs["rotation"], "topics": row["topics"], "complexity": row.get("complexity"),
            "clusterId": row.get("clusterId"), "variantGroupId": row.get("variantGroupId"),
            "variantIndex": row.get("variantIndex"), "variantType": row.get("variantType"),
            "firstSightAtSelection": row["firstSightAtSelection"]}


def _selection_context(inputs: dict[str, Any], states: list[dict[str, Any]], target_cards: int) -> ClusterCardSelection:
    flat_rows = [row for cluster_rows in (inputs.get("clusterCards") or {}).values() for row in cluster_rows]
    rows = eligible_rows(inputs, flat_rows)
    plan = inputs["options"].get("sessionPrepBlend") or {}
    prep_ids = set(plan.get("itemIds") or [])
    mix = inputs.get("deckStrandMix")
    return ClusterCardSelection(
        inputs=inputs, states=states, target_cards=target_cards, rows=rows,
        prep_ids=prep_ids, prep_weak_ids=set(plan.get("weakItemIds") or []),
        prep_by_cluster=_prep_cluster_map(rows, plan, prep_ids),
        prep_seats=_prep_budget(inputs, target_cards, prep_ids), mix=mix,
        seats=strand_seat_targets(mix, target_cards) if mix else {},
        selected_ids=set(inputs["excludedCardIds"]),
    )


def select_cluster_cards(inputs: dict[str, Any], states: list[dict[str, Any]], target_cards: int) -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, Any]]]:
    selection = _selection_context(inputs, states, target_cards)
    selection.run()
    by_cluster = {str(state["clusterId"]): state for state in states}
    wrapped = [_wrap_card(inputs, by_cluster[str(row.get("clusterId"))], row) for row in selection.selected_items]
    return wrapped, selection.selected_clusters, selection.identities
