"""Queue phase (unified-scheduler-phase-queue.ts) and its ordering helpers.

Item embeddings and the flow axis are storage-bound; they arrive as explicit
snapshot inputs (``itemEmbeddings`` keyed by item id, ``flowAxis``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .math import cosine_similarity

WALK_DEFAULTS = {
    "sweetSpotMin": 0.4, "sweetSpotMax": 0.7, "maxConsecutiveSameType": 2, "hardMaxConsecutiveSameType": 4,
    "minItemsForWalk": 3, "flowWeight": 0.15, "conceptPairWeight": -0.35,
}
CONCEPT_CADENCE_WINDOW = 6
LEGACY_CONCEPT_PAIR_WINDOW = 3


def _walk_seed(embedded: list[dict[str, Any]]) -> dict[str, Any]:
    seed = embedded[0]
    for item in embedded[1:]:
        if item["priority"] > seed["priority"]:
            seed = item
    return seed


def _walk_similarity_score(similarity: float) -> float:
    lo, hi = WALK_DEFAULTS["sweetSpotMin"], WALK_DEFAULTS["sweetSpotMax"]
    if lo <= similarity <= hi:
        return 1.0 - (abs(similarity - (lo + hi) / 2) / ((hi - lo) / 2)) * 0.3
    if similarity > hi:
        return 0.7 - (similarity - hi) * 2.0
    return 0.7 - (lo - similarity) * 0.5


def _walk_flow_score(current: list[float], candidate: list[float], flow_axis: list[float] | None) -> float:
    if not flow_axis:
        return 0.0
    cur_dot = cand_dot = 0.0
    for d in range(min(len(current), len(flow_axis))):
        cur_dot += current[d] * flow_axis[d]
        cand_dot += candidate[d] * flow_axis[d]
    return math.tanh((cand_dot - cur_dot) * 3) * WALK_DEFAULTS["flowWeight"]


def _walk_concept_proximity(walk: list[dict[str, Any]], candidate: dict[str, Any]) -> float:
    weight = WALK_DEFAULTS["conceptPairWeight"]
    if weight == 0 or not candidate.get("conceptId"):
        return 0.0
    cadence = weight < 0
    window = CONCEPT_CADENCE_WINDOW if cadence else LEGACY_CONCEPT_PAIR_WINDOW
    for k in range(1, min(window, len(walk)) + 1):
        if walk[-k].get("conceptId") == candidate["conceptId"]:
            return weight * ((window - k + 1) / window) if cadence else weight / k
    return 0.0


def _walk_candidate_score(
    walk: list[dict[str, Any]], candidate: dict[str, Any], current: list[float], centroid: list[float],
    embeddings: dict[str, list[float]], flow_axis: list[float] | None, consecutive: int,
) -> float:
    emb = embeddings[candidate["id"]]
    sim_score = _walk_similarity_score(cosine_similarity(current, emb))
    coverage = (1.0 - cosine_similarity(emb, centroid)) * 0.3
    same_type_capped = candidate["type"] == walk[-1]["type"] and consecutive >= WALK_DEFAULTS["maxConsecutiveSameType"]
    type_penalty = -0.5 if same_type_capped else 0
    flow = _walk_flow_score(current, emb, flow_axis)
    proximity = _walk_concept_proximity(walk, candidate)
    return sim_score + coverage + type_penalty + flow + proximity


def _update_centroid(centroid: list[float], new: list[float], count: int) -> None:
    for d in range(len(centroid)):
        centroid[d] = centroid[d] * ((count - 1) / count) + new[d] / count


def order_by_manifold_walk(items: list[dict[str, Any]], embeddings: dict[str, list[float]],
                           flow_axis: list[float] | None = None) -> list[dict[str, Any]]:
    embedded = [i for i in items if i["id"] in embeddings]
    unembedded = [i for i in items if i["id"] not in embeddings]
    if len(embedded) < WALK_DEFAULTS["minItemsForWalk"]:
        return list(items)
    seed = _walk_seed(embedded)
    visited = {seed["id"]}
    walk = [seed]
    centroid = list(embeddings[seed["id"]])
    consecutive = 1
    while len(walk) < len(embedded):
        current_type = walk[-1]["type"]
        hard_cap = consecutive >= WALK_DEFAULTS["hardMaxConsecutiveSameType"]
        require_switch = hard_cap and any(c["id"] not in visited and c["type"] != current_type for c in embedded)
        best_score, best = -math.inf, None
        for cand in embedded:
            if cand["id"] in visited or (require_switch and cand["type"] == current_type):
                continue
            score = _walk_candidate_score(
                walk, cand, embeddings[walk[-1]["id"]], centroid, embeddings, flow_axis, consecutive
            )
            if score > best_score:
                best_score, best = score, cand
        if best is None:
            break
        visited.add(best["id"])
        walk.append(best)
        consecutive = consecutive + 1 if best["type"] == current_type else 1
        _update_centroid(centroid, embeddings[best["id"]], len(walk))
    return distribute_evenly(walk, unembedded) if unembedded else walk


def distribute_evenly(base: list[Any], extras: list[Any]) -> list[Any]:
    if not extras:
        return list(base)
    if not base:
        return list(extras)
    result: list[Any] = []
    interval = (len(base) + len(extras)) / len(extras)
    extra_idx = 0
    next_at = math.floor(interval / 2)
    for item in base:
        result.append(item)
        if extra_idx < len(extras) and len(result) > next_at:
            result.append(extras[extra_idx])
            extra_idx += 1
            next_at = math.floor(interval / 2 + interval * extra_idx)
    result.extend(extras[extra_idx:])
    return result


def compute_concept_pairing_rate(items: list[dict[str, Any]]) -> float | None:
    positions: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if item.get("conceptId"):
            positions.setdefault(item["conceptId"], []).append(i)
    multi = [p for p in positions.values() if len(p) >= 2]
    if not multi:
        return None
    paired = sum(1 for p in multi if any(p[j] - p[i] <= 3 for i in range(len(p) - 1) for j in range(i + 1, len(p))))
    return paired / len(multi)


RHYTHM_PARAMS = {
    "visitor": (4, 0.2), "browser": (6, 0.3), "student": (10, 0.5), "committed": (16, 0.6), "superfan": (24, 0.7),
}
_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


def get_difficulty_bias(position: int, level: str) -> float:
    wavelength, amplitude = RHYTHM_PARAMS.get(level, RHYTHM_PARAMS["browser"])
    return amplitude * math.sin((2 * math.pi * position) / wavelength)


def apply_difficulty_rhythm(items: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    if len(items) < 3:
        return list(items)
    if RHYTHM_PARAMS.get(level, RHYTHM_PARAMS["browser"])[1] < 0.25:
        return list(items)
    result = list(items)

    def fit(rank: int, bias: float) -> float:
        return -abs(rank - 1 - bias)

    for i in range(len(result) - 1):
        here, nxt = get_difficulty_bias(i, level), get_difficulty_bias(i + 1, level)
        rh = _DIFFICULTY_RANK.get(result[i].get("difficulty") or "medium", 1)
        rn = _DIFFICULTY_RANK.get(result[i + 1].get("difficulty") or "medium", 1)
        if fit(rn, here) + fit(rh, nxt) > fit(rh, here) + fit(rn, nxt):
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


MAX_CROSS_SOURCE_ITEMS_PER_SESSION = 2


def source_corpus_for(rotation: str | None, source_file: str | None = None) -> str:
    value = rotation or "unknown"
    if value == "anatomy":
        return "bluelink"
    if value == "surgical-sciences":
        return "kubie" if (source_file or "").startswith("kubie-") else "plates"
    if value == "neurosurg":
        return "kubie"
    return value


def cap_cross_source_session_items(items: list[dict[str, Any]], *, session_rotation: str, allowed: list[str],
                                   max_items: float | None, get_rotation: Any, get_source_key: Any = None):
    sources = {r for r in allowed if r and r != session_rotation}
    if max_items is None:
        limit = MAX_CROSS_SOURCE_ITEMS_PER_SESSION
    elif not math.isfinite(max_items):
        limit = 0
    else:
        limit = max(0, math.floor(max_items))

    def admissible(item: dict[str, Any]) -> bool:
        r = get_rotation(item)
        return r != session_rotation and r in sources

    if get_source_key is not None and limit > 0:
        queues: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            if admissible(item):
                queues.setdefault(get_source_key(item), []).append(index)
        seated: set[int] = set()
        rings = list(queues.values())
        cursor = 0
        while len(seated) < limit and any(rings):
            ring = rings[cursor % len(rings)]
            if ring:
                seated.add(ring.pop(0))
            cursor += 1
        return [it for i, it in enumerate(items)
                if (i in seated if admissible(it) else get_rotation(it) == session_rotation)]
    emitted = 0
    out = []
    for item in items:
        r = get_rotation(item)
        if r == session_rotation:
            out.append(item)
        elif r in sources and emitted < limit:
            emitted += 1
            out.append(item)
    return out


MODALITY_MAX_SAME_TYPE_RUN = 5


def max_same_type_run(items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    best = run = 1
    for i in range(1, len(items)):
        run = run + 1 if items[i]["type"] == items[i - 1]["type"] else 1
        best = max(best, run)
    return best


def _stride_redistribute(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(items)
    if total <= MODALITY_MAX_SAME_TYPE_RUN:
        return list(items)
    counts: dict[str, int] = {}
    for it in items:
        counts[it["type"]] = counts.get(it["type"], 0) + 1
    if len(counts) <= 1:
        return list(items)
    maj, maj_count = items[0]["type"], counts[items[0]["type"]]
    for t, c in counts.items():
        if c > maj_count:
            maj, maj_count = t, c
    maj_q = [it for it in items if it["type"] == maj]
    min_q = [it for it in items if it["type"] != maj]
    m, k = len(maj_q), len(min_q)
    result: list[Any] = [None] * total
    for i in range(k):
        clamped = max(0, min(total - 1, math.ceil(((i + 1) * m) / (k + 1)) + i))
        p = clamped
        while p < total and result[p] is not None:
            p += 1
        if p >= total:
            p = clamped - 1
            while p >= 0 and result[p] is not None:
                p -= 1
        if p < 0:
            p = next(q for q in range(total) if result[q] is None)
        result[p] = min_q[i]
    it_maj = iter(maj_q)
    return [r if r is not None else next(it_maj) for r in result]


def break_modality_runs(items: list[dict[str, Any]], get_ordering_group: Any = None) -> list[dict[str, Any]]:
    if get_ordering_group is not None and items:
        out: list[dict[str, Any]] = []
        start = 0
        while start < len(items):
            group = get_ordering_group(items[start])
            end = start + 1
            while end < len(items) and get_ordering_group(items[end]) == group:
                end += 1
            out.extend(break_modality_runs(items[start:end]))
            start = end
        return out
    if len(items) <= MODALITY_MAX_SAME_TYPE_RUN:
        return list(items)
    result = list(items)
    for i in range(len(result)):
        if i < MODALITY_MAX_SAME_TYPE_RUN - 1:
            continue
        run_type = result[i]["type"]
        run = 1
        j = i - 1
        while j >= 0 and result[j]["type"] == run_type:
            run += 1
            j -= 1
        if run < MODALITY_MAX_SAME_TYPE_RUN:
            continue
        if i + 1 >= len(result):
            break
        if result[i + 1]["type"] != run_type:
            continue
        swap = next((k for k in range(i + 2, len(result)) if result[k]["type"] != run_type), -1)
        if swap == -1:
            break
        result[i + 1], result[swap] = result[swap], result[i + 1]
    if max_same_type_run(result) > MODALITY_MAX_SAME_TYPE_RUN:
        return _stride_redistribute(items)
    return result


def hash_string(value: str) -> int:
    h = 0
    data = value.encode("utf-16-le")
    for i in range(0, len(data), 2):
        h = ((h << 5) - h + (data[i] | (data[i + 1] << 8))) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x1_0000_0000
    return abs(h)


def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def shuffle_with_seed(items: list[Any], seed: str) -> list[Any]:
    state = hash_string(seed)

    def rng() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(state ^ (state >> 15), state | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = math.floor(rng() * (i + 1))
        result[i], result[j] = result[j], result[i]
    return result


def _specific(topics: list[str] | None) -> set[str]:
    from .policies import specific_clinical_topics

    return set(specific_clinical_topics(topics or []))


def _is_anchor(item: dict[str, Any]) -> bool:
    if item.get("interventionReason") == "concept_followup":
        return False
    if item["type"] == "question":
        return bool(_specific(item.get("topics")))
    if item["type"] == "card":
        c = item.get("complexity")
        return c is not None and c >= 2 and bool(_specific(item.get("topics")))
    return False


def _default_pairing_limit(items: list[dict[str, Any]]) -> int:
    anchors = sum(1 for it in items if _is_anchor(it))
    return max(2, min(max(2, anchors), max(2, len(items) // 3)))


def _find_preemptive_scaffold(
    candidates: list[dict[str, Any]], used: set[str], present: set[str], groups: set[str], left: set[str]
) -> dict[str, Any] | None:
    return next(
        (
            c for c in candidates
            if c["id"] not in used and c["id"] not in present
            and (not c.get("variantGroupId") or c["variantGroupId"] not in groups)
            and bool(left) and any(t in left for t in _specific(c["topics"]))
        ),
        None,
    )


def _preemptive_scaffold_item(item: dict[str, Any], card: dict[str, Any], rotation: str) -> dict[str, Any]:
    return {
        "type": "card", "id": card["id"], "rotation": item["rotation"] if item.get("rotation") is not None else rotation,
        "priority": 0.5, "conceptId": item.get("conceptId") or "", "conceptName": item.get("conceptName") or "",
        "topics": card["topics"], "complexity": card.get("complexity"), "clusterId": card.get("clusterId"),
        "variantGroupId": card.get("variantGroupId"), "interventionReason": "preemptive_scaffold",
        "struggleIntervention": {"strategy": "preemptive", "isScaffold": True, "targetCardId": item["id"]},
    }


def _default_scaffold_pairing_limit(items: list[dict[str, Any]], max_pairings: int | None) -> int:
    if max_pairings is not None:
        return max_pairings
    return _default_pairing_limit(items)


def _preemptive_scaffold_candidates(bulk: dict[str, Any], rotation_seed: str,
                                    excluded_card_ids: set[str] | None) -> list[dict[str, Any]]:
    excluded = excluded_card_ids or set()
    pool = [card for card in bulk["unseenCards"] if card["id"] not in excluded]
    scaffolds = [card for card in pool if card.get("complexity") == 1]
    return shuffle_with_seed(scaffolds, rotation_seed) if rotation_seed else scaffolds


def _anchor_needs_scaffold(items: list[dict[str, Any]], index: int, step_down_concept_ids: set[str]) -> bool:
    item = items[index]
    if not _is_anchor(item) or (item.get("conceptId") or "") not in step_down_concept_ids:
        return False
    nxt = items[index + 1] if index + 1 < len(items) else None
    return not (nxt and nxt["type"] == "card" and nxt.get("complexity") == 1)


def _collect_preemptive_insertions(items: list[dict[str, Any]], candidates: list[dict[str, Any]],
                                   step_down_concept_ids: set[str], max_pairings: int,
                                   rotation: str) -> dict[int, dict[str, Any]]:
    used: set[str] = set()
    present = {item["id"] for item in items}
    groups = {item["variantGroupId"] for item in items if item["type"] == "card" and item.get("variantGroupId")}
    insertions: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if len(insertions) >= max_pairings:
            break
        if not _anchor_needs_scaffold(items, index, step_down_concept_ids):
            continue
        card = _find_preemptive_scaffold(candidates, used, present, groups, _specific(item.get("topics")))
        if card is None:
            continue
        used.add(card["id"])
        if card.get("variantGroupId"):
            groups.add(card["variantGroupId"])
        insertions[index] = _preemptive_scaffold_item(item, card, rotation)
    return insertions


def _insert_scaffolds(items: list[dict[str, Any]], insertions: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        out.append(item)
        if index in insertions:
            out.append(insertions[index])
    return out


def inject_preemptive_scaffolds(items: list[dict[str, Any]], bulk: dict[str, Any], *, rotation_seed: str,
                                step_down_concept_ids: set[str], max_pairings: int | None,
                                excluded_card_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not step_down_concept_ids:
        return items
    max_pairings = _default_scaffold_pairing_limit(items, max_pairings)
    candidates = _preemptive_scaffold_candidates(bulk, rotation_seed, excluded_card_ids)
    if max_pairings <= 0 or not items or not candidates:
        return items
    insertions = _collect_preemptive_insertions(items, candidates, step_down_concept_ids, max_pairings, bulk["rotation"])
    return _insert_scaffolds(items, insertions) if insertions else items


def apply_struggle_to_selection(inputs: dict[str, Any], build: Any) -> list[dict[str, Any]]:
    from .struggle import apply_struggle_interventions

    return apply_struggle_interventions(inputs, build.selected_items)


def _mastery_assignment_allowed(assignment: dict[str, Any], *, quota_satisfied: bool,
                                eligible_core: set[str], eligible_applied: set[str]) -> bool:
    stage, unit_id = assignment.get("stage"), assignment.get("unitId")
    if stage == "scheduled-atomic-core":
        return unit_id in eligible_core
    if stage == "applied-distinction":
        return quota_satisfied and unit_id in eligible_applied
    return stage == "breadth-exploration" and quota_satisfied


def _annotate_mastery_assignments(inputs: dict[str, Any], build: Any, items: list[dict[str, Any]]) -> None:
    evidence = inputs.get("masteryEvidence")
    if not (inputs.get("applyExamTarget") and evidence):
        return
    assignments = evidence.get("itemStageMap") or {}
    eligible_core = set(inputs.get("eligibleCoreUnitIds") or [])
    eligible_applied = set(inputs.get("eligibleAppliedUnitIds") or [])
    for item in items:
        if item["type"] not in ("card", "question"):
            continue
        assignment = assignments.get(f"{item['type']}:{item['id']}")
        if not assignment:
            continue
        if _mastery_assignment_allowed(assignment, quota_satisfied=build.core_target_quota_satisfied,
                                       eligible_core=eligible_core, eligible_applied=eligible_applied):
            item["examTargetMasteryStage"] = assignment.get("stage")
            item["examTargetMasteryUnitId"] = assignment.get("unitId")


def _group_mastery_items(items: list[dict[str, Any]], rank_item: Any) -> list[dict[str, Any]]:
    scaffold_indexes: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        intervention = item.get("struggleIntervention") or {}
        if intervention.get("isScaffold") and intervention.get("targetCardId"):
            scaffold_indexes.setdefault(intervention["targetCardId"], []).append(index)
    grouped_indexes: set[int] = set()
    groups: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        intervention = item.get("struggleIntervention") or {}
        if index in grouped_indexes or intervention.get("isScaffold"):
            continue
        scaffolds = scaffold_indexes.get(item["id"], [])
        grouped_indexes.add(index)
        grouped_indexes.update(scaffolds)
        groups.append({"rank": rank_item(item), "originalIndex": index,
                       "items": [item] + [items[scaffold] for scaffold in scaffolds]})
    for index, item in enumerate(items):
        if index not in grouped_indexes:
            groups.append({"rank": rank_item(item), "originalIndex": index, "items": [item]})
    groups.sort(key=lambda group: (group["rank"], group["originalIndex"]))
    return [item for group in groups for item in group["items"]]


def _augment_card_metadata(item: dict[str, Any], card_meta: dict[str, Any]) -> None:
    for key in ("topics", "complexity", "clusterId", "variantGroupId"):
        if item.get(key) is None:
            item[key] = card_meta[key]


def _augment_question_metadata(item: dict[str, Any], question: dict[str, Any], level: int) -> None:
    from .difficulty import get_effective_question_difficulty

    if item.get("topics") is None:
        item["topics"] = question["topics"]
    if level != 0:
        item["difficulty"] = get_effective_question_difficulty(
            question.get("difficulty"), question.get("facilityIndex"), question.get("totalAttempts")
        )


def _augment_order_metadata(items: list[dict[str, Any]], meta: dict[str, dict[str, Any]],
                            question_map: dict[str, dict[str, Any]], level: int) -> None:
    for item in items:
        if item["type"] == "card" and item["id"] in meta:
            _augment_card_metadata(item, meta[item["id"]])
        elif item["type"] == "question" and item["id"] in question_map:
            _augment_question_metadata(item, question_map[item["id"]], level)


def _mastery_ranker(card_rot: dict[str, str | None], q_rot: dict[str, str | None], rotation: str) -> Any:
    def rank(item: dict[str, Any]) -> int:
        stage = item.get("examTargetMasteryStage")
        if stage == "scheduled-atomic-core":
            return 0
        if stage == "applied-distinction":
            return 1
        item_rotation = item.get("rotation")
        if item_rotation is None and item["type"] == "card":
            item_rotation = card_rot.get(item["id"])
        if item_rotation is None and item["type"] == "question":
            item_rotation = q_rot.get(item["id"])
        if item_rotation is None:
            item_rotation = rotation
        if stage == "breadth-exploration":
            return 3
        return 2 if item_rotation == rotation else 3
    return rank


def _endpoint_filter(question_map: dict[str, dict[str, Any]], hard_gap: set[str],
                     complexity: dict[str, Any], policy: dict[str, Any]) -> Any:
    from .policies import modality_eligible_for_review_learning

    def endpoint(item: dict[str, Any]) -> bool:
        if not modality_eligible_for_review_learning(item["type"], policy["level"]):
            return False
        if policy["questionsOnly"]:
            question = question_map.get(item["id"]) if item["type"] == "question" else None
            if not question or question.get("difficulty") != "hard" or item["id"] not in hard_gap:
                return False
        if policy["scaffoldsOnly"]:
            return item["type"] == "card" and complexity.get(item["id"]) == 1
        return True
    return endpoint


def _recompute_mastery_seat_counts(build: Any, final: list[dict[str, Any]]) -> None:
    core_units = {item.get("examTargetMasteryUnitId") for item in final
                  if item.get("examTargetMasteryStage") == "scheduled-atomic-core" and item.get("examTargetMasteryUnitId")}
    surplus_units = {item.get("examTargetMasteryUnitId") for item in final
                     if item.get("examTargetMasteryStage") in ("applied-distinction", "breadth-exploration")
                     and item.get("examTargetMasteryUnitId")}
    build.selected_core_target_seats = len(core_units)
    build.selected_surplus_target_seats = len(surplus_units)


@dataclass(frozen=True)
class SessionOrderingContext:
    inputs: dict[str, Any]
    plan: dict[str, Any]
    build: Any
    options: dict[str, Any]
    bulk: dict[str, Any]
    rotation: str
    level: int
    policy: dict[str, Any]
    hard_gap: set[str]
    cards: list[dict[str, Any]]
    complexity: dict[str, Any]
    meta: dict[str, dict[str, Any]]
    card_rotations: dict[str, str | None]
    question_rotations: dict[str, str | None]


def _card_ordering_metadata(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        card["id"]: {
            "stableId": card.get("stableId"), "topics": card["topics"], "complexity": card.get("complexity"),
            "clusterId": card.get("clusterId"), "variantGroupId": card.get("variantGroupId"),
            "facilityIndex": card.get("facilityIndex"), "sampleSize": card.get("sampleSize"),
            "similarCards": card.get("similarCards"),
        }
        for card in cards
    }


def _session_ordering_context(inputs: dict[str, Any], plan: dict[str, Any], build: Any) -> SessionOrderingContext:
    from .policies import review_learning_policy
    from .scoring import normalize_review_challenge_level

    options, bulk, rotation = inputs["options"], inputs["bulk"], inputs["rotation"]
    level = normalize_review_challenge_level(options.get("reviewChallenge"))
    policy = review_learning_policy(options.get("reviewChallenge"))
    hard_gap = {link["questionId"] for link in bulk["questionConceptLinks"] if link["questionId"] in bulk["questionMap"]} \
        if policy["questionsOnly"] else set()
    cards = bulk["unseenCards"] + bulk["seenCards"]
    return SessionOrderingContext(
        inputs=inputs, plan=plan, build=build, options=options, bulk=bulk, rotation=rotation,
        level=level, policy=policy, hard_gap=hard_gap, cards=cards,
        complexity={card["id"]: card.get("complexity") for card in cards}, meta=_card_ordering_metadata(cards),
        card_rotations={card["id"]: card.get("rotation") for card in cards},
        question_rotations={question["id"]: question.get("rotation") for question in bulk["rotationQuestions"]},
    )


def _priority_difficulty(priority: float) -> str:
    if priority > 0.66:
        return "hard"
    if priority > 0.33:
        return "medium"
    return "easy"


def _initial_session_order(context: SessionOrderingContext, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flow = context.inputs.get("flowAxis") or None
    ordered = order_by_manifold_walk(items, context.inputs.get("itemEmbeddings") or {}, flow)
    for item in ordered:
        item["difficulty"] = _priority_difficulty(item["priority"])
    rhythm = ordered
    if context.level == 0:
        rhythm = apply_difficulty_rhythm(ordered, context.options.get("commitmentLevel") or "browser")
    return break_modality_runs(rhythm)


def _pair_preemptive_scaffolds(context: SessionOrderingContext, mono: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return inject_preemptive_scaffolds(
        mono, context.bulk, rotation_seed=context.inputs["servingRotationSeed"],
        step_down_concept_ids=context.inputs["recentFailureConceptIds"],
        max_pairings=0 if context.inputs.get("effectiveMaxNewCards") == 0 else None,
        excluded_card_ids=context.build.selected_card_ids,
    )


def _cap_endpoint_items(context: SessionOrderingContext, paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .scoring import enforce_concept_teaching_caps

    endpoint = _endpoint_filter(context.bulk["questionMap"], context.hard_gap, context.complexity, context.policy)
    capped = enforce_concept_teaching_caps([item for item in paired if endpoint(item)], context.build.teaching_cap_for)
    _annotate_mastery_assignments(context.inputs, context.build, capped)
    return capped


def _rotation_accessors(context: SessionOrderingContext) -> tuple[Any, Any]:
    unknown = context.plan["unknownItemRotation"]
    card_rotations = context.card_rotations
    question_rotations = context.question_rotations

    def get_rotation(item: dict[str, Any]) -> str:
        if item.get("rotation"):
            return item["rotation"]
        if item["type"] == "card":
            return card_rotations.get(item["id"]) or unknown
        if item["type"] == "question":
            return question_rotations.get(item["id"]) or unknown
        return unknown

    card_source = {card["id"]: card.get("sourceFile") for card in context.cards}

    def get_source_key(item: dict[str, Any]) -> str:
        item_rotation = item.get("rotation") or (
            card_rotations.get(item["id"]) if item["type"] == "card" else question_rotations.get(item["id"])
        )
        return source_corpus_for(item_rotation, card_source.get(item["id"]) if item["type"] == "card" else None)

    return get_rotation, get_source_key


def _cap_cross_sources(context: SessionOrderingContext, capped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    get_rotation, get_source_key = _rotation_accessors(context)
    return cap_cross_source_session_items(
        capped, session_rotation=context.rotation, allowed=context.options.get("crossSourceRotations") or [],
        max_items=context.options.get("maxCrossSourceItems"), get_rotation=get_rotation,
        get_source_key=get_source_key if context.inputs.get("composedDeck") else None,
    )


def _finish_session_order(context: SessionOrderingContext, source_capped: list[dict[str, Any]],
                          mastery_stage_rank: Any) -> list[dict[str, Any]]:
    from .policies import apply_review_challenge_wave

    ordering_group = mastery_stage_rank if context.inputs.get("applyExamTarget") else None
    final = break_modality_runs(source_capped, ordering_group)
    if context.level != 0:
        final = apply_review_challenge_wave(final, context.level)
    if context.inputs.get("applyExamTarget"):
        _recompute_mastery_seat_counts(context.build, final)
    return final


def order_session_items(inputs: dict[str, Any], plan: dict[str, Any], build: Any,
                        items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    context = _session_ordering_context(inputs, plan, build)
    mono = _initial_session_order(context, items)
    _augment_order_metadata(mono, context.meta, context.bulk["questionMap"], context.level)
    paired = _pair_preemptive_scaffolds(context, mono)
    capped = _cap_endpoint_items(context, paired)
    mastery_stage_rank = _mastery_ranker(context.card_rotations, context.question_rotations, context.rotation)
    if inputs.get("applyExamTarget"):
        capped = _group_mastery_items(capped, mastery_stage_rank)
    source_capped = _cap_cross_sources(context, capped)
    return _finish_session_order(context, source_capped, mastery_stage_rank), context.meta
