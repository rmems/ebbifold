"""Queue phase (unified-scheduler-phase-queue.ts) and its ordering helpers.

Item embeddings and the flow axis are storage-bound; they arrive as explicit
snapshot inputs (``itemEmbeddings`` keyed by item id, ``flowAxis``).
"""

from __future__ import annotations

import math
from typing import Any

from .math import cosine_similarity

WALK_DEFAULTS = {
    "sweetSpotMin": 0.4, "sweetSpotMax": 0.7, "maxConsecutiveSameType": 2, "hardMaxConsecutiveSameType": 4,
    "minItemsForWalk": 3, "flowWeight": 0.15, "conceptPairWeight": -0.35,
}
CONCEPT_CADENCE_WINDOW = 6
LEGACY_CONCEPT_PAIR_WINDOW = 3


def order_by_manifold_walk(items: list[dict[str, Any]], embeddings: dict[str, list[float]],
                           flow_axis: list[float] | None = None) -> list[dict[str, Any]]:
    o = WALK_DEFAULTS
    embedded = [i for i in items if i["id"] in embeddings]
    unembedded = [i for i in items if i["id"] not in embeddings]
    if len(embedded) < o["minItemsForWalk"]:
        return list(items)
    seed = embedded[0]
    for item in embedded[1:]:
        if item["priority"] > seed["priority"]:
            seed = item
    visited = {seed["id"]}
    walk = [seed]
    dim = len(embeddings[seed["id"]])
    centroid = list(embeddings[seed["id"]])
    consecutive = 1
    while len(walk) < len(embedded):
        current = embeddings[walk[-1]["id"]]
        current_type = walk[-1]["type"]
        hard_cap = consecutive >= o["hardMaxConsecutiveSameType"]
        require_switch = hard_cap and any(c["id"] not in visited and c["type"] != current_type for c in embedded)
        best_score, best = -math.inf, None
        for cand in embedded:
            if cand["id"] in visited or (require_switch and cand["type"] == current_type):
                continue
            emb = embeddings[cand["id"]]
            sim = cosine_similarity(current, emb)
            lo, hi = o["sweetSpotMin"], o["sweetSpotMax"]
            if lo <= sim <= hi:
                sim_score = 1.0 - (abs(sim - (lo + hi) / 2) / ((hi - lo) / 2)) * 0.3
            elif sim > hi:
                sim_score = 0.7 - (sim - hi) * 2.0
            else:
                sim_score = 0.7 - (lo - sim) * 0.5
            coverage = (1.0 - cosine_similarity(emb, centroid)) * 0.3
            type_penalty = -0.5 if cand["type"] == current_type and consecutive >= o["maxConsecutiveSameType"] else 0
            flow = 0.0
            if flow_axis:
                cur_dot = cand_dot = 0.0
                for d in range(min(len(current), len(flow_axis))):
                    cur_dot += current[d] * flow_axis[d]
                    cand_dot += emb[d] * flow_axis[d]
                flow = math.tanh((cand_dot - cur_dot) * 3) * o["flowWeight"]
            proximity = 0.0
            weight = o["conceptPairWeight"]
            if weight != 0 and cand.get("conceptId"):
                cadence = weight < 0
                window = CONCEPT_CADENCE_WINDOW if cadence else LEGACY_CONCEPT_PAIR_WINDOW
                for k in range(1, min(window, len(walk)) + 1):
                    if walk[-k].get("conceptId") == cand["conceptId"]:
                        proximity = weight * ((window - k + 1) / window) if cadence else weight / k
                        break
            total = sim_score + coverage + type_penalty + flow + proximity
            if total > best_score:
                best_score, best = total, cand
        if best is None:
            break
        visited.add(best["id"])
        walk.append(best)
        consecutive = consecutive + 1 if best["type"] == current_type else 1
        n = len(walk)
        new = embeddings[best["id"]]
        for d in range(dim):
            centroid[d] = centroid[d] * ((n - 1) / n) + new[d] / n
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


def inject_preemptive_scaffolds(items: list[dict[str, Any]], bulk: dict[str, Any], *, rotation_seed: str,
                                step_down_concept_ids: set[str], max_pairings: int | None) -> list[dict[str, Any]]:
    if not step_down_concept_ids:
        return items
    if max_pairings is None:
        anchors = sum(1 for it in items if _is_anchor(it))
        max_pairings = max(2, min(max(2, anchors), max(2, len(items) // 3)))
    pool = bulk["unseenCards"]
    if max_pairings <= 0 or not items or not pool:
        return items
    scaffolds = [c for c in pool if c.get("complexity") == 1]
    if not scaffolds:
        return items
    candidates = shuffle_with_seed(scaffolds, rotation_seed) if rotation_seed else scaffolds
    used: set[str] = set()
    present = {it["id"] for it in items}
    groups = {it["variantGroupId"] for it in items if it["type"] == "card" and it.get("variantGroupId")}
    insertions: dict[int, dict[str, Any]] = {}
    for i, item in enumerate(items):
        if len(insertions) >= max_pairings:
            break
        if not _is_anchor(item) or (item.get("conceptId") or "") not in step_down_concept_ids:
            continue
        nxt = items[i + 1] if i + 1 < len(items) else None
        if nxt and nxt["type"] == "card" and nxt.get("complexity") == 1:
            continue
        left = _specific(item.get("topics"))
        card = next((c for c in candidates if c["id"] not in used and c["id"] not in present
                     and (not c.get("variantGroupId") or c["variantGroupId"] not in groups)
                     and bool(left) and any(t in left for t in _specific(c["topics"]))), None)
        if card is None:
            continue
        used.add(card["id"])
        if card.get("variantGroupId"):
            groups.add(card["variantGroupId"])
        insertions[i] = {
            "type": "card", "id": card["id"], "rotation": item["rotation"] if item.get("rotation") is not None else bulk["rotation"], "priority": 0.5,
            "conceptId": item.get("conceptId") or "", "conceptName": item.get("conceptName") or "",
            "topics": card["topics"], "complexity": card.get("complexity"), "clusterId": card.get("clusterId"),
            "variantGroupId": card.get("variantGroupId"), "interventionReason": "preemptive_scaffold",
            "struggleIntervention": {"strategy": "preemptive", "isScaffold": True, "targetCardId": item["id"]},
        }
    if not insertions:
        return items
    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        out.append(item)
        if i in insertions:
            out.append(insertions[i])
    return out


def apply_struggle_to_selection(inputs: dict[str, Any], build: Any) -> list[dict[str, Any]]:
    from .struggle import apply_struggle_interventions

    return apply_struggle_interventions(inputs, build.selected_items)


def order_session_items(inputs: dict[str, Any], plan: dict[str, Any], build: Any,
                        items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from .difficulty import get_effective_question_difficulty
    from .errors import UnsupportedSchedulerInput
    from .policies import apply_review_challenge_wave, modality_eligible_for_review_learning, review_learning_policy
    from .scoring import enforce_concept_teaching_caps, normalize_review_challenge_level

    if inputs.get("composedDeck"):
        raise UnsupportedSchedulerInput("composed-deck source fairness is not ported in Phase 3a")
    options, bulk, rotation = inputs["options"], inputs["bulk"], inputs["rotation"]
    level = normalize_review_challenge_level(options.get("reviewChallenge"))
    policy = review_learning_policy(options.get("reviewChallenge"))
    hard_gap = {l["questionId"] for l in bulk["questionConceptLinks"] if l["questionId"] in bulk["questionMap"]} \
        if policy["questionsOnly"] else set()
    cards = bulk["unseenCards"] + bulk["seenCards"]
    complexity = {c["id"]: c.get("complexity") for c in cards}
    flow = inputs.get("flowAxis") or None
    ordered = order_by_manifold_walk(items, inputs.get("itemEmbeddings") or {}, flow)
    for item in ordered:
        p = item["priority"]
        item["difficulty"] = "hard" if p > 0.66 else "medium" if p > 0.33 else "easy"
    rhythm = apply_difficulty_rhythm(ordered, options.get("commitmentLevel") or "browser") if level == 0 else ordered
    mono = break_modality_runs(rhythm)
    meta = {c["id"]: {"stableId": c.get("stableId"), "topics": c["topics"], "complexity": c.get("complexity"),
                      "clusterId": c.get("clusterId"), "variantGroupId": c.get("variantGroupId"),
                      "facilityIndex": c.get("facilityIndex"), "sampleSize": c.get("sampleSize"),
                      "similarCards": c.get("similarCards")} for c in cards}
    for it in mono:
        if it["type"] == "card" and it["id"] in meta:
            m = meta[it["id"]]
            for key in ("topics", "complexity", "clusterId", "variantGroupId"):
                if it.get(key) is None:
                    it[key] = m[key]
        elif it["type"] == "question" and it["id"] in bulk["questionMap"]:
            q = bulk["questionMap"][it["id"]]
            if it.get("topics") is None:
                it["topics"] = q["topics"]
            if level != 0:
                it["difficulty"] = get_effective_question_difficulty(q.get("difficulty"), q.get("facilityIndex"),
                                                                     q.get("totalAttempts"))
    paired = inject_preemptive_scaffolds(
        mono, bulk, rotation_seed=inputs["servingRotationSeed"], step_down_concept_ids=inputs["recentFailureConceptIds"],
        max_pairings=0 if inputs.get("effectiveMaxNewCards") == 0 else None,
    )

    def endpoint(item: dict[str, Any]) -> bool:
        if not modality_eligible_for_review_learning(item["type"], policy["level"]):
            return False
        if policy["questionsOnly"]:
            q = bulk["questionMap"].get(item["id"]) if item["type"] == "question" else None
            if not q or q.get("difficulty") != "hard" or item["id"] not in hard_gap:
                return False
        if policy["scaffoldsOnly"]:
            return item["type"] == "card" and complexity.get(item["id"]) == 1
        return True

    capped = enforce_concept_teaching_caps([i for i in paired if endpoint(i)], build.teaching_cap_for)
    card_rot = {c["id"]: c.get("rotation") for c in cards}
    q_rot = {q["id"]: q.get("rotation") for q in bulk["rotationQuestions"]}
    unknown = plan["unknownItemRotation"]

    def get_rotation(item: dict[str, Any]) -> str:
        if item.get("rotation"):
            return item["rotation"]
        if item["type"] == "card":
            return card_rot.get(item["id"]) or unknown
        if item["type"] == "question":
            return q_rot.get(item["id"]) or unknown
        return unknown

    source_capped = cap_cross_source_session_items(
        capped, session_rotation=rotation, allowed=options.get("crossSourceRotations") or [],
        max_items=options.get("maxCrossSourceItems"), get_rotation=get_rotation,
    )
    final = break_modality_runs(source_capped)
    if level != 0:
        final = apply_review_challenge_wave(final, level)
    return final, meta
