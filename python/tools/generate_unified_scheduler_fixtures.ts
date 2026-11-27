/**
 * Golden-fixture generator for the Python port, Phase 3a (unified scheduler).
 *
 * One-time generator, NOT a runtime dependency: it feeds fixed snapshots
 * (fixed clock, fixed 64-hex seed, stable IDs, precomputed vectors) through the
 * TypeScript scheduler phases and writes input/output pairs as JSON under
 * python/tests/fixtures/. Regenerate from the repository root with:
 *
 *   node --import tsx python/tools/generate_unified_scheduler_fixtures.ts
 *
 * Storage reads (setup/load phases) are replaced by the explicit snapshot; the
 * phase functions downstream of the read boundary run unmodified.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  findPristineConcepts,
  planCardQuota,
  rankConcepts,
  selectWeakConcepts,
  sortConceptsByPriority,
} from '../../src/lib/knowledge/unified-scheduler-phase-concepts';
import { SessionBuildState } from '../../src/lib/knowledge/unified-scheduler-phase-state';
import { releaseFirstSightReservation } from '../../src/lib/knowledge/unified-scheduler-phase-backfill';
import {
  buildClusterRoundRobin,
  clusterExposureDampening,
  computeExamPressure,
  computeRecallSoftPenalty,
  enforceConceptTeachingCaps,
  isCalibratedReviewItem,
  resolveMaxNewCards,
} from '../../src/lib/knowledge/unified-scheduler-scoring';
import { stableTargetRandom } from '../../src/lib/knowledge/unified-scheduler-config';
import {
  buildDifficultyPlan,
  getEffectiveQuestionDifficulty,
  normalizeQuestionDifficulty,
} from '../../src/lib/knowledge/candidate-ranking';
import { buildConceptTopicIndex, resolveCardConceptId } from '../../src/lib/knowledge/scheduler-attribution';
import { resolveInterventionReason } from '../../src/lib/knowledge/intervention-reason';
import { classifyTeachingState } from '../../src/lib/scheduler/concept-teaching-state';
import { expandTopicSet, normalizeTopic } from '../../src/lib/topics';
import type { ConceptState } from '../../src/lib/knowledge/unified-scheduler-types';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = any;

const NOW_MS = Date.UTC(2026, 0, 15, 12, 0, 0);
const SEED = 'a3f1c9e2b7d4058e6a1f3c5d7e9b2a4c6e8f0a1b3c5d7e9f1a2b3c4d5e6f7a8b';
const DAY = 86_400_000;

function toJson(value: unknown): Json {
  return JSON.parse(JSON.stringify(value, (_k, v) => {
    if (typeof v === 'number' && !Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
    if (v instanceof Set) return [...v];
    if (v instanceof Map) return Object.fromEntries(v);
    return v;
  }));
}

/** Snapshot JSON -> the SchedulerInputs shape the phase functions read. */
function toSchedulerInputs(s: Json): any {
  const stateMap = new Map<string, any>();
  for (const [id, row] of Object.entries<any>(s.stateMap ?? {})) {
    stateMap.set(id, {
      ...row,
      lastProbeAt: row.lastProbeAt == null ? null : new Date(row.lastProbeAt),
      lastExposureAt: row.lastExposureAt == null ? null : new Date(row.lastExposureAt),
    });
  }
  const options = { ...s.options };
  if (options.topicTeachingWeeks) options.topicTeachingWeeks = new Map(Object.entries(options.topicTeachingWeeks));
  if (options.topicRecencyDays) options.topicRecencyDays = new Map(Object.entries(options.topicRecencyDays));
  return {
    applyExamTarget: false,
    targetSidecarsValid: false,
    runtimeTargetSnapshot: null,
    conceptTargetScores: { scores: new Map() },
    effectiveTargetInfluence: null,
    concepts: s.concepts,
    conceptMap: new Map(s.concepts.map((c: any) => [c.id, c])),
    stateMap,
    conceptEmbeddings: new Map(Object.entries(s.conceptEmbeddings ?? {})),
    gapDirection: s.gapDirection ?? null,
    likedConceptIds: new Set(s.likedConceptIds ?? []),
    recentFailureConceptIds: new Set(s.recentFailureConceptIds ?? []),
    practiceMissConceptIds: new Set(s.practiceMissConceptIds ?? []),
    chronicFailureConceptIds: new Set(s.chronicFailureConceptIds ?? []),
    recentClusterExposures: new Map(Object.entries(s.recentClusterExposures ?? {})),
    daysToExam: s.daysToExam,
    examPressure: s.examPressure,
    effectiveMaxNewCards: s.effectiveMaxNewCards ?? undefined,
    cardRatio: s.cardRatio,
    size: s.size,
    rotation: s.rotation,
    options,
    selectionNowMs: () => s.nowMs,
    bulk: {
      unseenCards: s.bulk.unseenCards ?? [],
      seenCards: s.bulk.seenCards ?? [],
      rotationQuestions: s.bulk.rotationQuestions ?? [],
      passSetDeferredCardIds: new Set(s.bulk.passSetDeferredCardIds ?? []),
      questionFamiliarity: new Map((s.bulk.questionFamiliarity ?? []).map((id: string) => [id, {}])),
    },
  };
}

function runConceptPlan(snapshot: Json): Json {
  const inputs = toSchedulerInputs(snapshot);
  const ranking = rankConcepts(inputs);
  const ranked = toJson(ranking.conceptStates);
  const clusters = Object.fromEntries(ranking.conceptStates.map(s => [s.conceptId, ranking.getConceptCluster(s)]));
  sortConceptsByPriority(inputs, ranking);
  const pristine = findPristineConcepts(inputs);
  const selection = selectWeakConcepts(inputs, ranking, pristine);
  const quota = planCardQuota(inputs, selection.cardRatio);
  const roundRobin = buildClusterRoundRobin(selection.weakConcepts, ranking.getConceptCluster).map(s => s.conceptId);
  return toJson({
    rankedStates: ranked,
    conceptClusters: clusters,
    sortedConceptIds: ranking.conceptStates.map(s => s.conceptId),
    sortedPriorities: ranking.conceptStates.map(s => s.priority),
    pristineConceptIds: [...pristine].sort(),
    weakConceptIds: selection.weakConcepts.map(s => s.conceptId),
    cardRatio: selection.cardRatio,
    quota,
    clusterRoundRobin: roundRobin,
    unseenCardIds: [...ranking.unseenCardIds].sort(),
    crossSourceCardIds: [...ranking.crossSourceCardIds].sort(),
    crossSourceQuestionIds: [...ranking.crossSourceQuestionIds].sort(),
    maxCrossSourceItems: ranking.maxCrossSourceItems,
    maxNewCards: ranking.maxNewCards,
    unknownItemRotation: ranking.unknownItemRotation,
  });
}

/** Scripted SessionBuildState session: every add/pick result plus final state. */
function runBuildScript(script: Json): Json {
  const weak: ConceptState[] = script.weakConcepts;
  const build = new SessionBuildState({
    size: script.size,
    minFirstSightItems: script.minFirstSightItems,
    maxNewCards: script.maxNewCards == null ? Infinity : script.maxNewCards,
    maxCrossSourceItems: script.maxCrossSourceItems ?? 0,
    unseenCardIds: new Set(script.unseenCardIds ?? []),
    crossSourceCardIds: new Set(script.crossSourceCardIds ?? []),
    crossSourceQuestionIds: new Set(script.crossSourceQuestionIds ?? []),
    bulk: { questionFamiliarity: new Map((script.questionFamiliarity ?? []).map((id: string) => [id, {} as any])) },
    excludedCardIds: new Set(script.excludedCardIds ?? []),
    excludedQuestionIds: new Set(script.excludedQuestionIds ?? []),
    excludedVideoIds: new Set(script.excludedVideoIds ?? []),
    weakConcepts: weak,
    conceptThreadAnchors: [],
    nowMs: NOW_MS,
  });
  const results: Json[] = [];
  for (const op of script.ops) {
    if (op.op === 'add') {
      results.push({ op: 'add', id: op.item.id, added: build.addItem(op.item, op.bypassNoveltyReservation ? { bypassNoveltyReservation: true } : undefined) });
    } else if (op.op === 'pick') {
      const picked = build.pickCardCandidate(op.candidates, { similarityThreshold: op.similarityThreshold, maxPerCluster: op.maxPerCluster });
      let added: boolean | null = null;
      if (picked) {
        added = build.addItem({ type: 'card', id: picked.id, conceptId: op.conceptId, conceptName: op.conceptId, priority: 1, interventionReason: 'weak_recall', variantGroupId: picked.variantGroupId } as any);
        if (added) {
          build.recordCluster(picked.clusterId);
          build.recordCardNeighborhood(picked);
        }
      }
      results.push({ op: 'pick', picked: picked?.id ?? null, added });
    } else if (op.op === 'release') {
      releaseFirstSightReservation({ size: script.size } as any, build);
      results.push({ op: 'release' });
    }
  }
  return toJson({
    results,
    selectedIds: build.selectedItems.map(i => `${i.type}:${i.id}`),
    deferredForNovelty: [...build.deferredForNovelty.keys()],
    selectedCardCount: build.selectedCardCount,
    selectedQuestionCount: build.selectedQuestionCount,
    newCardCount: build.newCardCount,
    firstSightSelected: build.firstSightSelected,
    selectedCrossSourceItems: build.selectedCrossSourceItems,
    enforceFirstSightReservation: build.enforceFirstSightReservation,
    teachingCaps: Object.fromEntries(weak.map(w => [w.conceptId, build.teachingCapFor(w.conceptId)])),
    cardsPerConcept: build.cardsPerConcept,
    questionsPerConcept: build.questionsPerConcept,
  });
}

const ago = (days: number) => NOW_MS - days * DAY;
const card = (id: string, topics: string[], extra: Json = {}) => ({
  id, topics, clusterId: null, rotation: 'rot-main', sourceFile: `${id}.md`, similarCards: null,
  variantGroupId: null, variantIndex: null, variantType: null, stableId: id, ...extra,
});
const concept = (id: string, topics: string[], extra: Json = {}) => ({
  id, name: id.toUpperCase(), topics, prerequisiteIds: [], examWeight: 3, ...extra,
});
const st = (recallProbability: number, confidence: number, exposureCount: number, exposedDaysAgo: number | null, probedDaysAgo: number | null) => ({
  recallProbability, confidence, exposureCount,
  lastExposureAt: exposedDaysAgo == null ? null : ago(exposedDaysAgo),
  lastProbeAt: probedDaysAgo == null ? null : ago(probedDaysAgo),
});

function baseSnapshot(): Json {
  return {
    nowMs: NOW_MS, daysToExam: 30, examPressure: computeExamPressure(30), size: 20, cardRatio: 0.7,
    rotation: 'rot-main', effectiveMaxNewCards: null,
    options: { examTargetTieBreakSeed: SEED },
    concepts: [
      concept('c-weak', ['Renal Physiology'], { prerequisiteIds: ['c-prereq'], examWeight: 4 }),
      concept('c-new', ['cardiac_output']),
      concept('c-pristine', ['Acid-Base'], { examWeight: 5 }),
      concept('c-strong', ['Glycolysis'], { examWeight: 1 }),
      concept('c-prereq', ['Nephron Anatomy'], { prerequisiteIds: ['c-root'] }),
      concept('c-root', ['Cell Membranes']),
      concept('c-hub', ['Integrated Renal'], { prerequisiteIds: ['c-weak', 'c-new'] }),
      concept('c-zero', ['Zeta Topic'], { examWeight: 0 }),
    ],
    stateMap: {
      'c-weak': st(0.35, 0.5, 5, 2, 4),
      'c-pristine': st(0.97, 0.9, 12, 1, 1),
      'c-strong': st(0.98, 0.95, 20, 0.5, 0.5),
      'c-prereq': st(0.55, 0.2, 2, 6, 6),
      'c-root': st(0.9, 0.8, 9, 1, 2),
      'c-hub': st(0.4, 0.45, 4, 3, 10),
      'c-zero': st(0.2, 0.1, 1, 30, null),
    },
    conceptEmbeddings: {
      'c-weak': [0.9, 0.1, 0, 0.4], 'c-new': [0.1, 0.8, 0.2, 0], 'c-hub': [0.5, 0.5, 0.5, 0.5], 'c-zero': [0, 0, 0, 0],
    },
    gapDirection: [0.8, 0.2, 0.1, 0.55],
    likedConceptIds: ['c-new'],
    recentFailureConceptIds: ['c-hub'],
    practiceMissConceptIds: ['c-weak'],
    chronicFailureConceptIds: ['c-strong'],
    bulk: {
      unseenCards: [
        card('u-acid-1', ['Acid-Base'], { clusterId: 'cl-renal' }),
        card('u-acid-2', ['acid base']),
        card('u-renal-1', ['Renal Physiology'], { clusterId: 'cl-renal', rotation: 'rot-cross' }),
        card('u-cardiac-1', ['Cardiac Output'], { clusterId: 'cl-cardio' }),
        card('u-deferred', ['Glycolysis'], { clusterId: 'cl-metab' }),
      ],
      seenCards: [
        card('s-glyco-1', ['Glycolysis'], { clusterId: 'cl-metab' }),
        card('s-nephron-1', ['Nephron Anatomy'], { clusterId: 'cl-renal' }),
      ],
      rotationQuestions: [{ id: 'q-1', rotation: 'rot-cross' }, { id: 'q-2', rotation: 'rot-main' }],
      passSetDeferredCardIds: ['u-deferred'],
    },
  };
}

function variant(name: string, mutate: (s: Json) => void): { name: string; snapshot: Json } {
  const snapshot = baseSnapshot();
  mutate(snapshot);
  return { name, snapshot };
}

const conceptPlanCases = [
  variant('weak_new_pristine_mix', () => {}),
  variant('cluster_dampening', s => { s.recentClusterExposures = { 'cl-renal': 9, 'cl-cardio': 5, 'cl-metab': 0 }; }),
  variant('empty_pools', s => { s.bulk = { unseenCards: [], seenCards: [], rotationQuestions: [] }; }),
  variant('crunch_no_new_cards', s => { s.options.mode = 'crunch'; s.effectiveMaxNewCards = 0; s.daysToExam = 5; s.examPressure = computeExamPressure(5); }),
  variant('explicit_card_ratio', s => { s.options.cardRatio = 0.5; s.cardRatio = 0.5; s.size = 7; }),
  variant('questions_only_challenge', s => { s.options.reviewChallenge = 2; }),
  variant('scaffolds_only_challenge', s => { s.options.reviewChallenge = -1.6; s.size = 9; }),
  variant('maintenance_shift', s => {
    s.stateMap = { 'c-strong': st(0.99, 0.95, 20, 0, 0) };
    s.bulk.unseenCards = [];
  }),
  variant('exam_day_horizon', s => { s.daysToExam = 0; s.examPressure = computeExamPressure(0); }),
  variant('far_exam_horizon', s => { s.daysToExam = 120; s.examPressure = computeExamPressure(120); }),
  variant('cross_source_scope', s => { s.options.crossSourceRotations = ['rot-cross']; s.options.maxCrossSourceItems = 1.7; }),
  variant('teaching_pace_boosts', s => {
    s.options.currentTeachingWeek = 3;
    s.options.topicTeachingWeeks = { 'renal-physiology': 3, 'cardiac-output': 2, 'acid-base': 1 };
    s.options.topicRecencyDays = { 'glycolysis': 2, 'zeta-topic': 13.5, 'cell-membranes': 14 };
  }),
  variant('all_strong_no_unseen', s => {
    for (const c of s.concepts) s.stateMap[c.id] = st(0.99, 0.95, 20, 0.1, 0.1);
    s.bulk.unseenCards = [];
  }),
];

const ws = (conceptId: string, exposureCount: number, recallOnExamDay: number, confidence: number) => ({
  conceptId, conceptName: conceptId, currentRecall: recallOnExamDay, recallOnExamDay, confidence, exposureCount,
  daysSinceProbe: 1, daysSinceExposure: 1, priority: 1, intervention: 'probe',
});
const item = (type: string, id: string, conceptId: string, extra: Json = {}) => ({
  type, id, conceptId, conceptName: conceptId, priority: 1, interventionReason: 'weak_recall', ...extra,
});
const cand = (id: string, extra: Json = {}) => ({
  id, clusterId: null, similarCards: null, variantGroupId: null, variantIndex: null, variantType: null, ...extra,
});

const buildScripts = [
  {
    name: 'interference_cluster_variant_budget',
    size: 10, minFirstSightItems: 0, maxNewCards: 1, maxCrossSourceItems: 0,
    unseenCardIds: ['n-1', 'n-2'], crossSourceCardIds: ['x-1'], excludedCardIds: ['ex-1'],
    weakConcepts: [ws('c-a', 3, 0.6, 0.5)],
    ops: [
      { op: 'pick', conceptId: 'c-a', similarityThreshold: 0.85, maxPerCluster: 1, candidates: [
        cand('ex-1'), cand('x-1'), cand('n-1', { clusterId: 'k1', similarCards: [{ cardId: 'z-9', similarity: 0.95 }, { cardId: 7 }, 'junk'] }),
      ] },
      { op: 'pick', conceptId: 'c-a', similarityThreshold: 0.85, maxPerCluster: 1, candidates: [
        cand('n-2'), cand('k-1b', { clusterId: 'k1' }), cand('z-9'), cand('sim-1', { similarCards: [{ cardId: 'n-1', similarity: 0.9 }] }),
        cand('v-1', { variantGroupId: 'g1' }),
      ] },
      { op: 'pick', conceptId: 'c-a', similarityThreshold: 0.95, maxPerCluster: 5, candidates: [
        cand('v-2', { variantGroupId: 'g1' }), cand('z-9'), cand('ok-1'),
      ] },
    ],
  },
  {
    name: 'teaching_caps_and_synthetic',
    size: 12, minFirstSightItems: 0, maxNewCards: null,
    weakConcepts: [ws('c-naive', 0, 0, 0), ws('c-mastered', 9, 0.9, 0.8), ws('c-learning', 4, 0.3, 0.4), ws('c-consol', 4, 0.9, 0.5)],
    questionFamiliarity: ['q-1', 'q-2', 'q-3', 'q-4', 'q-5', 'q-6'],
    ops: [
      ...['a', 'b', 'c'].map(k => ({ op: 'add', item: item('card', `naive-${k}`, 'c-naive') })),
      ...['a', 'b'].map(k => ({ op: 'add', item: item('question', `q-m-${k}`, 'c-mastered') })),
      { op: 'add', item: item('video', 'vid-1', 'c-mastered') },
      ...['1', '2', '3', '4'].map(k => ({ op: 'add', item: item('question', `q-${k}`, 'c-learning', { variantGroupId: 'vg', variantType: k === '1' ? 'near-duplicate' : 'paraphrase' }) })),
      { op: 'add', item: item('card', 'orphan-1', '_unattached') },
      { op: 'add', item: item('card', 'rot-1', 'rotation:rot-main') },
      { op: 'add', item: item('card', 'other-1', 'c-unknown') },
    ],
  },
  {
    name: 'first_sight_reservation_release',
    size: 5, minFirstSightItems: 3, maxNewCards: null,
    unseenCardIds: ['new-1', 'new-2'], questionFamiliarity: ['q-seen-1', 'q-seen-2', 'q-seen-3'],
    weakConcepts: [ws('c-a', 3, 0.4, 0.5), ws('c-b', 3, 0.4, 0.5), ws('c-c', 3, 0.4, 0.5)],
    ops: [
      { op: 'add', item: item('card', 'seen-1', 'c-a') },
      { op: 'add', item: item('question', 'q-seen-1', 'c-b') },
      { op: 'add', item: item('question', 'q-seen-2', 'c-c') },
      { op: 'add', item: item('question', 'q-seen-3', 'c-c'), bypassNoveltyReservation: true },
      { op: 'add', item: item('card', 'new-1', 'c-a') },
      { op: 'add', item: item('card', 'seen-2', 'c-b') },
      { op: 'release' },
      { op: 'add', item: item('card', 'new-2', 'c-a') },
    ],
  },
  {
    name: 'cross_source_cap_and_session_full',
    size: 3, minFirstSightItems: 0, maxNewCards: null, maxCrossSourceItems: 1,
    crossSourceCardIds: ['x-1', 'x-2'], crossSourceQuestionIds: ['xq-1'], questionFamiliarity: ['xq-1'],
    weakConcepts: [ws('c-a', 3, 0.4, 0.5)],
    ops: [
      { op: 'add', item: item('card', 'x-1', 'rotation:rot-cross') },
      { op: 'add', item: item('question', 'xq-1', 'rotation:rot-cross') },
      { op: 'pick', conceptId: 'c-a', similarityThreshold: 0.85, maxPerCluster: 3, candidates: [cand('x-2'), cand('plain-1')] },
      { op: 'add', item: item('card', 'plain-2', '_unattached') },
      { op: 'add', item: item('card', 'plain-3', '_unattached') },
      { op: 'release' },
    ],
  },
  { name: 'empty_script', size: 0, minFirstSightItems: 0, maxNewCards: 0, weakConcepts: [], ops: [{ op: 'add', item: item('card', 'c-1', 'c-a') }, { op: 'release' }] },
];

function runUnits(): Json {
  const capItems = [
    item('card', 'k-1', 'c-a'), item('question', 'k-2', 'c-a', { struggleIntervention: { isScaffold: true, targetCardId: 'k-4' } }),
    item('card', 'k-3', 'c-a', { examTargetMasteryStage: 'scheduled-atomic-core' }), item('card', 'k-4', 'c-a'),
    item('card', 'k-5', 'c-a'), item('card', 'k-6', 'c-b'), item('card', 'k-7', '_unattached'), item('card', 'k-8', 'c-b'),
    item('card', 'k-9', 'question:abc'), item('card', 'k-10', 'c-c'),
  ];
  const caps: Record<string, number> = { 'c-a': 2, 'c-b': 1.9, 'c-c': -1 };
  const capVariants = [0, 1, 2, 3].map(n => ({ capA: n, result: enforceConceptTeachingCaps(capItems as any, id => (id === 'c-a' ? n : caps[id] ?? 3)).map(i => i.id) }));
  const rrConcepts = [
    { ...ws('r-1', 1, 0, 0), priority: 0.5 }, { ...ws('r-2', 1, 0, 0), priority: 0.9 }, { ...ws('r-3', 1, 0, 0), priority: 0.7 },
    { ...ws('r-4', 1, 0, 0), priority: 0.2 }, { ...ws('r-5', 1, 0, 0), priority: 0.95 }, { ...ws('r-6', 1, 0, 0), priority: 0.6 },
  ];
  const rrCluster: Record<string, string> = { 'r-1': 'A', 'r-2': 'B', 'r-3': 'A', 'r-4': 'C', 'r-5': 'A', 'r-6': 'B' };
  const rank: Record<string, number | null> = { 'r-1': 2, 'r-2': 1, 'r-4': null, 'r-5': 0, 'r-3': 5, 'r-6': 1 };
  const difficultyInputs: Json[] = [
    { difficulty: ' HARD ', facilityIndex: null, totalAttempts: null }, { difficulty: 'Extreme', facilityIndex: null, totalAttempts: null },
    { difficulty: 5, facilityIndex: null, totalAttempts: null }, { difficulty: null, facilityIndex: 0.9, totalAttempts: 19 },
    { difficulty: 'easy', facilityIndex: 0.7, totalAttempts: 20 }, { difficulty: 'easy', facilityIndex: 0.4, totalAttempts: 50 },
    { difficulty: 'easy', facilityIndex: 0.39, totalAttempts: 20 }, { difficulty: 'hard', facilityIndex: 0.69999, totalAttempts: 20.5 },
    { difficulty: '\u00a0medium\n', facilityIndex: '0.9', totalAttempts: 100 },
  ];
  const attrConcepts = [concept('a-1', ['Renal Physiology', '']), concept('a-2', ['renal_physiology', 'Shared']), concept('a-3', ['Shared', 'Cardiac'])];
  const attrIndex = buildConceptTopicIndex(attrConcepts);
  const current = new Set(['a-1', 'a-3']);
  const attrCards = [
    { conceptId: 'a-3', topics: ['Shared'] }, { conceptId: 'a-2', topics: ['Cardiac'] }, { conceptId: null, topics: ['Renal-Physiology!'] },
    { conceptId: null, topics: ['Cardiac', 'cardiac'] }, { conceptId: null, topics: ['nothing', '***'] }, { conceptId: 'zz', topics: ['Shared'] },
  ];
  const topicInputs = ['Renal Physiology', '  acid_base--balance ', 'TBI', 'bls', 'Café au lait', 'and the of', 'vs ECG', 'α-Thalassemia', '***', 'ab cd ef'];
  return toJson({
    stableTargetRandom: ['c-weak', 'c-new', '', 'unicode-é', 'k'.repeat(200)].map(key => ({ key, value: stableTargetRandom(SEED, key) })),
    examPressure: [-5, 0, 7, 21, 30, 120].map(d => ({ daysToExam: d, value: computeExamPressure(d) })),
    resolveMaxNewCards: [
      { explicit: 4, mode: 'crunch', value: resolveMaxNewCards({ explicit: 4, mode: 'crunch' }) },
      { explicit: null, mode: 'crunch', value: resolveMaxNewCards({ explicit: null, mode: 'crunch' }) },
      { explicit: null, mode: 'standard', value: resolveMaxNewCards({ explicit: null, mode: 'standard' as any }) },
      { explicit: 0, mode: null, value: resolveMaxNewCards({ explicit: 0 }) },
    ],
    recallSoftPenalty: [[0.95, 0.9, false], [0.95, 0.9, true], [0.9, 0.9, false], [0.95, 0.39, false], [1, 0.4, false]].map(([r, c, ch]) => ({ recall: r, confidence: c, chronic: ch, value: computeRecallSoftPenalty(r as number, c as number, { isChronicFailure: ch as boolean }) })),
    clusterDampening: [0, 3, 4, 5, 8, 11, 12, 50].map(n => ({ count: n, value: clusterExposureDampening(n) })),
    calibratedReviewItem: [['question', null], ['card', 1], ['card', 2], ['card', null], ['video', 3]].map(([t, c]) => ({ type: t, complexity: c, value: isCalibratedReviewItem({ type: t as string, complexity: c as number | null }) })),
    teachingCaps: { items: capItems, caps, variants: capVariants },
    clusterRoundRobin: {
      concepts: rrConcepts, clusters: rrCluster, ranks: rank,
      byPriority: buildClusterRoundRobin(rrConcepts as any, c => rrCluster[c.conceptId]).map(c => c.conceptId),
      byRank: buildClusterRoundRobin(rrConcepts as any, c => rrCluster[c.conceptId], c => rank[c.conceptId] ?? null).map(c => c.conceptId),
    },
    difficultyPlan: [[['hard', 'easy'], 5], [['bogus'], 3], [[], 2], [['medium', 'x', 'hard'], 4], [['easy'], 0]].map(([p, n]) => ({ pattern: p, length: n, value: buildDifficultyPlan(p as any, n as number) })),
    normalizeDifficulty: [' Easy ', 'HARD', 'extreme', '', null, 3, '\tmedium\u2028'].map(v => ({ input: v, value: normalizeQuestionDifficulty(v) })),
    effectiveDifficulty: difficultyInputs.map(i => ({ ...i, value: getEffectiveQuestionDifficulty(i) })),
    attribution: { concepts: attrConcepts, currentConceptIds: [...current], cards: attrCards.map(c => ({ ...c, value: resolveCardConceptId(c as any, current, attrIndex) })) },
    interventionReason: [['c-f', 0.9, 'weak_recall'], ['c-p', 0.8, 'reinforcement'], ['c-p', 0.79, 'reinforcement'], ['c-x', 0.95, 'needs_retest']].map(([id, r, base]) => ({ conceptId: id, recall: r, base, value: resolveInterventionReason({ conceptId: id as string, recallOnExamDay: r as number, baseReason: base as any, recentFailureConceptIds: new Set(['c-f']), conceptHasPristine: new Set(['c-p', 'c-f']), targetRecall: 0.8 }) })),
    teachingState: [[0, 0.9, 0.9], [1, 0.49, 0.9], [1, 0.5, 0.9], [1, 0.85, 0.69], [1, 0.85, 0.7]].map(([e, r, c]) => ({ exposureCount: e, recall: r, confidence: c, value: classifyTeachingState({ exposureCount: e, recallOnExamDay: r, confidence: c }) })),
    topics: topicInputs.map(t => ({ input: t, normalized: normalizeTopic(t), expanded: expandTopicSet([t]) })),
  });
}

const fixture = {
  generator: 'python/tools/generate_unified_scheduler_fixtures.ts',
  nowMs: NOW_MS,
  seed: SEED,
  conceptPlan: conceptPlanCases.map(c => ({ name: c.name, snapshot: c.snapshot, expected: runConceptPlan(c.snapshot) })),
  buildScripts: buildScripts.map(s => ({ name: s.name, script: s, expected: runBuildScript(s) })),
  units: runUnits(),
};
const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '..', 'tests', 'fixtures', 'unified-scheduler.cases.json');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(fixture, null, 2) + '\n');
console.log(`wrote ${out}`);
