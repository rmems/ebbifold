/**
 * Generates python/tests/fixtures/unified-scheduler-session.cases.json: end-to-end
 * session construction through the real TypeScript phase functions.
 *
 * Storage adapters (the only non-original code paths):
 * - globalThis.prisma is a fake that serves item embeddings ($queryRaw) and an
 *   empty cluster table (cluster fallback) from the snapshot.
 * - rotation top-up rows come from snapshot.rotationTopUp (what
 *   getQuestionsForRotation would return), mirroring topUpFromRotationBank.
 * - struggle-history tables (card progress, question responses, bridge-card
 *   pools, scaffold/context/variant lookups) come from snapshot.struggleHistory.
 * - the global Date is frozen at NOW_MS so code that reads the wall clock
 *   (e.g. hasDemonstratedLearningGap's default `now`) sees the fixture clock.
 * - the reviewed simulation item registry is copied into snapshot.simulationItemMap.
 *
 * Run: node --import tsx python/tools/generate_unified_scheduler_session_fixtures.ts
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

type Json = any;
// eslint-disable-next-line @typescript-eslint/no-require-imports
declare const require: (id: string) => any;
const NOW_MS = Date.UTC(2026, 0, 15, 12, 0, 0);
const RealDate = Date;
class FrozenDate extends RealDate {
  constructor(...args: unknown[]) {
    if (args.length === 0) super(NOW_MS);
    else super(...(args as [string | number]));
  }

  static now(): number {
    return NOW_MS;
  }
}
(globalThis as any).Date = FrozenDate;
const fakeDb = { embeddings: new Map<string, number[]>(), struggle: {} as Json, cluster: {} as Json };
const at = (ms: number | null | undefined) => (ms == null ? null : new Date(ms));
const cardOfTopics = (topics: string[] | undefined) => (topics?.[0] ?? '').replace(/^__card:/, '');
const sameTopics = (a: string[], b: string[]) => JSON.stringify(a) === JSON.stringify(b);
const responseCounts = async ({ where }: Json) => (where.questionId.in as string[])
  .map((questionId) => {
    const stats = fakeDb.struggle.questionAttempts?.[questionId] ?? {};
    return { questionId, _count: { _all: (where.isCorrect ? stats.correct : stats.attempts) ?? 0 } };
  })
  .filter((row) => row._count._all > 0);
const bridgePool = async ({ where, take }: Json) => {
  const question = (fakeDb.struggle.questions ?? []).find((q: Json) => sameTopics(q.topics, where.topics.hasSome));
  const pools = fakeDb.struggle.bridgePools?.[question?.id] ?? {};
  const notIn = new Set<string>(where.id.notIn);
  return ((where.sourceComponent === 'KeyPoint' ? pools.keyPoint : pools.primary) ?? [])
    .filter((row: Json) => !notIn.has(row.id))
    .slice(0, take)
    .map((row: Json) => ({
      id: row.id,
      progress: row.progress
        ? [{ status: row.progress.status, lastReview: at(row.progress.lastReview), nextDueAt: at(row.progress.nextDueAt) }]
        : [],
    }));
};
(globalThis as any).prisma = {
  $queryRaw: async (strings: unknown, ...values: unknown[]) => {
    if (!Array.isArray(strings)) return fakeDb.struggle.bridgeCards ?? [];
    const ids = (values.find(v => Array.isArray(v)) ?? []) as string[];
    return ids.filter(id => fakeDb.embeddings.has(id))
      .map(id => ({ id, embedding: `[${fakeDb.embeddings.get(id)!.join(',')}]` }));
  },
  cluster: { findMany: async () => [] },
  cardProgress: {
    findMany: async ({ where }: Json) => (where.cardId.in as string[])
      .filter((cardId) => fakeDb.struggle.cardProgress?.[cardId])
      .map((cardId) => {
        const row = fakeDb.struggle.cardProgress[cardId];
        return { cardId, ...row, recentFailWindowStart: at(row.recentFailWindowStart) };
      }),
  },
  questionResponse: { groupBy: responseCounts },
  question: {
    findMany: async ({ where }: Json) => (fakeDb.struggle.questions ?? []).filter((q: Json) => where.id.in.includes(q.id)),
    findFirst: async ({ where }: Json) => {
      const variant = fakeDb.struggle.variantQuestions?.[cardOfTopics(where.topics?.hasSome)];
      return variant ? { id: variant.id } : null;
    },
  },
  card: {
    findUnique: async ({ where }: Json) => ({
      topics: [`__card:${where.id}`],
      rotation: fakeDb.struggle.variantQuestions?.[where.id]?.rotation ?? 'rot-main',
    }),
    findMany: bridgePool,
  },
  concept: {
    findFirst: async ({ where }: Json) => (where.rotation ? null : fakeDb.struggle.contextConcepts?.[cardOfTopics(where.topics?.hasSome)] ?? null),
  },
};

const K = '../../src/lib/knowledge/';

// Cluster-fallback storage reads are snapshot adapters: swap the query modules
// in the require cache before the scheduler modules first import them.
function stubModule(path: string, overrides: Json): void {
  const resolved = require.resolve(path);
  const real = require(resolved);
  require.cache[resolved]!.exports = { ...real, ...overrides };
}
const clusterSnapshot = () => fakeDb.cluster;
stubModule('../../src/lib/manifold/clustering', {
  getClusterMastery: async () => clusterSnapshot().clusterMastery ?? [],
});
stubModule(`${K}unified-scheduler-queries`, {
  getCardsForCluster: async (_u: string, _r: string, clusterId: string, max: number, opts: Json) =>
    (clusterSnapshot().clusterCards?.[clusterId] ?? [])
      .filter((row: Json) => !opts.selectedCardIds.has(row.id))
      .slice(0, max),
  getQuestionsForRotation: async (_u: string, _r: string, limit: number, opts: Json) =>
    (clusterSnapshot().clusterQuestions ?? [])
      .filter((row: Json) => !opts.selectedQuestionIds.has(row.id))
      .slice(0, limit),
  getUnseenScaffoldsForClusters: async (_u: string, _r: string, clusterIds: string[], selected: Set<string>) =>
    (clusterSnapshot().clusterScaffolds ?? [])
      .filter((row: Json) => clusterIds.includes(row.clusterId) && !selected.has(row.id)),
});
stubModule('../../src/lib/exam-target/learner-policy', {
  buildExamTargetLearnerPolicy: () => {
    const policy = clusterSnapshot().learnerTargetPolicy;
    if (!policy) throw new Error('fixture requires a learnerTargetPolicy snapshot');
    return {
      ...policy,
      conceptMultipliers: mapOf(policy.conceptMultipliers),
      itemPersonalizedScores: mapOf(policy.itemPersonalizedScores),
      desiredDomains: mapOf(policy.desiredDomains),
    };
  },
});
stubModule(`${K}bulk-candidates`, {
  loadQuestionFamiliarity: async () => new Map(Object.entries(clusterSnapshot().questionFamiliarity ?? {})),
});
const concepts = require(`${K}unified-scheduler-phase-concepts`);
const select = require(`${K}unified-scheduler-phase-select`);
const backfill = require(`${K}unified-scheduler-phase-backfill`);
const queue = require(`${K}unified-scheduler-phase-queue`);
const finalize = require(`${K}unified-scheduler-phase-finalize`);
const clusterSession = require(`${K}unified-scheduler-cluster-session`);
const variants = require(`${K}variant-suppression`);
const retirement = require(`${K}question-retirement`);
const types = require(`${K}unified-scheduler-types`);

const simRegistry = require('../../src/lib/study/simulation-item-map.json');
const DAY = 86_400_000;
const SEED = 'a3f1c9e2b7d4058e6a1f3c5d7e9b2a4c6e8f0a1b3c5d7e9f1a2b3c4d5e6f7a8b';

function toJson(value: unknown): Json {
  return JSON.parse(JSON.stringify(value, (_k, v) => {
    if (typeof v === 'number' && !Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
    if (v instanceof Set) return [...v];
    if (v instanceof Map) return Object.fromEntries(v);
    return v;
  }));
}

const mapOf = (o: Json) => new Map(Object.entries(o ?? {}));
const nested = (o: Json) => new Map(Object.entries(o ?? {}).map(([k, v]) => [k, mapOf(v)]));

function toSessionInputs(s: Json): any {
  const stateMap = new Map<string, any>();
  for (const [id, row] of Object.entries<any>(s.stateMap ?? {})) {
    stateMap.set(id, {
      ...row,
      lastProbeAt: row.lastProbeAt == null ? null : new Date(row.lastProbeAt),
      lastExposureAt: row.lastExposureAt == null ? null : new Date(row.lastExposureAt),
    });
  }
  const options = { suppressSchedulerSideEffects: true, ...s.options };
  for (const key of ['topicTeachingWeeks', 'topicRecencyDays', 'recentFigureExposures']) {
    if (options[key]) options[key] = mapOf(options[key]);
  }
  const b = s.bulk ?? {};
  const rotationQuestions = b.rotationQuestions ?? [];
  const rotationVideos = b.rotationVideos ?? [];
  const bulk = {
    rotation: s.rotation,
    unseenCards: b.unseenCards ?? [],
    seenCards: b.seenCards ?? [],
    seenCardReviewCounts: mapOf(b.seenCardReviewCounts),
    fragileSeenCards: new Set(b.fragileSeenCards ?? []),
    failedStretchCardIds: new Set(b.failedStretchCardIds ?? []),
    passedScaffoldCardIds: new Set(b.passedScaffoldCardIds ?? []),
    questionConceptLinks: b.questionConceptLinks ?? [],
    videoConceptLinks: b.videoConceptLinks ?? [],
    rotationQuestions,
    rotationVideos,
    questionMap: new Map(rotationQuestions.map((q: Json) => [q.id, q])),
    videoMap: new Map(rotationVideos.map((v: Json) => [v.id, v])),
    conceptEmbeddings: mapOf(s.conceptEmbeddings),
    cardScores: nested(b.cardScores),
    questionScores: nested(b.questionScores),
    videoScores: nested(b.videoScores),
    cardGapAlignment: new Map(),
    cardExamRelevance: mapOf(b.cardExamRelevance),
    questionExamRelevance: mapOf(b.questionExamRelevance),
    examTargetItemScores: mapOf(b.examTargetItemScores),
    cardExamTargetScores: new Map(),
    questionExamTargetScores: new Map(),
    examTargetRejectedItemKeys: [],
    cardVariantGroupHistory: new Map(Object.entries<any>(b.cardVariantGroupHistory ?? {}).map(([k, v]) => [k, new Set(v)])),
    questionFamiliarity: mapOf(b.questionFamiliarity),
    passSetDeferredCardIds: new Set(b.passSetDeferredCardIds ?? []),
    simulationPreparation: (b.simulationPreparation ?? []).map((r: Json) => ({
      ...r,
      lastReview: r.lastReview == null ? null : new Date(r.lastReview),
      nextDueAt: r.nextDueAt == null ? null : new Date(r.nextDueAt),
    })),
  };
  const pc = s.penaltyContext;
  return {
    userId: 'fixture-user', options, rotation: s.rotation, size: s.size,
    interferenceThreshold: s.interferenceThreshold ?? 0.85, includeVideos: Boolean(s.includeVideos),
    selectionDeterminism: null, servingRotationSeed: s.servingRotationSeed, nowMs: s.nowMs,
    selectionNowMs: () => s.nowMs, daysToExam: s.daysToExam, examPressure: s.examPressure, cardRatio: s.cardRatio,
    effectiveMaxNewCards: s.effectiveMaxNewCards ?? undefined,
    excludedCardIds: new Set(s.excludedCardIds ?? []), excludedQuestionIds: new Set(s.excludedQuestionIds ?? []),
    excludedVideoIds: new Set(s.excludedVideoIds ?? []),
    penaltyContext: pc ? {
      recentTopicExposures: mapOf(pc.recentTopicExposures), recentCardIds: new Set(pc.recentCardIds ?? []),
      recentFailureConceptIds: new Set(pc.recentFailureConceptIds ?? []), nowMs: pc.nowMs,
    } : undefined,
    recentClusterExposures: mapOf(s.recentClusterExposures),
    concepts: s.concepts, conceptMap: new Map(s.concepts.map((c: Json) => [c.id, c])),
    runtimeExamTarget: s.runtimeExamTarget ?? undefined, runtimeTargetSnapshot: s.runtimeTargetSnapshot ?? null,
    examTargetEvaluationOnly: Boolean(s.examTargetEvaluationOnly), effectiveTargetInfluence: s.effectiveTargetInfluence ?? null,
    targetComputeRequested: Boolean(s.targetComputeRequested), targetSidecarsValid: Boolean(s.targetSidecarsValid),
    applyExamTarget: Boolean(s.applyExamTarget),
    conceptTargetScores: { scores: new Map() }, observedExamTargetDomainCounts: new Map(), masteryEvidence: null,
    eligibleCoreUnitIds: new Set(), eligibleAppliedUnitIds: new Set(), targetWorkload: s.targetWorkload ?? null,
    stateMap, conceptEmbeddings: mapOf(s.conceptEmbeddings), gapDirection: s.gapDirection ?? null,
    likedConceptIds: new Set(s.likedConceptIds ?? []), chronicFailureConceptIds: new Set(s.chronicFailureConceptIds ?? []),
    practiceMissConceptIds: new Set(s.practiceMissConceptIds ?? []),
    recentFailureConceptIds: new Set(s.recentFailureConceptIds ?? []),
    recentFormatsByConcept: new Map(Object.entries<any>(s.recentFormatsByConcept ?? {}).map(([k, v]) => [k, new Set(v)])),
    conceptThreadAnchors: [], ratingReliability: null, budget: s.budget,
    bulk, flowAxisPromise: Promise.resolve(s.flowAxis ? { axis: s.flowAxis } : null),
  };
}

function topUpAdapter(inputs: any, plan: any, build: any, rows: Json[]): void {
  const { size, rotation, bulk } = inputs;
  if (build.selectedItems.length >= size) return;
  const deficit = Math.min(size - build.selectedItems.length, Math.max(0, plan.targetQuestionCount - build.selectedQuestionCount));
  if (deficit <= 0) return;
  const alreadyPicked = new Set<string>(build.selectedQuestionIds);
  for (const item of build.selectedItems) {
    if (item.type === 'question') alreadyPicked.add(item.id);
  }
  const selected: Json[] = [];
  for (const row of rows) {
    const key = variants.questionSuppressionKey(row);
    if (alreadyPicked.has(row.id) || (key && build.selectedQuestionVariantGroups.has(key))) continue;
    selected.push(row);
    if (selected.length === deficit) break;
  }
  build.masteredReentryCounter.masteredServed += retirement.countMastered(selected, (q: Json) => q.id, bulk.questionFamiliarity);
  for (const q of selected) {
    build.selectedQuestionIds.add(q.id);
    const key = variants.questionSuppressionKey(q);
    if (key) build.selectedQuestionVariantGroups.add(key);
  }
  for (const q of selected) {
    build.addItem({
      type: 'question', id: q.id, rotation, conceptId: `rotation:${q.id}`, conceptName: 'Rotation fill',
      priority: 0.4, interventionReason: 'needs_retest', ...types.conceptThreadPolicyReceipt(),
      variantGroupId: q.variantGroupId, variantType: q.variantType,
    });
  }
}

async function runSession(s: Json): Promise<Json> {
  fakeDb.embeddings = new Map(Object.entries<number[]>(s.itemEmbeddings ?? {}));
  fakeDb.struggle = s.struggleHistory ?? {};
  fakeDb.cluster = s;
  const inputs = toSessionInputs(s);
  if (inputs.concepts.length === 0) {
    const result = await clusterSession.constructClusterSession('fixture-user', {
      rotation: s.rotation, size: s.size, cardRatio: s.cardRatio, maxNewCards: s.effectiveMaxNewCards ?? Infinity,
      minFirstSightItems: s.options.minFirstSightItems ?? 0, excludeCardIds: [], excludeQuestionIds: [],
      practiceLocale: 'au', crossSourceRotations: [], crossSourceMappingMode: 'adjacent',
    });
    return toJson({ result });
  }
  const ranking = concepts.rankConcepts(inputs);
  concepts.sortConceptsByPriority(inputs, ranking);
  const pristine = concepts.findPristineConcepts(inputs);
  const selection = concepts.selectWeakConcepts(inputs, ranking, pristine);
  const quota = concepts.planCardQuota(inputs, selection.cardRatio);
  const plan = { ...ranking, ...selection, ...quota, conceptHasPristine: pristine };
  const build = select.createSessionBuild(inputs, plan);
  select.fillSimulationPreparation(inputs, build);
  select.reserveScheduledCore(inputs, plan, build);
  select.reserveAppliedSurplus(inputs, plan, build);
  select.holdBackUnearnedStages(inputs, build);
  select.reserveSessionPrepBlend(inputs, build);
  const laneCounts: Json[] = [];
  const lane = (id: string, fn: () => void) => { fn(); laneCounts.push({ id, result: build.selectedItems.length }); };
  lane('pre_teach_videos', () => select.preTeachVideos(inputs, plan, build));
  lane('pre_teach_naive_concepts', () => select.preTeachNaiveConcepts(inputs, plan, build));
  lane('fill_questions', () => select.fillQuestions(inputs, plan, build));
  lane('fill_cards', () => select.fillCards(inputs, plan, build));
  lane('fill_coverage_lane', () => select.fillCoverageLane(inputs, plan, build));
  lane('backfill_from_any_concept', () => backfill.backfillFromAnyConcept(inputs, plan, build));
  lane('probe_stale_strong_concepts', () => backfill.probeStaleStrongConcepts(inputs, plan, build));
  lane('top_up_from_rotation_bank', () => topUpAdapter(inputs, plan, build, s.rotationTopUp ?? []));
  lane('backfill_cards_uncapped', () => backfill.backfillCardsUncapped(inputs, plan, build));
  lane('rescue_orphans', () => backfill.rescueOrphans(inputs, plan, build));
  backfill.releaseFirstSightReservation(inputs, build);
  const selectedIds = build.selectedItems.map((i: Json) => `${i.type}:${i.id}`);
  const struggled = await queue.applyStruggleToSelection(inputs, build);
  const { finalOrdered, cardMetaById } = await queue.orderSessionItems(inputs, plan, build, struggled);
  const telemetry = finalize.attachDeliveryTelemetry(inputs, plan, build, finalOrdered, cardMetaById);
  const result = finalize.assembleSessionResult(inputs, plan, build, finalOrdered, telemetry);
  return toJson({ result, laneCounts, selectedIds });
}

const ago = (days: number) => NOW_MS - days * DAY;
const st = (recallProbability: number, confidence: number, exposureCount: number, exposed: number | null, probed: number | null, extra: Json = {}) => ({
  recallProbability, confidence, exposureCount, probeCount: exposureCount, recentFailRate: 0,
  lastExposureAt: exposed == null ? null : ago(exposed), lastProbeAt: probed == null ? null : ago(probed), ...extra,
});
const card = (id: string, topics: string[], extra: Json = {}) => ({
  id, stableId: id, rotation: 'rot-main', clusterId: null, similarCards: null, topics, sourceFile: `week2-${id}`,
  imageUrl: null, clipId: null, importance: 1, complexity: 2, facilityIndex: null, sampleSize: null,
  variantGroupId: null, variantIndex: null, variantType: null, ...extra,
});
const question = (id: string, topics: string[], extra: Json = {}) => ({
  id, rotation: 'rot-main', stem: `stem ${id}`, questionType: 'diagnosis', difficulty: 'medium', variantGroupId: null,
  variantType: null, format: null, facilityIndex: null, totalAttempts: null, source: 'bank', topics,
  examRelevance: null, examRelevancePct: null, ...extra,
});
const concept = (id: string, topics: string[], extra: Json = {}) => ({
  id, name: id.toUpperCase(), week: null, topics, prerequisiteIds: [], examWeight: 3, ...extra,
});

function baseSession(): Json {
  return {
    userId: 'fixture-user', nowMs: NOW_MS, daysToExam: 30, examPressure: 1 / (1 + Math.exp(-0.15 * (21 - 30))), size: 12, cardRatio: 0.7,
    rotation: 'rot-main', effectiveMaxNewCards: null, interferenceThreshold: 0.85, includeVideos: true,
    servingRotationSeed: `serving:${SEED}`, budget: { dailyThroughput: 41.6, totalRemaining: 812.4, sessionsRemaining: 9 },
    options: { examTargetTieBreakSeed: SEED },
    concepts: [
      concept('c-renal', ['Renal Tubular Acidosis'], { examWeight: 4 }),
      concept('c-cardiac', ['Cardiac Output']),
      concept('c-acid', ['Acid-Base'], { examWeight: 5 }),
      concept('c-glyco', ['Glycolysis'], { examWeight: 1 }),
      concept('c-naive', ['Sickle Cell']),
    ],
    stateMap: {
      'c-renal': st(0.3, 0.4, 5, 2, 4),
      'c-cardiac': st(0.55, 0.2, 2, 6, 6),
      'c-acid': st(0.97, 0.9, 12, 1, 9),
      'c-glyco': st(0.6, 0.75, 8, 3, 5),
    },
    recentFailureConceptIds: ['c-renal'],
    bulk: {
      unseenCards: [
        card('u-renal-1', ['Renal Tubular Acidosis'], { clusterId: 'cl-renal', complexity: 3, importance: 3 }),
        card('u-renal-2', ['renal tubular acidosis'], { clusterId: 'cl-renal', complexity: 4, similarCards: [{ cardId: 'u-renal-1', similarity: 0.93 }] }),
        card('u-renal-s', ['Renal Tubular Acidosis'], { clusterId: 'cl-renal', complexity: 1 }),
        card('u-cardiac-1', ['Cardiac Output'], { clusterId: 'cl-cardio', imageUrl: 'fig-1.png' }),
        card('u-cardiac-2', ['Cardiac Output'], { clusterId: 'cl-cardio', variantGroupId: 'vg-1', variantIndex: 0, variantType: 'cloze' }),
        card('u-cardiac-3', ['Cardiac Output'], { clusterId: 'cl-cardio', variantGroupId: 'vg-1', variantIndex: 1, variantType: 'cloze' }),
        card('u-acid-1', ['Acid-Base'], { importance: 2 }),
        card('u-sickle-s', ['Sickle Cell'], { complexity: 1 }),
        card('u-orphan', ['Unrelated Topic'], { importance: 2 }),
      ],
      seenCards: [
        card('s-glyco-1', ['Glycolysis'], { clusterId: 'cl-metab', sourceFile: 'bank-cards' }),
        card('s-glyco-2', ['Glycolysis'], { clusterId: 'cl-metab', facilityIndex: 0.8, sampleSize: 30 }),
        card('s-renal-1', ['Renal Tubular Acidosis'], { clusterId: 'cl-renal', complexity: 5 }),
        card('s-orphan', ['Lonely Topic']),
      ],
      seenCardReviewCounts: { 's-glyco-1': 2 },
      fragileSeenCards: ['s-glyco-2'],
      rotationQuestions: [
        question('q-renal-1', ['Renal Tubular Acidosis'], { difficulty: ' Hard ' }),
        question('q-renal-2', ['Renal Tubular Acidosis'], { difficulty: 'easy', facilityIndex: 0.2, totalAttempts: 40 }),
        question('q-cardiac-1', ['Cardiac Output'], { difficulty: 'bogus', variantGroupId: 'qg-1', variantType: 'near-duplicate' }),
        question('q-cardiac-2', ['Cardiac Output'], { variantGroupId: 'qg-1', variantType: 'near-duplicate' }),
        question('q-acid-1', ['Acid-Base'], { difficulty: 'hard', source: 'generated' }),
        question('q-glyco-1', ['Glycolysis'], { difficulty: 'medium', format: 'cloze' }),
        question('q-orphan', ['Other Topic']),
      ],
      questionConceptLinks: [
        { questionId: 'q-renal-1', conceptId: 'c-renal', isPrimary: true },
        { questionId: 'q-cardiac-1', conceptId: 'c-cardiac', isPrimary: false },
      ],
      rotationVideos: [{ id: 'vid-c-renal', title: 'RTA', thumbnailR2Key: 't.png', durationSecs: 300, r2Key: 'v.mp4', creatorName: null }],
      videoConceptLinks: [],
      cardScores: { 'c-cardiac': { 'u-cardiac-2': 0.9, 'u-cardiac-1': 0.7 } },
      questionScores: { 'c-renal': { 'q-renal-1': 0.4 } },
      questionFamiliarity: { 'q-glyco-1': { lastSeenAtMs: ago(3), corrects: 1 } },
    },
    itemEmbeddings: {
      'u-renal-1': [0.9, 0.1, 0, 0.1], 'u-cardiac-1': [0.1, 0.9, 0.1, 0], 'u-acid-1': [0.5, 0.5, 0.2, 0.1],
      'q-renal-1': [0.8, 0.2, 0.1, 0.3], 's-glyco-1': [0, 0.2, 0.9, 0.1], 'u-sickle-s': [0.3, 0.1, 0.2, 0.9],
    },
    rotationTopUp: [],
  };
}

function sessionVariant(name: string, mutate: (s: Json) => void = () => {}): { name: string; snapshot: Json } {
  const snapshot = baseSession();
  mutate(snapshot);
  return { name, snapshot };
}

const preEclampsia = simRegistry.records.filter((r: Json) => r.topic === 'pre-eclampsia');

function simulationSession(s: Json): void {
  const [scaffold, reviewed, prepared] = preEclampsia.map((r: Json) => r.stableId);
  s.options.simulationFocus = {
    fingerprint: 'fixture-focus', deadline: '2026-02-01', topics: ['pre-eclampsia'], preparationStableIds: [prepared],
  };
  s.concepts.push(concept('c-pe', ['Pre-eclampsia'], { examWeight: 4 }), concept('c-ecl', ['Eclampsia']));
  s.stateMap['c-ecl'] = st(0.4, 0.5, 4, 2, 3);
  s.bulk.unseenCards.push(
    card('u-pe-plain-s', ['Pre-eclampsia'], { stableId: null, complexity: 1 }),
    card('u-pe-s', ['Pre-eclampsia'], { stableId: scaffold, complexity: 1 }),
    card('u-ecl-plain', ['Eclampsia'], { stableId: null }),
    card('u-ecl-reviewed', ['Eclampsia'], { stableId: reviewed }),
    card('u-ecl-prepared', ['Eclampsia'], { stableId: prepared }),
  );
  const prep = (id: string, extra: Json) => ({
    id, stableId: `prep-${id}`, lastQuality: null, totalReviews: 0, lastReview: null, nextDueAt: null,
    card: { variantGroupId: null, variantIndex: null, variantType: null }, ...extra,
  });
  s.bulk.simulationPreparation = [
    prep('sp-unknown', {}),
    prep('sp-recalled', { lastQuality: 5, totalReviews: 3, lastReview: ago(2), nextDueAt: ago(-5) }),
    prep('sp-due', { lastQuality: 4, totalReviews: 2, lastReview: ago(9), nextDueAt: ago(1) }),
    prep('sp-weak', { lastQuality: 1, totalReviews: 1, lastReview: ago(3), nextDueAt: ago(-1) }),
    prep('sp-weak-older', { lastQuality: 1, totalReviews: 4, lastReview: ago(6), nextDueAt: null, card: { variantGroupId: 'pg', variantIndex: 0, variantType: 'cloze' } }),
  ];
  const ids = new Set([scaffold, reviewed, prepared]);
  s.simulationItemMap = simRegistry.records.filter((r: Json) => ids.has(r.stableId));
}

function struggleSession(s: Json, withLookups: boolean): void {
  const progress = (recentFailCount: number, windowHoursAgo: number | null, totalReviews: number, correctCount: number) => ({
    recentFailCount, recentFailWindowStart: windowHoursAgo == null ? null : NOW_MS - windowHoursAgo * 3_600_000,
    totalReviews, correctCount,
  });
  const cards = ['u-renal-1', 'u-cardiac-1', 'u-acid-1', 's-glyco-1', 's-glyco-2', 'u-cardiac-2', 'u-sickle-s', 'u-orphan'];
  s.struggleHistory = {
    cardProgress: {
      'u-renal-1': progress(3, 2, 4, 1), 'u-cardiac-1': progress(4, 30, 5, 1), 'u-acid-1': progress(1, 1, 6, 3),
      's-glyco-1': progress(3, 5, 3, 2), 's-glyco-2': progress(0, null, 10, 2), 'u-cardiac-2': progress(5, 1, 5, 0),
      'u-sickle-s': progress(2, 1, 2, 0), 'u-orphan': progress(3, 23, 3, 1),
    },
    questionAttempts: { 'q-renal-1': { attempts: 5, correct: 1 }, 'q-cardiac-1': { attempts: 4, correct: 1 }, 'q-acid-1': { attempts: 2, correct: 0 } },
    questions: [
      { id: 'q-cardiac-1', topics: ['Cardiac Output'], rotation: 'rot-main' },
      { id: 'q-renal-1', topics: ['Renal Tubular Acidosis'], rotation: 'rot-main' },
    ],
    bridgePools: {
      'q-renal-1': {
        primary: [
          { id: 'b-cooldown', progress: { status: 'mastered', lastReview: ago(0.5), nextDueAt: ago(1) } },
          { id: 'b-fresh', progress: null },
          { id: 'b-due', progress: { status: 'reviewing', lastReview: ago(3), nextDueAt: ago(1) } },
          { id: 'u-renal-1', progress: null },
        ],
      },
      'q-cardiac-1': { primary: [{ id: 'b-not-due', progress: { status: 'mastered', lastReview: ago(4), nextDueAt: ago(-2) } }], keyPoint: [{ id: 'kp-cardiac', progress: null }] },
    },
    bridgeCards: withLookups ? [{ id: 'bridge-a' }, { id: 'bridge-b' }] : [],
    contextConcepts: withLookups ? Object.fromEntries(cards.map((id) => [id, { id: `ctx-${id}`, name: 'Context', description: 'Teaching note' }])) : {},
    variantQuestions: withLookups ? Object.fromEntries(cards.map((id) => [id, { id: `vq-${id}`, rotation: 'rot-main' }])) : {},
  };
}

function examTargetSession(s: Json, mode: 'active' | 'shadow', assignment: string, applied: boolean): void {
  const influence = { allocator: 'active', maxItemRankMove: 4 };
  s.targetComputeRequested = true;
  s.targetSidecarsValid = true;
  s.applyExamTarget = applied;
  s.runtimeTargetSnapshot = {
    id: 'target-snapshot-1', targetVersion: 'target-v1', targetBasis: 'fixture-blueprint', scorerVersion: 'scorer-v1',
    definition: { influence },
  };
  s.runtimeExamTarget = {
    schedulerVersion: 'scheduler-v1', policyDigest: 'digest-1', activationRevision: 3,
    resolved: { effectiveMode: mode, assignment },
  };
  s.effectiveTargetInfluence = influence;
  s.targetWorkload = { coreTargetSeats: 0, surplusSeats: 0, shortfall: null, dailyCapacity: 40 };
  const cardIds = [...(s.bulk.unseenCards ?? []), ...(s.bulk.seenCards ?? [])].map((c: Json) => c.id);
  const questionIds = (s.bulk.rotationQuestions ?? []).map((q: Json) => q.id);
  const keys = [...cardIds.map((id: string) => `card:${id}`), ...questionIds.map((id: string) => `question:${id}`)];
  const domains = ['RENAL', 'CARDIO', 'HEME'];
  s.bulk.examTargetItemScores = Object.fromEntries(keys.map((key, i) => [key, {
    domainCode: domains[i % 3], sourceRotation: 'rot-main', embeddingHash: `emb-${i}`, fitPercentile: (i * 13) % 100,
    effectiveDomainWeight: 0.25 + (i % 3) * 0.125, itemTargetIndex: ((i * 7) % 10) / 10, weightProvenance: 'fixture',
  }]));
  s.learnerTargetPolicy = {
    conceptMultipliers: { 'c-renal': 1.4, 'c-cardiac': 0.6, 'c-acid': 1.1 },
    itemPersonalizedScores: Object.fromEntries(keys.filter((_k, i) => i % 4 !== 3).map((key, i) => [key, ((i * 37) % 100) / 100])),
    desiredDomains: {
      RENAL: { desiredShare: 0.5, userDomainGap: 0.3 },
      CARDIO: { desiredShare: 0.3, userDomainGap: 0.1 },
      NEURO: { desiredShare: 0.2, userDomainGap: 0.4 },
    },
    unmappedDomainCodes: ['HEME'],
  };
}

function clusterFallbackSession(s: Json): void {
  s.concepts = [];
  const card = (id: string, clusterId: string, complexity: number, firstSight: boolean, group: string | null = null) => ({
    id, rotation: 'rot-main', clusterId, similarCards: [], topics: [`Topic ${id}`], sourceFile: `${clusterId}.md`,
    importance: 3, complexity, variantGroupId: group, variantIndex: group ? 0 : null, variantType: group ? 'cloze' : null,
    firstSightAtSelection: firstSight,
  });
  const mastery = (clusterId: string, clusterName: string, m: number, cardCount: number, cardsReviewed: number, needsAttention: boolean) => ({
    clusterId, clusterName, mastery: m, cardCount, cardsReviewed, avgRetrieval: m, needsAttention,
  });
  s.clusterMastery = [
    mastery('cl-acid', 'Acid-base', 0, 0, 0, false),
    mastery('cl-renal', 'Renal', 0.2, 10, 2, true),
    mastery('cl-cardiac', 'Cardiac', 0.7, 8, 6, false),
  ];
  s.clusterCards = {
    'cl-renal': [card('k-renal-1', 'cl-renal', 2, true, 'g-renal'), card('k-renal-2', 'cl-renal', 2, false, 'g-renal'),
      card('k-renal-3', 'cl-renal', 3, false), card('k-renal-4', 'cl-renal', 4, true)],
    'cl-cardiac': [card('k-card-1', 'cl-cardiac', 2, false), card('k-card-2', 'cl-cardiac', 3, true), card('k-card-3', 'cl-cardiac', 4, false)],
    'cl-acid': [card('k-acid-1', 'cl-acid', 2, true)],
  };
  s.clusterQuestions = ['cq-1', 'cq-2', 'cq-3', 'cq-4'].map((id, i) => ({
    id, rotation: 'rot-main', topics: [`Question ${id}`], variantGroupId: i < 2 ? 'qg-1' : null, variantType: i < 2 ? 'mcq' : null,
  }));
  s.questionFamiliarity = { 'cq-2': { attempts: 2, correct: 1 } };
  s.clusterScaffolds = [
    { ...card('sc-renal', 'cl-renal', 1, true), topics: ['Topic k-renal-1'] },
    { ...card('sc-cardiac', 'cl-cardiac', 1, true), topics: ['Topic k-card-2'] },
    { ...card('sc-other', 'cl-other', 1, true), topics: ['Topic k-acid-1'] },
  ];
}

const sessionCases = [
  sessionVariant('full_session_weak_new_pristine'),
  sessionVariant('empty_pools', s => { s.bulk = { rotationQuestions: [], unseenCards: [], seenCards: [] }; s.itemEmbeddings = {}; }),
  sessionVariant('backfill_and_orphan_rescue', s => { s.size = 20; }),
  sessionVariant('exhausted_backfill', s => { s.size = 40; }),
  sessionVariant('rotation_bank_top_up', s => {
    s.size = 16; s.cardRatio = 0.3;
    s.rotationTopUp = [question('q-top-1', ['Misc'], { variantGroupId: 'tg', variantType: 'contrast-set' }), question('q-glyco-1', ['Glycolysis'])];
  }),
  sessionVariant('first_sight_reservation_release', s => { s.options.minFirstSightItems = 9; s.size = 14; }),
  sessionVariant('interference_tight_threshold', s => { s.interferenceThreshold = 0.5; s.size = 16; }),
  sessionVariant('cluster_dampening', s => { s.recentClusterExposures = { 'cl-renal': 9, 'cl-cardio': 5 }; }),
  sessionVariant('crunch_no_new_cards', s => { s.effectiveMaxNewCards = 0; s.daysToExam = 2; s.examPressure = 1 / (1 + Math.exp(-0.15 * 19)); }),
  sessionVariant('review_challenge_harder', s => { s.options.reviewChallenge = 1; s.size = 14; }),
  sessionVariant('review_challenge_easier', s => { s.options.reviewChallenge = -1; }),
  sessionVariant('struggle_step_down_scaffolds', s => { s.recentFailureConceptIds = ['c-renal', 'c-cardiac']; s.size = 16; }),
  sessionVariant('no_recent_failures_no_scaffolds', s => { s.recentFailureConceptIds = []; }),
  sessionVariant('penalty_context_and_formats', s => {
    s.penaltyContext = { recentTopicExposures: { Glycolysis: { count: 2, mostRecentMs: NOW_MS - 3_600_000 } }, recentCardIds: ['u-renal-1'], recentFailureConceptIds: [], nowMs: NOW_MS };
    s.recentFormatsByConcept = { 'c-glyco': ['cloze'] };
  }),
  sessionVariant('manifold_flow_axis', s => { s.flowAxis = [0.1, 0.2, 0.9, 0.3]; s.options.commitmentLevel = 'student'; }),
  sessionVariant('cross_source_cap', s => {
    s.options.crossSourceRotations = ['rot-cross']; s.options.maxCrossSourceItems = 1;
    s.bulk.unseenCards[3].rotation = 'rot-cross'; s.bulk.unseenCards[6].rotation = 'rot-cross';
  }),
  sessionVariant('teaching_pace_weeks', s => {
    s.rotation = 'paam'; s.options.currentTeachingWeek = 2;
    for (const row of [...s.bulk.unseenCards, ...s.bulk.seenCards, ...s.bulk.rotationQuestions]) row.rotation = 'paam';
    s.options.topicTeachingWeeks = { 'cardiac-output': 2, glycolysis: 4 };
  }),
  sessionVariant('excluded_items', s => { s.excludedCardIds = ['u-renal-1', 's-glyco-1']; s.excludedQuestionIds = ['q-renal-1']; }),
  sessionVariant('simulation_focus_preparation', s => { simulationSession(s); s.size = 16; }),
  sessionVariant('simulation_focus_recent_failure', s => { simulationSession(s); s.recentFailureConceptIds = ['c-renal', 'c-ecl']; }),
  sessionVariant('simulation_preparation_fills_session', s => { simulationSession(s); s.size = 3; }),
  sessionVariant('review_challenge_hardest_learning_gap', s => {
    s.options.reviewChallenge = 2;
    s.stateMap['c-renal'] = st(0.3, 0.4, 5, 2, 0.5, { recentFailRate: 0.5 });
    s.stateMap['c-cardiac'] = st(0.55, 0.2, 2, 6, 3, { recentFailRate: 0.4 });
  }),
  sessionVariant('struggle_history_interventions', s => { struggleSession(s, true); s.size = 16; }),
  sessionVariant('struggle_history_lookups_missing', s => { struggleSession(s, false); s.size = 16; }),
  sessionVariant('empty_concepts_cluster_fallback', s => { s.concepts = []; }),
  sessionVariant('exam_target_active', s => examTargetSession(s, 'active', 'treatment', true)),
  sessionVariant('exam_target_shadow_mode', s => examTargetSession(s, 'shadow', 'treatment', false)),
  sessionVariant('exam_target_control_assignment', s => examTargetSession(s, 'active', 'control', false)),
  sessionVariant('exam_target_invalid_sidecars', s => {
    examTargetSession(s, 'active', 'treatment', false);
    s.targetSidecarsValid = false;
  }),
  sessionVariant('card_exam_relevance_without_target', s => {
    s.bulk.cardExamRelevance = Object.fromEntries([...(s.bulk.unseenCards ?? []), ...(s.bulk.seenCards ?? [])]
      .map((c: Json, i: number) => [c.id, ((i * 29) % 100) / 100]));
    s.bulk.questionExamRelevance = Object.fromEntries((s.bulk.rotationQuestions ?? []).map((q: Json, i: number) => [q.id, ((i * 31) % 100) / 100]));
  }),
  sessionVariant('cluster_present_fallback', s => { clusterFallbackSession(s); s.size = 10; }),
  sessionVariant('cluster_present_first_sight_quota', s => {
    clusterFallbackSession(s);
    s.size = 6;
    s.options.minFirstSightItems = 3;
  }),
];

const fixture = {
  generator: 'python/tools/generate_unified_scheduler_session_fixtures.ts',
  seed: SEED,
  nowMs: NOW_MS,
  sessions: [] as Json[],
};
async function main(): Promise<void> {
for (const c of sessionCases) {
  fixture.sessions.push({ name: c.name, snapshot: c.snapshot, expected: await runSession(c.snapshot) });
}
const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '..', 'tests', 'fixtures', 'unified-scheduler-session.cases.json');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(fixture, null, 2) + '\n');
console.log(`wrote ${out} (${fixture.sessions.length} sessions)`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
