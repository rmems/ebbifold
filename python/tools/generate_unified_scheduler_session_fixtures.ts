/**
 * Generates python/tests/fixtures/unified-scheduler-session.cases.json: end-to-end
 * session construction through the real TypeScript phase functions.
 *
 * Storage adapters (the only non-original code paths):
 * - globalThis.prisma is a fake that serves item embeddings ($queryRaw) and an
 *   empty cluster table (cluster fallback) from the snapshot.
 * - rotation top-up rows come from snapshot.rotationTopUp (what
 *   getQuestionsForRotation would return), mirroring topUpFromRotationBank.
 * - struggle history is empty, so applyStruggleToSelection is the identity.
 *
 * Run: node --import tsx python/tools/generate_unified_scheduler_session_fixtures.ts
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

type Json = any;
// eslint-disable-next-line @typescript-eslint/no-require-imports
declare const require: (id: string) => any;
const fakeDb = { embeddings: new Map<string, number[]>() };
(globalThis as any).prisma = {
  $queryRaw: async (_strings: unknown, ...values: unknown[]) => {
    const ids = (values.find(v => Array.isArray(v)) ?? []) as string[];
    return ids.filter(id => fakeDb.embeddings.has(id))
      .map(id => ({ id, embedding: `[${fakeDb.embeddings.get(id)!.join(',')}]` }));
  },
  cluster: { findMany: async () => [] },
};

const K = '../../src/lib/knowledge/';
const concepts = require(`${K}unified-scheduler-phase-concepts`);
const select = require(`${K}unified-scheduler-phase-select`);
const backfill = require(`${K}unified-scheduler-phase-backfill`);
const queue = require(`${K}unified-scheduler-phase-queue`);
const finalize = require(`${K}unified-scheduler-phase-finalize`);
const clusterSession = require(`${K}unified-scheduler-cluster-session`);
const variants = require(`${K}variant-suppression`);
const retirement = require(`${K}question-retirement`);
const types = require(`${K}unified-scheduler-types`);

const NOW_MS = Date.UTC(2026, 0, 15, 12, 0, 0);
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
    examTargetItemScores: new Map(),
    cardExamTargetScores: new Map(),
    questionExamTargetScores: new Map(),
    examTargetRejectedItemKeys: [],
    cardVariantGroupHistory: new Map(Object.entries<any>(b.cardVariantGroupHistory ?? {}).map(([k, v]) => [k, new Set(v)])),
    questionFamiliarity: mapOf(b.questionFamiliarity),
    passSetDeferredCardIds: new Set(b.passSetDeferredCardIds ?? []),
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
    runtimeExamTarget: undefined, runtimeTargetSnapshot: null, examTargetEvaluationOnly: false,
    effectiveTargetInfluence: null, targetComputeRequested: false, targetSidecarsValid: false, applyExamTarget: false,
    conceptTargetScores: { scores: new Map() }, observedExamTargetDomainCounts: new Map(), masteryEvidence: null,
    eligibleCoreUnitIds: new Set(), eligibleAppliedUnitIds: new Set(), targetWorkload: null,
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
  const selected = rows.slice(0, deficit);
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
  const { finalOrdered, cardMetaById } = await queue.orderSessionItems(inputs, plan, build, build.selectedItems);
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
    nowMs: NOW_MS, daysToExam: 30, examPressure: 1 / (1 + Math.exp(-0.15 * (21 - 30))), size: 12, cardRatio: 0.7,
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
  sessionVariant('empty_concepts_cluster_fallback', s => { s.concepts = []; }),
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
