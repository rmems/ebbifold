/**
 * Golden-fixture generator for the Python port, Phase 0.
 *
 * One-time generator, NOT a runtime dependency: it feeds fixed inputs through
 * the TypeScript original (src/lib/usmle/step1-adaptive.ts) and writes the
 * input/output pairs as JSON under python/tests/fixtures/. Regenerate with:
 *
 *   node --import tsx python/tools/generate_step1_adaptive_fixtures.ts
 *
 * from the repository root. The output is committed; Python tests consume the
 * JSON and never execute TypeScript.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  rankAdaptive,
  type AdaptiveQuestion,
  type AdaptiveHistoryRow,
} from '../../src/lib/usmle/step1-adaptive';

interface FixtureHistoryRow {
  questionId: string;
  isCorrect: boolean;
  /** ISO 8601 — Date is not JSON-portable. */
  createdAt: string;
}

interface FixtureCase {
  name: string;
  notes: string;
  input: {
    questions: AdaptiveQuestion[];
    history: FixtureHistoryRow[];
    size: number;
  };
  expected: {
    order: string[];
  };
}

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, '..', 'tests', 'fixtures');

function minutesAfter(base: number, minutes: number): Date {
  return new Date(base + minutes * 60_000);
}

const EPOCH = Date.UTC(2026, 0, 1, 12, 0, 0);

/**
 * One shared pool keeps the cases comparable: the same learner faces the same
 * questions and only their history changes. Ladders: L1 is a complete
 * easy/medium/hard cardio ladder; L2 spans cardio and renal; renal/neuro
 * questions give the scaffold preferences something to rank against.
 */
const POOL: AdaptiveQuestion[] = [
  { id: 'q-easy-1', difficulty: 'easy', domain: 'cardio', ladderId: 'L1', preferenceRank: 1 },
  { id: 'q-easy-2', difficulty: 'easy', domain: 'cardio', ladderId: 'L1', preferenceRank: 2 },
  { id: 'q-med-1', difficulty: 'medium', domain: 'cardio', ladderId: 'L1', preferenceRank: 1 },
  { id: 'q-med-2', difficulty: 'medium', domain: 'cardio', ladderId: 'L2', preferenceRank: 2 },
  { id: 'q-med-3', difficulty: 'moderate', domain: 'renal', ladderId: 'L2', preferenceRank: 3 },
  { id: 'q-med-4', difficulty: 'medium', domain: 'neuro', preferenceRank: 4 },
  { id: 'q-hard-1', difficulty: 'hard', domain: 'cardio', ladderId: 'L1', preferenceRank: 1 },
  { id: 'q-hard-2', difficulty: 'hard', domain: 'renal', ladderId: 'L3', preferenceRank: 2 },
];

function row(questionId: string, isCorrect: boolean, minute: number): AdaptiveHistoryRow {
  return { questionId, isCorrect, createdAt: minutesAfter(EPOCH, minute) };
}

function toFixture(rows: AdaptiveHistoryRow[]): FixtureHistoryRow[] {
  return rows.map((r) => ({
    questionId: r.questionId,
    isCorrect: r.isCorrect,
    createdAt: r.createdAt.toISOString(),
  }));
}

interface Scenario {
  name: string;
  notes: string;
  questions?: AdaptiveQuestion[];
  history: AdaptiveHistoryRow[];
  size: number;
}

const SCENARIOS: Scenario[] = [
  {
    name: 'cold-start',
    notes: 'No history: target medium; order falls to preferenceRank then id.',
    history: [],
    size: 5,
  },
  {
    name: 'one-correct-medium-stays',
    notes: 'Streak 1 < PROMOTION_STREAK: a single correct answer does not promote.',
    history: [row('q-med-1', true, 0)],
    size: 5,
  },
  {
    name: 'two-correct-medium-promotes',
    notes: 'Two consecutive correct answers at medium move the target up to hard.',
    history: [row('q-med-1', true, 0), row('q-med-2', true, 1)],
    size: 5,
  },
  {
    name: 'miss-medium-demotes-and-scaffolds',
    notes:
      'A miss drops the target to easy and scaffolds: the missed ladder L1 first, ' +
      'then its domain cardio, before widening. Preferences, not filters.',
    history: [row('q-med-1', false, 0)],
    size: 6,
  },
  {
    name: 'miss-at-easy-floor',
    notes: 'Missing at easy cannot fall below the floor: the target stays easy.',
    history: [row('q-easy-1', false, 0)],
    size: 5,
  },
  {
    name: 'promote-at-hard-ceiling',
    notes: 'Correct answers at hard cannot rise above the ceiling: the target stays hard.',
    history: [row('q-hard-1', true, 0), row('q-hard-2', true, 1)],
    size: 5,
  },
  {
    name: 'answered-excluded',
    notes: 'Answered questions never reappear, whatever the tier.',
    history: [row('q-med-1', true, 0), row('q-med-2', true, 1), row('q-hard-1', true, 2)],
    size: 10,
  },
  {
    name: 'tier-exhaustion-falls-back',
    notes:
      'Pool without any medium questions: distance from the medium target is 1 for ' +
      'both easy and hard, so both tiers interleave by rank and id rather than dead-ending.',
    questions: [
      { id: 'q-easy-1', difficulty: 'easy', domain: 'cardio', preferenceRank: 1 },
      { id: 'q-hard-1', difficulty: 'hard', domain: 'cardio', preferenceRank: 2 },
    ],
    history: [],
    size: 2,
  },
  {
    name: 'ladder-exhaustion-still-serves',
    notes:
      'Missed question on ladder L3 with no other L3 members in the pool: the exhausted ' +
      'ladder must not end the session — off-ladder questions still rank, just later.',
    questions: [
      { id: 'q-hard-2', difficulty: 'hard', domain: 'renal', ladderId: 'L3' },
      { id: 'q-easy-r1', difficulty: 'easy', domain: 'renal', ladderId: 'L4' },
      { id: 'q-easy-r2', difficulty: 'easy', domain: 'renal' },
      { id: 'q-easy-c1', difficulty: 'easy', domain: 'cardio', ladderId: 'L1' },
    ],
    history: [row('q-hard-2', false, 0)],
    size: 3,
  },
];

function buildCase(scenario: Scenario): FixtureCase {
  const questions = scenario.questions ?? POOL;
  const ranked = rankAdaptive(questions, scenario.history, scenario.size);
  return {
    name: scenario.name,
    notes: scenario.notes,
    input: {
      questions,
      history: toFixture(scenario.history),
      size: scenario.size,
    },
    expected: { order: ranked.map((q) => q.id) },
  };
}

const fixture = {
  source: 'src/lib/usmle/step1-adaptive.ts',
  generatedBy: 'python/tools/generate_step1_adaptive_fixtures.ts',
  cases: SCENARIOS.map(buildCase),
};

mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, 'step1-adaptive.cases.json');
writeFileSync(outPath, JSON.stringify(fixture, null, 2) + '\n');
console.log(`wrote ${fixture.cases.length} cases to ${outPath}`);
