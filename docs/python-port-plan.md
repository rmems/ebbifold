# ebbifold — Python Port Plan

Fork: `rmems/ebbifold` (from `todd866/cohort`, main-branch-only copy, Nov 26, 2026)
Direction decided Nov 26, 2026: **local app first, site later** (shipoftheseus.dev reserved for a future demo).
License: MIT (code) / CC BY 4.0 (content) — attribution to upstream stays in every phase.

## Principles

1. **Parity before features.** Every ported module ships with tests proving the Python output matches the historical TypeScript implementation on identical inputs. No new scheduling behavior until parity is proven.
2. **Corpus is frozen.** The JSON files under `open-content/` are the data contract. Python reads them as-is; no format conversion, no re-authoring.
3. **Local-first stack.** Python package + CLI + FastAPI on localhost. SQLite locally instead of Postgres/pgvector; embeddings as NumPy arrays. Production web stack (Next.js, NextAuth, Prisma) is retired, not part of the port.
4. **Julia earns its place.** Julia enters only for the math layer (Phase 5), over the established ZMQ boundary — not subprocess/FFI. Rust only if Python profiling shows a measured bottleneck.
5. **Python is the implementation of record.** Golden fixtures are committed, self-contained artifacts. Tests, CLI, API, and CI must not execute Node or TypeScript.

## Historical starting facts (inspected Nov 26, 2026)

- `src/lib/usmle/step1-adaptive.ts` — 141 lines, pure, dependency-free. The public reviewer's actual scheduler.
- `src/lib/knowledge/unified-scheduler.ts` — 290 lines. The concept-space scheduler.
- `src/lib/exam-target/` — the heavy server layer. The port keeps the policy concepts, not the server stack.
- Corpus: 556 Step 1 questions + 25 baseline items (`open-content/usmle/step1/`), 392 clinical-module questions, 2,136 cloze cards.
- Upstream Vitest tests were the source of golden fixtures. Those TypeScript sources now exist only in git history.

## Layout

```
python/
  pyproject.toml
  src/ebbifold/
    adaptive.py          # Phase 1
    corpus/              # Phase 2
    scheduler/           # Phase 3a
    manifold/            # Phase 3b
    review/              # Phase 4 CLI + API + SQLite boundary
  tests/
    fixtures/            # committed golden vectors exported from the TS originals
open-content/            # frozen validation corpus
schemas/                 # versioned corpus/attempt contracts
julia/                   # Phase 5, planned
```

Tooling: `uv` + `pytest`. Python ≥3.12, type hints throughout, pydantic models for corpus records.

## Phases

### Phase 0 — Scaffold + parity harness

- `python/` package skeleton, pyproject, pytest running green on an empty suite.
- Fixture pipeline: fixed inputs were run once through the TypeScript originals; input/output pairs are committed as JSON. Python tests consume the artifacts directly.
- **Gate:** harness runs; fixtures committed; CI green without Node execution.

### Phase 1 — Adaptive scheduler (first real port)

- Port `step1-adaptive.ts` → `adaptive.py`: 3 tiers, 2 correct at a tier moves up, 1 miss drops a tier immediately, follow-ups drawn from the missed concept's teaching ladder before widening.
- **Gate:** 100% parity on golden fixtures, including edge cases (tier ceiling/floor, ladder exhaustion).

### Phase 2 — Corpus layer

- Pydantic models matching the corpus contract; loaders for the Step 1 bank, baseline set, module questions, and cloze cards, reading `open-content/` in place.
- Corpus audit equivalent: counts, citation presence, answer-key integrity.
- **Gate:** loader reads the frozen corpus with zero schema failures; counts match the published numbers.

### Phase 3 — Concept-space scheduler + manifold

- Port `unified-scheduler.ts` → `scheduler/`.
- `manifold/`: embedding vectors and cosine neighbourhood lookup in NumPy (no pgvector locally), concept graph traversal for question selection.
- Embedding generation is provider-agnostic and outside the scheduler core; vectors arrive as inputs and cached artifacts.
- **Gate:** scheduler parity on fixtures; neighbourhood results match a reference cosine implementation to float tolerance.

### Phase 4 — Local app (v1 finish line)

- CLI: start a session, answer, get next question — full domain-neutral review loop using Phase 1 + 3.
- FastAPI on localhost exposing the same loop (sessions, answer submission, next-item), SQLite persistence for attempts.
- Phase 4c retires the inherited TypeScript implementation after the Python CLI/API gate passes.
- **Gate:** end-to-end scripted session completes; attempt history persists across restarts; CI, tests, CLI, and API require no Node or TypeScript execution.

### Phase 5 — Julia math layer

- ZMQ boundary (the established Rust↔Julia pattern, applied Python↔Julia): Python sends attempt histories, Julia returns fitted parameters.
- `decay.jl`: forgetting-curve fitting — retention `R = e^(−t/S)`, per-concept memory-strength estimation, half-life regression; scheduler interval optimization.
- Benchmark Julia vs the NumPy implementation on identical histories; Julia stays only where it measurably wins or expresses the math materially better.
- **Gate:** fitted parameters agree with the Python reference within tolerance on fixture histories; benchmark numbers recorded in the repo.

### Phase 6 — Dataset #2: AI Engineering blueprint

- Author an AI-engineering blueprint in the exam-target *shape* (registry + policy concepts only — not the retired server stack): courses/competencies as concepts, questions authored or generated against them.
- Same engine, second dataset — the degree tie-in made concrete.
- **Gate:** a blueprint session runs end-to-end through the Phase 4 app, unmodified engine code.

## Explicitly out of scope

- Next.js UI, NextAuth, Prisma/Postgres production deployment, video/clip modules, payments/email, md3.info private features.
- Any public site — deferred by decision; shipoftheseus.dev is reserved for when a demo is worth serving.
- Rust — revisited only with profiling evidence from Phase 3/5.
- Reintroducing Node or TypeScript into the active engine, test, CLI, API, or CI path.

## Degree mapping (AI Engineering)

- Phases 1–3: ML systems fundamentals — embeddings, similarity search, scheduling algorithms, test-verified reimplementation.
- Phase 5: the mathematical core — exponential decay models, parameter fitting, optimization — in Julia, the degree's math-engineering showcase.
- Phase 6: applied AI engineering — one engine, two domains, blueprint-driven targeting.
