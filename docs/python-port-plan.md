# ebbifold — Python Port Plan

Fork: `rmems/ebbifold` (from `todd866/cohort`, main-branch-only copy, Nov 26, 2026)
Direction decided Nov 26, 2026: **local app first, site later** (shipoftheseus.dev reserved for a future demo).
License: MIT (code) / CC BY 4.0 (content) — attribution to upstream stays in every phase.

## Principles

1. **Parity before features.** Every ported module ships with tests proving the Python output matches the TypeScript original on identical inputs. No new scheduling behavior until parity is proven.
2. **Corpus is frozen.** The 1,357 JSON files under `open-content/` are the data contract. Python reads them as-is; no format conversion, no re-authoring.
3. **Local-first stack.** Python package + CLI + FastAPI on localhost. SQLite locally instead of Postgres/pgvector; embeddings as NumPy arrays. Production web stack (Next.js, NextAuth, Prisma) is out of scope for the port.
4. **Julia earns its place.** Julia enters only for the math layer (Phase 5), over the established ZMQ boundary — not subprocess/FFI. Rust only if Python profiling shows a measured bottleneck.

## Verified starting facts (inspected Nov 26, 2026)

- `src/lib/usmle/step1-adaptive.ts` — **141 lines**, pure, dependency-free. The public reviewer's actual scheduler.
- `src/lib/knowledge/unified-scheduler.ts` — **290 lines**. The concept-space scheduler (runs upstream's private md3.info; not wired to the public reviewer).
- `src/lib/exam-target/` — the heavy layer: `operations.server.ts` 2,110 lines, `mastery-evidence.server.ts` 1,333, `decision-set-persistence.server.ts` 1,400, plus policy/mastery/allocator modules. Port the *concepts*, not this server stack.
- Corpus: 556 Step 1 questions + 25 baseline items (`open-content/usmle/step1/`), 392 clinical-module questions, 2,136 cloze cards; 1,357 JSON files total under `open-content/`.
- Upstream tests exist (Vitest, `src/lib/usmle/*.test.ts`, `src/lib/question-bank/*.test.ts`) — these become the source of golden fixtures.

## Layout

New code lives alongside the TS app, not replacing it:

```
python/
  pyproject.toml
  src/ebbifold/
    adaptive.py        # Phase 1
    corpus.py          # Phase 2
    scheduler.py       # Phase 3
    manifold.py        # Phase 3
    api.py             # Phase 4 (FastAPI)
    cli.py             # Phase 4
  tests/
    fixtures/          # golden vectors exported from the TS originals
    test_adaptive_parity.py
    test_corpus.py
    test_scheduler_parity.py
julia/                 # Phase 5
  Project.toml
  src/decay.jl         # forgetting-curve fitting behind ZMQ
```

Tooling: `uv` + `pytest`. Python ≥3.12, type hints throughout, pydantic models for corpus records.

## Phases

### Phase 0 — Scaffold + parity harness

- `python/` package skeleton, pyproject, pytest running green on an empty suite.
- Fixture pipeline: a small TS script (run once, output committed) that feeds fixed inputs through `step1-adaptive.ts` and dumps input/output pairs as JSON. Python tests consume the same fixtures.
- **Gate:** harness runs; fixtures committed; CI (GitHub Actions) green.

### Phase 1 — Adaptive scheduler (first real port)

- Port `step1-adaptive.ts` (141 lines) → `adaptive.py`: 3 tiers, 2 correct at a tier moves up, 1 miss drops a tier immediately, follow-ups drawn from the missed concept's teaching ladder before widening.
- **Gate:** 100% parity on golden fixtures, including edge cases (tier ceiling/floor, ladder exhaustion).

### Phase 2 — Corpus layer

- Pydantic models matching `step1-contract.ts`; loaders for the Step 1 bank, baseline set, and module questions, reading `open-content/` in place.
- Corpus audit equivalent: counts, citation presence, answer-key integrity — mirroring upstream's `public-corpus-audit` checks.
- **Gate:** loader reads all 1,357 JSON files with zero schema failures; counts match upstream's published numbers (556 / 25 / 392 / 2,136).

### Phase 3 — Concept-space scheduler + manifold

- Port `unified-scheduler.ts` (290 lines) → `scheduler.py`.
- `manifold.py`: 3,072-dim embedding vectors, cosine neighbourhood lookup in NumPy (no pgvector locally), concept graph traversal for question selection.
- Embedding generation stays an external call (Gemini, as upstream) behind an interface; cached to disk so local runs are offline after first embed.
- **Gate:** scheduler parity on fixtures; neighbourhood results match a reference cosine implementation to float tolerance.

### Phase 4 — Local app (v1 finish line)

- CLI: start a session, answer, get next question — full review loop against the USMLE bank using Phase 1 + 3.
- FastAPI on localhost exposing the same loop (sessions, answer submission, next-item), SQLite persistence for attempts.
- **Gate:** end-to-end scripted session completes; attempt history persists across restarts; this is the "local app" milestone — v1 done.

### Phase 5 — Julia math layer

- ZMQ boundary (the established Rust↔Julia pattern, applied Python↔Julia): Python sends attempt histories, Julia returns fitted parameters.
- `decay.jl`: forgetting-curve fitting — retention `R = e^(−t/S)`, per-concept memory-strength estimation, half-life regression; scheduler interval optimization.
- Benchmark Julia vs the NumPy implementation on identical histories; Julia stays only where it measurably wins or expresses the math materially better.
- **Gate:** fitted parameters agree with the Python reference within tolerance on fixture histories; benchmark numbers recorded in the repo.

### Phase 6 — Dataset #2:  AI Engineering blueprint

- Author a WGU blueprint in the `exam-target/` *shape* (registry + policy concepts only — not the 5k-line server stack): courses/competencies as concepts, questions authored or generated against them.
- Same engine, second dataset — the degree tie-in made concrete.
- **Gate:** a blueprint session runs end-to-end through the Phase 4 app, unmodified engine code.

## Explicitly out of scope

- Next.js UI, NextAuth, Prisma/Postgres production deployment, video/clip modules, payments/email (Resend), md3.info private features.
- Any public site — deferred by decision; shipoftheseus.dev is reserved for when a demo is worth serving.
- Rust — revisited only with profiling evidence from Phase 3/5.

## Degree mapping (AI Engineering)

- Phases 1–3: ML systems fundamentals — embeddings, similarity search, scheduling algorithms, test-verified reimplementation.
- Phase 5: the mathematical core — exponential decay models, parameter fitting, optimization — in Julia, the degree's math-engineering showcase.
- Phase 6: applied AI engineering — one engine, two domains, blueprint-driven targeting.