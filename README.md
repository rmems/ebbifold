# ebbifold

Spaced repetition over an embedding of concepts — not a deck of facts.

The name folds three things together: **Ebbinghaus** (the forgetting curve),
**e-fold** (the `e` in exponential decay), and **fold** (this is a fold of
upstream's code into a new stack).

## What this is

ebbifold is a Python port of [todd866/cohort](https://github.com/todd866/cohort),
the open-source spaced-repetition engine. The TypeScript app remains in this
repository as the reference implementation; new development happens under
`python/` and `julia/`:

- **Python** (`python/`, `src/ebbifold/`) — the engine: adaptive scheduling,
  the concept-space scheduler over 3,072-dimensional embeddings, corpus
  loaders, a CLI and a local FastAPI app with SQLite persistence.
- **Julia** (`julia/`) — the math layer: forgetting-curve fitting
  (`R = e^(−t/S)`), per-concept memory-strength estimation and half-life
  regression, reached over a ZMQ boundary.

The port follows one rule above all: **parity before features** — every
ported module ships tests proving it matches the TypeScript original on
identical inputs before new behavior is added. The full plan, including phase
gates, is at [docs/python-port-plan.md](docs/python-port-plan.md).

## Datasets

| | |
|---|---|
| **Dataset #1 — USMLE Step 1** | 556 original, citation-backed questions plus a 25-item baseline set, under `open-content/usmle/step1/` |
| **Dataset #2 — AI Engineering blueprint** | Planned. A second exam blueprint exercising the same engine, unmodified |

The corpus is frozen: 1,357 JSON files under `open-content/` are the data
contract. Python reads them as-is — no format conversion, no re-authoring.
Every record carries its provenance envelope, and admission is fail-closed:
nothing unverified enters the bank.

## Upstream

This repository is a fork of [todd866/cohort](https://github.com/todd866/cohort)
(main-branch-only copy). Software is MIT (`LICENSE`); original educational
content is CC BY 4.0 (`LICENSE-CONTENT.md`). Attribution to upstream is kept
throughout the port. Upstream's live deployment is
[cohort.md](https://cohort.md); this fork does not serve it.

## The inherited app

The TypeScript application (Next.js, PostgreSQL/pgvector) still lives in this
tree and remains the parity reference. Its key pieces:

- `src/lib/usmle/step1-adaptive.ts` — the public reviewer's tiered scheduler
  (3 tiers; 2 correct moves up, 1 miss drops immediately; follow-ups come from
  the missed concept's teaching ladder before widening).
- `src/lib/knowledge/unified-scheduler.ts` — the concept-space scheduler over
  the embedding manifold.
- `src/lib/exam-target/` — blueprint targeting; ships with an empty registry.
- `open-content/` — the frozen corpus described above.

Upstream's own README-era details (quick start, seeding, embedding pipeline)
still apply to the TypeScript app; see
[CONTRIBUTING.md](CONTRIBUTING.md) for the rights boundary and provenance
rules every contribution must satisfy.

## Licence and provenance

Software: MIT (`LICENSE`). Original educational content: CC BY 4.0
(`LICENSE-CONTENT.md`). Third-party material keeps its own terms, recorded
beside it; citing a page does not relicense it.

## Trademark

USMLE® is a registered trademark of the Federation of State Medical Boards and
the National Board of Medical Examiners. ebbifold is independent and is not
affiliated with or endorsed by the USMLE program. See
[About USMLE](https://www.usmle.org/about-usmle) and the
[exam security guidance](https://www.usmle.org/what-to-know/exam-security-fairness).
This corpus uses original questions, not recalled exam items.
