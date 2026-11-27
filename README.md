# ebbifold

Spaced repetition over an embedding of concepts — not a deck of facts.

The name folds three things together: **Ebbinghaus** (the forgetting curve),
**e-fold** (the `e` in exponential decay), and **fold** (this is a fold of
upstream's code into a new stack).

## What this is

ebbifold is a local-first personal AI-engineering study system. It began as a
Python port of [todd866/cohort](https://github.com/todd866/cohort); the Python
package is now the implementation of record and the inherited TypeScript
application has been retired after golden-fixture parity was captured.

- **Python** (`python/src/ebbifold/`) — adaptive scheduling, the concept-space
  scheduler over embedding vectors, corpus loaders, a CLI, and a local FastAPI
  app with SQLite persistence.
- **Julia** (`julia/`, planned) — the math layer for forgetting-curve fitting
  (`R = e^(−t/S)`), per-concept memory-strength estimation, and half-life
  regression behind a ZMQ boundary.

The port follows one rule above all: **parity before features**. Golden vectors
exported from the upstream TypeScript implementation remain committed under
`python/tests/fixtures/`; Python tests consume those artifacts directly and
never execute TypeScript or Node.

## Local review app

```bash
cd python
uv sync --group dev
uv run pytest
uv run ebbifold-review --help
uv run ebbifold-review-api --help
```

The API binds to `127.0.0.1` by default. Learner memory is local runtime data:
SQLite databases default to the user data directory and are gitignored.

## Datasets

|                                           |                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Dataset #1 — USMLE Step 1**             | 556 original, citation-backed questions plus a 25-item baseline set, under `open-content/usmle/step1/` |
| **Dataset #2 — AI Engineering blueprint** | Planned. A second exam blueprint exercising the same engine, unmodified                                |

The corpus is frozen: the JSON files under `open-content/` are the data
contract. Python reads them as-is — no format conversion, no re-authoring.
Every record carries its provenance envelope, and admission is fail-closed:
nothing unverified enters the bank.

## TypeScript cutover

The upstream Next.js/PostgreSQL application, Node tooling, Prisma schema, and
TypeScript-only CI were removed in Phase 4c. The cutover inventory and fixture
provenance rules are in [docs/typescript-cutover.md](docs/typescript-cutover.md);
the phase plan is in [docs/python-port-plan.md](docs/python-port-plan.md).

## Upstream

This repository is a fork of [todd866/cohort](https://github.com/todd866/cohort)
(main-branch-only copy). Software is MIT (`LICENSE`); original educational
content is CC BY 4.0 (`LICENSE-CONTENT.md`). Attribution to upstream is kept
throughout the port. Upstream's live deployment is
[cohort.md](https://cohort.md); this fork does not serve it.

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
