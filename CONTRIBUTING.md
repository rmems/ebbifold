# Contributing to ebbifold

ebbifold is a local-first personal AI-engineering study system. Contributions
to the Python engine, tests, documentation, source registry, and original
educational content are welcome.

## Licence boundary

By contributing code, you agree that your contribution may be distributed under
the MIT License. By contributing original educational content, you agree that
it may be distributed under CC BY 4.0 with attribution to “MD3 contributors.”
Compatible third-party material keeps its own licence and attribution.

Only contribute material you have the right to share. Do not copy or closely
paraphrase commercial question banks, textbooks, lecture slides, private decks,
social-media posts, or restricted images. Public accessibility is not the same
as permission to redistribute. See `LICENSE-CONTENT.md` for the complete
boundary.

## Python engineering changes

Use Python 3.12+, `uv`, type hints, and pytest. Keep scheduler behavior changes
separate from refactors and prove parity against the committed golden fixtures
before changing behavior intentionally.

Before opening a pull request, run:

```bash
cd python
uv sync --group dev
uv run pytest -q
```

The command-line review loop and FastAPI surface are part of the engine contract.
For API changes, also exercise a scripted HTTP session and verify that attempts
persist across a SQLite restart.

Do not reintroduce Node.js, TypeScript, Prisma, or the retired upstream web
application into the active Python path. Historical upstream source remains in
git history; see `docs/typescript-cutover.md` before changing fixture provenance
or regeneration documentation.

## Open Step 1 validation corpus

Repo-native validation questions live under `open-content/usmle/step1/`. Each
question must use original wording and include a versioned `publicUsmle`
provenance envelope. Every released item requires an exact source and passage
pointer in the checked-in source registry and an approved open evidence licence.
Item text, evidence quotations, and media are separate works with separate rights.

Adding, removing, or changing an item also requires an intentional update to
`release-v1.json` and, when baseline membership changes, `baseline-v1.json`.
Changing a registered quote requires rechecking its canonical source and rights
page, updating `verifiedAt`, and recomputing the registry's quote-set SHA-256.
Any source metadata change also requires recomputing the full registry SHA-256.

The Python corpus tests are the active validation gate in this repository.
They check counts, provenance shape, answer-key integrity, and fail-closed
admission behavior. Do not weaken those tests to admit an invalid record.

## Content standard

- Write one-best-answer questions with a clear learning objective and plausible
  distractors.
- Cite the exact official or openly licensed passage that entails the keyed
  claim; a related page is not enough.
- Keep explanations within what the cited evidence supports.
- Prefer text-only items until every media asset has typed rights and
  attribution.
- Do not claim exam-score or pass prediction without prospective calibration.
- Treat all material as education, not medical advice.

Never include credentials, identifiable learner data, patient-linked material,
or real local review databases in a commit. Runtime learner memory belongs in
the user data directory and stays outside version control.
