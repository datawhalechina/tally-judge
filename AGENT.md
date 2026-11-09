# Tally Judge contributor guide for coding assistants

## Repository and development environment

The Ragas library lives in `src/ragas/`. It contains evaluation metrics,
test-set generation, datasets, experiments, storage backends and integrations.
Tests live in `tests/`. The installable `ragas-examples` workspace package lives
in `examples/`. Each package declares its dependencies in its own `pyproject.toml`.
Run the following commands from the repository root.

| Task | Command |
| --- | --- |
| Install minimal development dependencies | `make install-minimal` |
| Install the full ML development environment | `make install` |
| Install or build the examples package | `uv sync --package ragas-examples` or `uv build --package ragas-examples` |
| Format and lint | `make format` |
| Type check | `make type` |
| Run formatting and type checks | `make check` |
| Run unit or end-to-end tests | `make test` or `make test-e2e` |
| Run a specific test | `uv run pytest tests/unit -k "test_name"` |
| Run local CI | `make run-ci` |
| Run benchmarks locally or in Docker | `make benchmarks` or `make benchmarks-docker` |
| Build or preview documentation | `make build-docs` or `make serve-docs` |

Use `uv run` for Python and related tools such as pytest, Ruff and Pyright.
The minimal installation uses the `dev-minimal` extra. The full installation
uses the `dev` dependency group. For a new worktree that needs the same local
credentials, copy the root worktree's local `.env` to the new worktree, then
run `make install-minimal` and `make check`.

Store scratch notes in `_experiments/` and requested plans in `plan/`.
For debugging, enable the Python logger for the relevant module, such as
`logging.getLogger("ragas._analytics")`, and attach a DEBUG-level stream handler.

## Issues and pull requests

Read a reported issue with `gh issue view`, trace the affected code and verify
the fix with focused tests, linting and type checking. For a requested pull
request, work on a feature branch, run `make format` and commit the changes.
Check the working tree and upstream branch. Review `git log main..HEAD` and
`git diff main...HEAD` so the title, summary and test plan cover the whole PR.
Use `.github/pull_request_template.md` and review the draft with the requester
before publishing. When authorized, push the branch, create the PR with
`gh pr create` and return its URL.

## Documentation

Give each page one purpose and place it in the matching MkDocs section.

| Page type | Location | What to include |
| --- | --- | --- |
| Tutorial | `docs/getstarted/` or `docs/tutorials/` | A guided lesson with a complete working example |
| How-to | `docs/howtos/` | Steps for one user goal, with outputs and links to relevant reference pages |
| Reference | `docs/references/` | Public API signatures, parameters, return values and usage examples |
| Explanation | `docs/concepts/` | Concepts, design decisions and rationale |

Check `mkdocs.yml` for the current navigation and add new pages to the relevant
section. Open with the purpose and prerequisites, including required services
and configuration. Use second person, active voice, relative links and
copyable code with imports. Put verbose outputs or prompts in collapsible
`??? "Click to expand"` blocks. Leave a blank line before lists and keep
numbered steps together. Cross-link pages when readers need a different level
of detail.

Use `docs/ipynb_to_md.py` to convert notebooks when maintaining generated
Markdown. Put images in `docs/_static/`. Keep custom styling in `docs/extra/`.
Check public exports before adding mkdocstrings links, for example
`[BasePrompt][ragas.prompt.BasePrompt]`. When working on the RAG application
how-to, update `docs/howtos/applications/evaluate-and-improve-rag.md` after
each verified step.

### Migrating a how-to guide

1. Read the guide and identify the behavior its example demonstrates.
   Check support in `src/ragas/experiment.py`, `src/ragas/evaluation.py` and
   `src/ragas/metrics/collections/`. If the feature requires legacy
   `evaluate()`, keep the guide and report its path and the missing feature.
   If metric decorators already cover the use case, recommend retiring the
   obsolete example. Review the proposed migration with the requester.
2. When converting a notebook-derived guide to standalone Markdown, replace
   the notebook and generated underscored page with one maintained `.md`
   page. Update navigation and links. Keep the example focused on its original
   goal. Use metrics from `ragas.metrics.collections`, `llm_factory` from
   `ragas.llms` and `embedding_factory` from `ragas.embeddings.base` where
   appropriate.
3. Verify public signatures against source and official library documentation.
   Exercise all Python examples in `tests/docs/test_<guide_name>.py` with
   `uv run python`. Use locally configured credentials for examples that need
   a provider. Keep the verification script and report which examples ran,
   any missing prerequisites and any remaining gaps.
