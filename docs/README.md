# Working on the documentation

This folder is the Sphinx source for <https://nfa-nrcwe.github.io/aerosoltools/>.
This file is for maintainers and is not part of the built site.

## Build and review locally

Install the docs dependencies once (they are an extra on the package itself, so
there is no separate requirements file):

```bash
pip install -e ".[docs]"
```

You also need the `pandoc` binary on PATH — nbsphinx uses it to convert the
markdown cells in the example notebooks. It ships with Anaconda; otherwise see
<https://pandoc.org/installing.html>.

Build:

```bash
python -m sphinx -b html -d docs/_build/doctrees docs docs/_build/html
```

Serve the result and open <http://localhost:8000>:

```bash
python -m http.server -d docs/_build/html 8000
```

`docs/_build/` and the generated `docs/api/` tree are both gitignored — they are
build artifacts and are recreated from scratch every time.

A full build takes roughly a minute and a half, because **the example notebooks
are executed** (see below).

## How this is wired together

- **The landing page is the repository `README.md`**, pulled in by
  `docs/index.md` via a MyST `include`. Edit the README, not `index.md` — there
  is deliberately no second copy to keep in sync.
- **The API reference is generated**, by sphinx-autoapi, from the source in
  `src/`. There are no hand-written API stubs to update when you add or rename a
  module, and `docs/api/` is not checked in: stale committed stubs are how
  deleted classes stayed on the published site for months.
  The GUI is excluded (`autoapi_ignore`) — it is an application, not a library
  API.
- **Example notebooks are re-executed on every build**
  (`nbsphinx_execute = "always"`, `nbsphinx_allow_errors = False`). Their stored
  outputs are ignored and are stripped in git. This means the published examples
  can never show results from an API that no longer exists — but it also means
  **a notebook that raises will fail the build**. Notebooks read sample files
  from `tests/data/` using paths relative to `docs/examples/`, so use forward
  slashes: a Windows-style `r"..\..\tests\data\x.txt"` works locally and breaks
  on the Linux CI runner.

## Releasing the docs

1. Build and review locally (above).
2. Push your branch and open a pull request to `main`. **Docs check** builds the
   docs in CI without publishing, and attaches the built site to the run as a
   downloadable `docs-site` artifact.
3. Merge to `main`.
4. Run the **Docs publish (gh-pages, live)** workflow from the Actions tab. It
   writes to the `gh-pages` branch; GitHub's automatic `pages-build-deployment`
   job then serves it. Nothing else needs running by hand.

Note that a `workflow_dispatch` button only appears for workflow files that
exist on the **default branch**, so a new or renamed docs workflow is not
runnable until it has been merged to `main`.
