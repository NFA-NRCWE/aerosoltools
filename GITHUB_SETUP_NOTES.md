# GitHub / `gh` — Setup Notes & To-Do

> Summary of our discussion (2026-07-15) about giving Claude access to GitHub
> beyond local git, what that enables, how to keep it safe, and the workflow we
> want. Read this, then work through the **To-Do** at the bottom.
> _(This is a working note — not committed to git unless you decide to.)_

---

## TL;DR — the decisions so far

- **git ≠ gh.** I already have local-git (commit + push branches). Adding the
  `gh` CLI is a *separate* channel that reaches the GitHub **platform** (issues,
  PRs, Actions, releases…). One does not expand the other.
- **You'll add `gh` with a *fine-grained token*, scoped down**, so I get only the
  capabilities you choose — provably, not on trust.
- **You'll protect `main`** (branch protection) — mainly to guard against
  accidental direct pushes/merges by you or less-gh-experienced colleagues.
- **Division of labor:** *I draft (PRs, release notes, CI); you approve every
  merge — especially to `main`; test-gated automation does the rest.*

---

## 1. Two separate channels: `git` vs `gh`

| | Channel | What it can touch | How I get it |
|---|---|---|---|
| **git** (already granted) | git protocol | Repo *contents*: commit, push/create/force-push branches, push tags, fetch. **Cannot** create PRs, releases, issues, or trigger Actions. | Your existing local git credential |
| **gh** (to add) | GitHub REST API | The *platform*: issues, PRs, Actions, releases, settings, other repos, gists — **but only what the token permits** | A fine-grained token fed to `gh` |

**Key correction to a common assumption:** *pushing a branch is NOT the same as
opening a PR.* A pull request is a GitHub object; git has no concept of it.
Today I **cannot** open/merge/review PRs at all — that needs `gh` (and PR
permission).

**Capability vs. policy vs. enforcement (re: pushing to `main`):**
- *Capability:* the git channel technically **could** merge locally and push to
  `main` if the credential allows and `main` isn't protected.
- *Policy:* my instructions forbid it (never touch `main`, never merge, push
  only when asked) — and I've followed that all session.
- *Enforcement:* only **branch protection** makes it impossible regardless of
  intent. That's the belt-and-suspenders.

---

## 2. What I could do on GitHub via `gh` (capability → permission)

Read-only permissions are low-risk and already very useful.

### High-value cluster
| Area | What I'd do for you | Token permission |
|---|---|---|
| **Issues** | Backlog: create/label/milestone/close, comment, triage, bulk-file, search, link issues↔commits | Issues: **RW** |
| **Actions / CI** | Read a failed run's logs & diagnose, re-run failed jobs, trigger manual workflows (e.g. docs deploy), download artifacts | Actions: **Read** (RW only for re-run/trigger); Checks: Read |
| **Pull Requests** | Open `GUI_test → main` PRs with written summaries/changelogs, request reviewers, read reviews, check CI status, comment | Pull requests: **RW** (+ Contents: RW if pushing the branch via API) |
| **Releases & tags** | Draft release notes from the commit log, create the `v*` tag / GitHub Release that triggers PyPI | Contents: **RW** |
| **Labels & milestones** | Set up a taxonomy (`bug`/`feature`/`docs`/`good-first-issue`) + release milestones | Issues: RW |

### Nice, lower priority
- **Repo metadata** — edit description, add topics for discoverability _(Administration: RW)_
- **Dependabot / security alerts** — read dep/vuln alerts, help bump deps _(Dependabot: Read; Security events: Read)_
- **Projects (kanban)** — issues on a board _(Projects: RW, org-level)_
- **Discussions / Wiki** — probably not needed for a lib this size

### Grant sparingly / never
- **Administration** (branch protection, webhooks, collaborators) — powerful, sensitive. Keep **off** unless a specific task needs it.
- **Secrets & variables** (e.g. `PYPI_API_TOKEN`) — **never** grant write; values can't be read back anyway.

---

## 3. Controlling access — the fine-grained token

**Least-privilege ladder** (you don't have to grant it all at once):

1. **Start:** `Issues: RW` + `Metadata: RO` → the backlog. Tiny, safe.
2. **Add when useful:** `Actions: Read` + `Checks: Read` + `Pull requests: Read`
   → I can *diagnose CI failures and read PR/CI status* while still basically
   read-only. **Sweet spot** — most bang for least risk.
3. **Add at release time:** `Pull requests: RW` + `Contents: RW` → I can open the
   merge PR and draft/create releases. First point I can write code/tags — grant
   intentionally; branch protection is the backstop.

**Token recipe (issue-only starter):**
1. GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate.
2. Resource owner `NFA-NRCWE`; Repository access → *Only select* → `aerosoltools`.
3. Permissions: **Issues: Read and write**, **Metadata: Read-only** (mandatory), everything else **No access**.
4. Set an **expiration** (e.g. 90 days).
5. Feed it to `gh` **without** the broad browser scopes:
   ```bash
   gh auth login --with-token < token.txt
   gh issue list --repo NFA-NRCWE/aerosoltools   # verify
   ```

**Controls you keep:** the token's permission list is a hard ceiling I can't
exceed; expiry auto-revokes; you can revoke instantly; I never see the token
(`gh` stores it, I just call `gh`); every command is a visible, approvable tool
call.

> ⚠️ Org policy: fine-grained tokens may need enabling in `NFA-NRCWE` org
> settings — you're an org admin, so this is a quick toggle.

---

## 4. Protecting `main` (branch protection)

`aerosoltools` → Settings → Branches → Add rule for `main`:
- Require a pull request before merging (+ at least one approving review)
- Require status checks to pass before merging (select the test/lint checks)
- Block force pushes
- (optional) Restrict who can push

Result: no direct push/merge to `main` — from me, a script, or a colleague —
without a green PR you approved.

---

## 5. The workflow we want

**I draft → you approve → test-gated automation ships.**

- **Me:** open PRs with a "what changed & why" summary, draft release notes from
  commits, propose CI/test/workflow improvements — all reviewable.
- **You:** review + merge (especially into `main`); press "release" when ready.
- **Automation:** on your chosen events, run tests → only if green → publish +
  deploy.

**Recommendation:** *decouple "merge to main" from "publish to PyPI."* Not every
main commit is a release. Trigger the gated `test → publish → deploy` chain from
**you cutting a GitHub Release / pushing a `v*` tag**, so shipping is an explicit
act you take. Docs can still auto-refresh on merges to `main`.

---

## 6. Current CI/CD state & gaps

**Three workflows in `.github/workflows/`:**

| Workflow | Trigger | Does |
|---|---|---|
| `code_quality.yml` | push/PR to **`main`, `dev`** + manual | pytest (Py 3.10); ruff `--fix` + black (auto-commits on PRs) |
| `deploy-docs.yml` | **manual only** | Sphinx → `gh-pages` (pandoc, `requirements.txt`, Py 3.10) |
| `publish-to-pypi.yml` | push tag **`v*`** | build sdist+wheel → `twine upload` (setuptools-scm from tag; `PYPI_API_TOKEN`) |

**Gaps vs. the workflow we want:**
1. **Release isn't test-gated** — `publish-to-pypi` uploads with *no test run
   first*; a broken build could ship. → make it `test → publish → deploy`.
2. **Docs don't auto-deploy** — `deploy-docs` is manual-only. → trigger on
   release (and/or push to `main`).
3. **CI doesn't run on `GUI_test`** — only `main`/`dev`; my pushes are
   unvalidated until merge. → add the branch / rely on PRs.
4. **GUI is untested in CI** — installs `-e .` (no `[gui]`) and no
   `QT_QPA_PLATFORM=offscreen`, so every `importorskip("PyQt5")` test skips. →
   `pip install -e .[gui]` + set offscreen if you want GUI coverage.
5. **Single Python version** — tests only 3.10 though lib targets 3.10–3.12. →
   add a matrix.
6. **Minor** — docs workflow uses `requirements.txt` (confirm it carries the doc
   deps, or switch to `pip install -e .[docs]` which already exists in
   `pyproject.toml`); action-version drift (`checkout@v3` vs `v4`).

---

## ✅ To-Do (work through tomorrow)

**A. Access & safety (you do these)**
- [ ] Org settings: enable fine-grained personal access tokens for `NFA-NRCWE`.
- [ ] Mint the starter token: `aerosoltools` only, **Issues RW + Metadata RO**, with an expiry.
- [ ] `gh auth login --with-token` and verify `gh issue list` works.
- [ ] Add **branch protection on `main`** (require PR + review + status checks; block force-push).
- [ ] (Optional now / later) Extend the token with **Actions Read + Checks Read + Pull requests Read** so I can diagnose CI and read PR status.

**B. Backlog (I do these once Issues RW is live)**
- [ ] File the initial issues (drafts below), labelled.
- [ ] Set up a label taxonomy + a release milestone.

**C. CI/CD upgrades (I draft as a reviewable PR; you merge)**
- [ ] Make the release pipeline test-gated: `test → publish → deploy` on `v*`/Release.
- [ ] Auto-deploy docs on release (and/or push to `main`).
- [ ] Run Code Quality on `GUI_test` / PRs; add Python 3.10–3.12 matrix.
- [ ] Add GUI test coverage in CI (`.[gui]` + `QT_QPA_PLATFORM=offscreen`).
- [ ] Tidy docs workflow deps (`.[docs]` vs `requirements.txt`) + action versions.

**D. Release prep (the original "tomorrow" goal)**
- [ ] Regenerate Sphinx docs (the checked-in `docs/api/` tree is stale).
- [ ] Review + merge `GUI_test → main` via a PR (I'll draft the summary/changelog).
- [ ] Cut the release → PyPI publish + docs deploy (once the gated pipeline is in).

---

## Draft issue backlog (ready to `gh issue create`)

1. **dusttrak: keep-all-columns + unit resolution** — apply the Phase-3 pattern
   (default `extra_data=True`, `resolve_extra_columns`). *Blocked:* needs a real
   DustTrak export in `tests/data/` to verify. `label: loaders, blocked`
2. **CI: run Code Quality on `GUI_test` and PRs** + Python 3.10–3.12 matrix.
   `label: ci`
3. **CI: cover the GUI headlessly** (`pip install -e .[gui]`,
   `QT_QPA_PLATFORM=offscreen`). `label: ci, gui`
4. **CI: make the PyPI release test-gated** (`test → publish → deploy`).
   `label: ci, release`
5. **Docs: auto-deploy on release** (trigger `deploy-docs`; confirm doc deps).
   `label: docs, ci`
6. **Units: extend cross-instrument metric aliases** as new instruments are
   added (`_CANONICAL_METRICS` in `_core/metrics.py`). `label: enhancement`
7. **(Optional) Surface ambient T/RH for instruments that currently drop them**
   (e.g. curate more per-loader units). `label: loaders, enhancement`
