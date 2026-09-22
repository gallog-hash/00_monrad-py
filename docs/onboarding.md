# Setting up a development environment

This guide takes you from a bare machine to a working `monrad` checkout with
the test suite passing. It should take about ten minutes, most of which is
waiting for the test run.

You do **not** need to install Python yourself, and you should not use `pip`,
`conda`, `virtualenv`, or `python -m venv` anywhere in this project. `uv`
handles all of it.

## 1. Install uv

`uv` is the package and project manager this repository uses. It installs
Python interpreters, resolves dependencies, creates the virtual environment,
and runs commands inside it.

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your shell, then confirm it is on your `PATH`:

```bash
uv --version
```

Any reasonably recent version works; the project is developed against uv 0.9.

## 2. Clone the repository

```bash
git clone https://github.com/gallog-hash/00_monrad-py.git
cd 00_monrad-py
```

## 3. Python version

You don't have to do anything here — this step is just so you know what
happens.

`.python-version` pins the project to **Python 3.12**. `pyproject.toml`
declares `requires-python = ">=3.10"`, so the code is expected to run on 3.10
and later, but 3.12 is what the environment is built with and what you should
develop against.

If you don't have 3.12, uv downloads it for you in the next step. It installs
into uv's own directory and does not touch your system Python.

## 4. Create the environment

```bash
uv sync --frozen
```

That one command:

- creates `.venv/` in the project root,
- installs Python 3.12 if it isn't already available,
- installs every dependency at the exact version recorded in `uv.lock`,
- installs `monrad` itself in editable mode, so your edits under `src/` take
  effect immediately with no reinstall.

`--frozen` means "use the lockfile exactly, don't try to re-resolve". This is
what CI runs, so it guarantees your environment matches the one your PRs are
tested in. Use it every time unless you are deliberately changing dependencies
(see §9).

### What gets installed

| Group | Packages | Why |
|---|---|---|
| Runtime | `numpy>=1.24` | The entire pipeline. That's the only hard runtime dependency. |
| `dev` (installed by default) | `pytest>=8` | Test suite |
| | `scipy>=1.11` | Optimisation and statistics in the pose fit |
| | `matplotlib>=3.7` | Diagnostic plots from the monitoring drivers |
| | `plotly>=5.18` | Interactive plots |
| | `ruff>=0.15.14` | Linter and formatter |

The `dev` group is a
[dependency group](https://packaging.python.org/en/latest/specifications/dependency-groups/),
and uv installs it by default — `uv sync --frozen` is all you need, there is no
extra flag.

## 5. Verify it worked

```bash
uv run pytest
```

You should see **491 tests pass**. The suite is slow — well over fifteen
minutes on a typical laptop — because several tests generate synthetic
detector streams and push them through all five pipeline stages end to end.
Start it and go do something else; it is not hung.

To iterate faster while working, run one file or one test:

```bash
uv run pytest tests/test_stage3.py
uv run pytest tests/test_stage5.py::TestPoseParameterRecovery
uv run pytest -x            # stop at the first failure
```

If all 491 pass, your environment is correct and you're done with setup.

## 6. The `uv run` habit

Prefix project commands with `uv run`:

```bash
uv run pytest
uv run python scripts/run_pipeline.py --help
uv run monrad-decode-header <file>
```

`uv run` executes inside the project venv and re-syncs it first if `uv.lock`
changed — which is what you want after every `git pull`. You never need to
activate the venv, and activating it by hand is the usual way people end up
debugging an environment that silently drifted from the lockfile.

The package installs seven console scripts, all available under `uv run`:

| Command | Purpose |
|---|---|
| `monrad-decode-header` | Inspect a header file (run-aware) |
| `monrad-decode-gps` | Inspect a `*_GPS.bin` timing file |
| `monrad-decode-bin` | Inspect a `*.bin` position file |
| `monrad-align` | Telescope alignment calibration and drift monitor |
| `monrad-monitor` | Single-probe pose monitoring |
| `monrad-multiprobe` | Multi-probe pose monitoring |
| `monrad-resolution` | Probe-resolution study (fully synthetic) |

One wrinkle worth knowing before it confuses you: the three `monrad-decode-*`
scripts read their argument as a bare filename rather than going through
`argparse`, so `monrad-decode-header --help` fails with a
`FileNotFoundError: '--help'` instead of printing usage. Pass a real file.
The other four commands support `--help` normally.

## 7. Linting and formatting with Ruff

Ruff is both the linter and the formatter. There is no separate Black or Flake8
here, and you should not add one.

```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint and auto-fix what it can
uv run ruff format .         # format
```

CI runs `ruff check .` and `ruff format --check .`, and **both must pass before
a pull request can merge**. `--check` does not reformat, it just fails if
anything is unformatted. So run `uv run ruff format .` before you push, or let
the pre-commit hook do it for you (§8).

The project currently uses Ruff's default rule set — there is no `[tool.ruff]`
section in `pyproject.toml` and no `ruff.toml`. Don't add one without asking.

## 8. Pre-commit hooks (recommended)

`.pre-commit-config.yaml` runs `ruff check --fix` and `ruff format` on every
commit, which keeps you from ever failing CI on formatting.

Note that `pre-commit` is **not** part of the `dev` dependency group, so
`uv sync` does not install it. Install it as a standalone tool:

```bash
uv tool install pre-commit
pre-commit install
```

That second command writes the git hook into your local `.git/hooks/`. It is
per-clone, so you do it once here and again in any future clone.

This step is optional — CI catches the same problems — but it catches them in
one second instead of after a round trip through GitHub.

## 9. Changing dependencies

If you genuinely need a new package, don't `pip install` it. Add it properly so
the lockfile and CI follow:

```bash
uv add somepackage              # runtime dependency
uv add --group dev somepackage  # dev-only (test/plot/tooling)
```

Both update `pyproject.toml` and `uv.lock`. **Commit the updated `uv.lock`** —
CI runs `--frozen` and will fail if the lockfile doesn't match
`pyproject.toml`.

Adding a runtime dependency is a design decision, not a routine one: the
pipeline deliberately depends on NumPy alone. Ask before adding to the runtime
list.

## 10. About the data

`data/` is in `.gitignore` and is **not** in your clone. On the original
machine it holds several gigabytes of raw detector files, which don't belong in
git.

This does not block you. Every test uses synthetic data generated by
`monrad.synthetic.generate()`, so the full suite — including the end-to-end
pipeline test — runs on a fresh clone with no real files at all. Same for
`monrad-resolution`, which is fully synthetic.

What you *can't* do without real data is run the pipeline or the monitoring
drivers on an actual acquisition (`run_pipeline.py`, `monrad-monitor`,
`monrad-multiprobe`, `monrad-align`, and the `macros/*.args` files, which
contain machine-specific absolute paths). Ask your supervisor for a dataset and
where to put it when you reach that point.

## 11. How to contribute changes

`main` is protected. You cannot push to it directly, and neither can anyone
else — this is deliberate, not a permissions problem to work around.

```bash
git switch -c feat/short-description
# ... work, commit ...
git push -u origin feat/short-description
gh pr create        # or open the PR from the GitHub web UI
```

For the PR to merge, CI must be green: `ruff check`, `ruff format --check`, and
the full test suite.

One setup detail that will save you confusion: **link the email address you
commit with to your GitHub account** (Settings → Emails). If your commits
aren't attributed to a GitHub user, the branch rules will hold your PR for an
extra approval, which looks like an unexplained block. Check what you're
committing as with:

```bash
git config user.email
```

## 12. What to read next

| File | What it is |
|---|---|
| `README.md` | What the pipeline does, file formats, worked usage examples |
| `DESIGN.md` | The authoritative algorithm reference. Start at §11 — the synthetic end-to-end test — then read the stage you're working on |
| `CLAUDE.md` | Project conventions and key invariants, worth reading even though it's addressed to an AI assistant |
| `docs/handoffs/` | Dated notes on past investigations; useful history, not current instructions |

If `DESIGN.md` and the code disagree, **the code wins** — and that's worth
raising, because it usually means the document needs an update.
