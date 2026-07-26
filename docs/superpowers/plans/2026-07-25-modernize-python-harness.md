# Modernize Python/LangChain Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace this course repo's ad-hoc tooling (black+isort, no tests, no CI, no type-checking) with a modern, low-ceremony `uv`-based harness: `ruff` for format/lint, `pyright` for types, `pytest` with fake-model tests, a flat `examples/` layout for course exercises, LangSmith tracing wired up via env vars, `pre-commit`, and GitHub Actions CI.

**Architecture:** Each task is an independent, mechanical tooling addition on top of the existing `uv`-managed project. Task 2 (restructure to `examples/`) refactors `main.py` into a testable `summarize_person(llm, information)` function so Task 3 can inject a fake chat model in tests instead of hitting a real LLM. All later tasks (pyright, pre-commit, CI) just point at whatever `examples/`/`tests/` contain by then.

**Tech Stack:** Python 3.12, `uv`, `ruff`, `pyright`, `pytest`, `langchain-core` fake chat models, `pre-commit`, GitHub Actions.

## Global Constraints

- Always add/remove packages via `uv add` / `uv remove` (never hand-edit the `dependencies`/`dependency-groups` arrays in `pyproject.toml`) — per existing CLAUDE.md convention.
- Python `>=3.12` throughout (already pinned in `.python-version` and `pyproject.toml`).
- Keep `main.py`'s current runtime behavior (Ollama `gemma3:270m` chain that prints a summary) working end-to-end after the move to `examples/`.
- Repo has a real GitHub remote (`github.com/fernandohermida/langchain-course`), so the CI task should produce a workflow that actually runs on push/PR.

---

### Task 1: Replace black+isort with ruff (dev dependency group)

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md:12-13` (Commands section)

**Interfaces:** None (tooling-only, no runtime code affected).

- [ ] **Step 1: Remove the old formatters**

Run: `uv remove black isort`

Expected: `pyproject.toml`'s `dependencies` list no longer contains `black` or `isort`; `uv.lock` is updated.

- [ ] **Step 2: Add ruff as a dev dependency**

Run: `uv add --dev ruff`

Expected: `pyproject.toml` gains a `[dependency-groups]` table with `dev = ["ruff>=..."]`.

- [ ] **Step 3: Add ruff config to `pyproject.toml`**

Append this block to `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 4: Format and lint the existing code**

Run: `uv run ruff format .`
Run: `uv run ruff check --fix .`
Expected: both exit 0; `main.py` gets reformatted (e.g. the stray trailing whitespace and misaligned comment on the `ChatOllama`/`ChatOpenAI` lines get cleaned up).

- [ ] **Step 5: Update CLAUDE.md commands**

In `CLAUDE.md`, replace:

```markdown
- Format code: `uv run black .`
- Sort imports: `uv run isort .`
```

with:

```markdown
- Format code: `uv run ruff format .`
- Lint (and autofix): `uv run ruff check --fix .`
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md main.py
git commit -m "chore: replace black+isort with ruff"
```

---

### Task 2: Restructure into a flat `examples/` layout

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/ch01_hello_chain.py`
- Delete: `main.py`
- Modify: `CLAUDE.md:11` (Commands section — run command)

**Interfaces:**
- Produces: `summarize_person(llm: BaseChatModel, information: str) -> str` in `examples/ch01_hello_chain.py`, and `INFORMATION: str` (the sample bio text) — Task 3's tests import both.

- [ ] **Step 1: Create the examples package marker**

Create `examples/__init__.py` (empty file).

- [ ] **Step 2: Create `examples/ch01_hello_chain.py`**

```python
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

INFORMATION = """Lionel Andrés Messi Cuccittini (Rosario, 24 de junio de 1987), conocido como Leo Messi, es un futbolista argentino que juega como delantero o centrocampista. Desde 2023, integra el plantel del Inter Miami de la MLS canadoestadounidense. Es también internacional con la selección de Argentina, de la que es capitán.

Considerado con frecuencia el mejor jugador del mundo y uno de los mejores de todos los tiempos,[11] es el único en la historia que ha ganado, entre otras distinciones, ocho veces el Balón de Oro, ocho premios de la FIFA al mejor jugador del mundo, seis Botas de Oro y dos Balones de Oro de la Copa Mundial de Fútbol. En 2020, se convirtió en el primer futbolista y el primer argentino en recibir un premio Laureus y fue incluido en el Dream Team del Balón de Oro.

Con el Fútbol Club Barcelona, al que estuvo ligado más de veinte años, ganó 34 títulos, entre ellos diez de La Liga, cuatro de la Liga de Campeones de la UEFA y siete de la Copa del Rey. Tiene, entre otros, los récords por más goles en una temporada,[12] en un mismo club y en un año calendario. Es, además, el máximo goleador histórico del Barcelona y de la selección argentina, de La Liga, la Supercopa de España, la Supercopa de Europa y el jugador no europeo con más goles en la Liga de Campeones de la UEFA. También es el futbolista con más títulos oficiales con diferentes clubes y selección.

Nacido y criado en la ciudad de Rosario, a los 13 años se radicó en España, donde el Barcelona accedió a pagar el tratamiento de la enfermedad hormonal que le habían diagnosticado de niño. Después de una rápida progresión por la Academia juvenil del Barcelona, hizo su debut oficial con el primer equipo en octubre de 2004, a los diecisiete años. A pesar de haber sido propenso a lesiones en los inicios de su carrera, ya en 2006 se estableció como jugador fundamental para el club. Su primera temporada ininterrumpida fue la 2008-09, en la que el Barcelona alcanzó el primer triplete del fútbol español. Por su estilo de juego de pequeño driblador zurdo,[13] pronto se lo comparó con su compatriota Diego Maradona quien, en 2007, lo declaró su «sucesor».

En 2009, a los veintidós años, ganó su primer Balón de Oro y el premio al Jugador Mundial de la FIFA del año. Siguieron tres temporadas exitosas, en las que ganó cuatro Balones de Oro de forma consecutiva, hecho que no tenía precedentes. Hasta el momento, su mejor campaña personal fue en 2011-12, cuando estableció el récord de más goles en una temporada, tanto en La Liga como en otras competiciones europeas. Durante las dos siguientes temporadas, también sufrió lesiones y, en 2014, perdió el Balón de Oro frente a Cristiano Ronaldo, a quien se considera su rival. Recuperó su mejor forma durante la campaña 2014-15, en la que superó los registros de máximo goleador absoluto en La Liga y la Liga de Campeones y logró con el Barcelona un histórico segundo triplete, además de ganar su quinto Balón de Oro. Volvería a ganarlo en 2019, 2021 y 2023.

Como internacional argentino, ha representado a su país en catorce torneos mayores. A nivel juvenil, en 2005 participó con la selección sub-20 en el Sudamericano de Colombia y ganó la Copa Mundial de Países Bajos, torneo en el que finalizó como mejor jugador y máximo goleador y, con la sub-23, recibió la medalla de oro en los Juegos Olímpicos de 2008. Después de debutar en la selección mayor en agosto de 2005, en el Mundial de Alemania 2006 se convirtió en el argentino más joven en jugar y en marcar en un mundial. Al año siguiente, en la Copa América, fue nombrado mejor jugador joven del torneo. Como capitán desde agosto de 2011, llegó con su equipo a las finales del Mundial de Brasil 2014, de la Copa América 2015, de la Copa América Centenario y del Mundial de Norteamérica 2026. Ganó, además, la Copa América 2021 ante Brasil en el Maracaná, la Finalissima 2022 frente a Italia en Wembley, el Mundial de Catar 2022 contra Francia en el estadio Lusail y la Copa América 2024 ante Colombia en el Hard Rock Stadium.

El 16 de junio, en el primer partido de Argentina contra Argelia en la Copa Mundial de Fútbol de 2026, pasó a ser el segundo futbolista en participar en seis mundiales de la FIFA e igualó el récord de Miroslav Klose de dieciséis goles."""

SUMMARY_TEMPLATE = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
"""


def summarize_person(llm: BaseChatModel, information: str) -> str:
    prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke(input={"information": information})
    return response.content


def main() -> None:
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    print(summarize_person(llm, INFORMATION))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Delete the old root script**

Run: `git rm main.py`

- [ ] **Step 4: Update CLAUDE.md run command**

In `CLAUDE.md`, replace:

```markdown
- Run the app: `uv run main.py`
```

with:

```markdown
- Run an example: `uv run examples/ch01_hello_chain.py`
```

- [ ] **Step 5: Verify the example still runs**

Run: `uv run examples/ch01_hello_chain.py`
Expected: prints a summary + two facts about Messi (requires local Ollama running `gemma3:270m`; if Ollama isn't running locally, confirm instead via `uv run python -c "import examples.ch01_hello_chain"` that the module imports cleanly with no syntax/import errors).

- [ ] **Step 6: Commit**

```bash
git add -A examples CLAUDE.md
git commit -m "refactor: move hello-world script into examples/ch01_hello_chain.py"
```

---

### Task 3: Add pytest with a fake-model test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_ch01_hello_chain.py`
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md` (Commands section)

**Interfaces:**
- Consumes: `summarize_person(llm, information)` and `INFORMATION` from `examples/ch01_hello_chain.py` (Task 2).

- [ ] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Append:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: tests that call a real, locally running model (e.g. Ollama) — excluded by default in CI",
]
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file)

- [ ] **Step 4: Write the failing test**

Create `tests/test_ch01_hello_chain.py`:

```python
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from examples.ch01_hello_chain import ChatOllama, summarize_person


def test_summarize_person_invokes_chain_and_returns_content():
    fake_llm = FakeListChatModel(responses=["1. Summary. 2. Fact one. Fact two."])

    result = summarize_person(fake_llm, information="Some short bio text.")

    assert result == "1. Summary. 2. Fact one. Fact two."


@pytest.mark.integration
def test_summarize_person_with_real_ollama():
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    result = summarize_person(llm, information="Ada Lovelace was a mathematician.")

    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 5: Run the fake-model test to verify it passes, and confirm the integration test is excludable**

Run: `uv run pytest -v`
Expected: `test_summarize_person_invokes_chain_and_returns_content` PASSES; `test_summarize_person_with_real_ollama` also attempts to run (and will pass/fail depending on whether Ollama is running locally — that's expected, it's a real integration test).

Run: `uv run pytest -m "not integration" -v`
Expected: only `test_summarize_person_invokes_chain_and_returns_content` runs, and PASSES.

- [ ] **Step 6: Update CLAUDE.md**

Add to the Commands section:

```markdown
- Run tests: `uv run pytest` (add `-m "not integration"` to skip tests that require a locally running Ollama model)
```

Also update the "There is no test suite configured yet." line — remove it.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests CLAUDE.md
git commit -m "test: add pytest with a fake-model chain test"
```

---

### Task 4: Wire up LangSmith tracing env vars

**Files:**
- Create: `.env.example`
- Modify: `CLAUDE.md` (Environment section)

**Interfaces:** None (env-var/documentation only; `langsmith` already reads `LANGCHAIN_*` env vars automatically once `load_dotenv()` has populated them — no code changes needed).

- [ ] **Step 1: Create `.env.example`**

```
# LLM providers
OPENAI_API_KEY=
GOOGLE_API_KEY=

# LangSmith tracing (https://smith.langchain.com) — optional but recommended:
# turns on full prompt/response traces for every chain call.
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=langchain-course
```

- [ ] **Step 2: Document it in CLAUDE.md**

In the Environment section of `CLAUDE.md`, add:

```markdown
- Copy `.env.example` to `.env` and fill in keys. Set `LANGCHAIN_TRACING_V2=true` plus `LANGCHAIN_API_KEY` to see full prompt/response traces for each chain call in the [LangSmith](https://smith.langchain.com) UI — `langsmith` is already a dependency and reads these vars automatically once `load_dotenv()` runs.
```

- [ ] **Step 3: Verify tracing picks up**

Run: `uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('LANGCHAIN_TRACING_V2'))"` after adding real values to your local `.env`.
Expected: prints `true` (confirms `load_dotenv()` + the env var name are wired correctly; the actual trace will show up in the LangSmith UI on the next real chain invocation).

- [ ] **Step 4: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: document LangSmith tracing env vars"
```

---

### Task 5: Add pyright type-checking

**Files:**
- Modify: `pyproject.toml`

**Interfaces:** None.

- [ ] **Step 1: Add pyright as a dev dependency**

Run: `uv add --dev pyright`

- [ ] **Step 2: Add pyright config**

Append to `pyproject.toml`:

```toml
[tool.pyright]
include = ["examples", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "basic"
```

- [ ] **Step 3: Run pyright and fix any reported issues**

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`. If it reports anything (e.g. a missing return type or an untyped fake-model call), fix the specific line it points at in `examples/ch01_hello_chain.py` or `tests/test_ch01_hello_chain.py` and re-run until clean.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyright type-checking"
```

---

### Task 6: Add pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: `CLAUDE.md` (Commands section)

**Interfaces:** None.

- [ ] **Step 1: Add pre-commit as a dev dependency**

Run: `uv add --dev pre-commit`

- [ ] **Step 2: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix]
  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright
        language: system
        types: [python]
        pass_filenames: false
```

- [ ] **Step 3: Pull the latest hook versions**

Run: `uv run pre-commit autoupdate`
Expected: rewrites the `rev:` pin for `ruff-pre-commit` to whatever its current latest tag is.

- [ ] **Step 4: Install the git hook and run it once against all files**

Run: `uv run pre-commit install`
Run: `uv run pre-commit run --all-files`
Expected: all hooks report `Passed` (ruff-format and ruff should be no-ops since Task 1 already formatted everything; pyright should be clean from Task 5).

- [ ] **Step 5: Update CLAUDE.md**

Add to the Commands section:

```markdown
- Install git hooks (one-time): `uv run pre-commit install`
- Type-check: `uv run pyright`
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml CLAUDE.md
git commit -m "chore: add pre-commit hooks for ruff and pyright"
```

---

### Task 7: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:** None.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups
      - run: uv run ruff format --check .
      - run: uv run ruff check .
      - run: uv run pyright
      - run: uv run pytest -m "not integration"
```

- [ ] **Step 2: Validate the workflow file is well-formed YAML**

Run: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('valid yaml')"`
Expected: prints `valid yaml` (add `uv add --dev pyyaml` first only if `yaml` isn't already importable — check with `uv run python -c "import yaml"` before adding a dependency just for this check; if it's not present, it's fine to skip this validation step and instead visually confirm indentation matches the block above).

- [ ] **Step 3: Commit and push, then confirm the run**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for ruff, pyright, and pytest"
```

Push the branch and open the Actions tab (or run `gh run list --limit 1` after pushing) to confirm the workflow triggers and every step passes. `pytest -m "not integration"` should pass in CI even without Ollama installed on the runner, since the integration test is excluded.

---

## Suggested follow-ups (not part of this plan)

- LangGraph, `.with_structured_output()`, and LangSmith eval datasets are agent-authoring topics, not harness/tooling — pull them in when an actual multi-step or structured-output exercise is written, per the original recommendations doc.
