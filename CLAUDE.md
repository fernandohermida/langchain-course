# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A learning/course repository for LangChain, managed with `uv`. Exercises live as standalone scripts under `examples/` (e.g. `examples/ch01_hello_chain.py`) — expect more to be added as the course progresses.

## Commands

- Run an example: `uv run examples/ch01_hello_chain.py`
- Add a dependency: `uv add <package>`
- Sync the environment after pulling changes (installs deps from `uv.lock`): `uv sync`
- Run any script inside the project's venv: `uv run <script.py>`
- Format code: `uv run ruff format .`
- Lint (and autofix): `uv run ruff check --fix .`
- Run tests: `uv run pytest` (add `-m "not integration"` to skip tests that require a locally running Ollama model)

## Environment

- Python `>=3.12` (pinned to 3.12 in `.python-version`).
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock` — always add packages via `uv add` rather than editing `pyproject.toml` by hand, so the lockfile stays in sync.
- Key dependencies: `langchain`, `langchain-openai`, `python-dotenv` — expect API keys (e.g. `OPENAI_API_KEY`) to be loaded from a `.env` file via `python-dotenv` rather than hardcoded.
- Copy `.env.example` to `.env` and fill in keys. Set `LANGCHAIN_TRACING_V2=true` plus `LANGCHAIN_API_KEY` to see full prompt/response traces for each chain call in the [LangSmith](https://smith.langchain.com) UI — `langsmith` is already a dependency and reads these vars automatically once `load_dotenv()` runs.
- `ruff` is a dev dependency handling both formatting and linting; config lives in `[tool.ruff]` in `pyproject.toml`.
