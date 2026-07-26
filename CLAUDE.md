# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A learning/course repository for LangChain, managed with `uv`. This branch demonstrates building a search agent with LangChain v1's `create_agent` interface: a `TavilySearch` tool and a structured (Pydantic) response format (`examples/ch02_search_agent.py`).

## Commands

- Run the example: `uv run examples/ch02_search_agent.py`
- Add a dependency: `uv add <package>`
- Sync the environment after pulling changes (installs deps from `uv.lock`): `uv sync`
- Run any script inside the project's venv: `uv run <script.py>`
- Format code: `uv run ruff format .`
- Lint (and autofix): `uv run ruff check --fix .`
- Run tests: `uv run pytest` (add `-m "not integration"` to skip tests that call the real OpenAI/Tavily APIs)
- Type-check: `uv run pyright`
- Install git hooks (one-time): `uv run pre-commit install`

## Environment

- Python `>=3.12` (pinned to 3.12 in `.python-version`).
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock` — always add packages via `uv add` rather than editing `pyproject.toml` by hand, so the lockfile stays in sync.
- Key dependencies: `langchain`, `langchain-openai`, `langchain-tavily`, `python-dotenv` — expect API keys to be loaded from a `.env` file via `python-dotenv` rather than hardcoded.
- Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `TAVILY_API_KEY` — both are required to run `examples/ch02_search_agent.py` for real.
- Set `LANGCHAIN_TRACING_V2=true` plus `LANGCHAIN_API_KEY` to see full prompt/response traces for each agent call in the [LangSmith](https://smith.langchain.com) UI — `langsmith` is already a dependency and reads these vars automatically once `load_dotenv()` runs.
- `ruff` is a dev dependency handling both formatting and linting; config lives in `[tool.ruff]` in `pyproject.toml`.
