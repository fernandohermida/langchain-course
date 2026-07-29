# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A learning/course repository for LangChain, managed with `uv`. Exercises live as standalone scripts under `examples/`, named `ch<NN>_<topic>.py`, each paired with a test at `tests/test_ch<NN>_<topic>.py` — expect more to be added as the course progresses.

- **ch01** — `ch01_hello_chain.py`: a first LCEL chain (`prompt | llm`), summarizing a bio via `ChatOllama`.
- **ch02** — `ch02_search_agent.py` / `ch02_search_agent_2.py` / `ch02_search_job_agent.py`: building search agents with LangChain v1's `create_agent`, progressing from a stubbed search tool, to a real Tavily-backed tool, to the official `TavilySearch` tool with a structured (Pydantic) response format.
- **ch03** — `ch03_agent_loop_langchain_tool_calling.py` / `ch03_agent_loop_raw_tool_calling.py`: the same hand-rolled shopping-assistant tool-calling loop implemented twice, first via LangChain's `bind_tools`/`ToolMessage` abstractions, then via a raw `ollama.chat()` call with a hand-authored JSON tool schema — written to show what an agent's tool-calling loop is actually doing under the hood.

## Commands

- Run an example: `uv run examples/ch01_hello_chain.py` (swap in any `ch<NN>_*.py` file)
- Add a dependency: `uv add <package>`
- Sync the environment after pulling changes (installs deps from `uv.lock`): `uv sync`
- Run any script inside the project's venv: `uv run <script.py>`
- Format code: `uv run ruff format .`
- Lint (and autofix): `uv run ruff check --fix .`
- Run tests: `uv run pytest` (add `-m "not integration"` to skip tests that need a locally running Ollama model or real external APIs)
- Type-check: `uv run pyright`
- Install git hooks (one-time): `uv run pre-commit install`

## Environment

- Python `>=3.12` (pinned to 3.12 in `.python-version`).
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock` — always add packages via `uv add` rather than editing `pyproject.toml` by hand, so the lockfile stays in sync.
- Copy `.env.example` to `.env` and fill in keys as needed per chapter:
  - **ch01** needs `OPENAI_API_KEY` and/or `GOOGLE_API_KEY` for cloud models, or a locally running Ollama (no key needed) for the default `ChatOllama` model.
  - **ch02** needs `OPENAI_API_KEY`, plus `TAVILY_API_KEY` for the two Tavily-backed variants (`ch02_search_agent_2.py`, `ch02_search_job_agent.py`).
  - **ch03** needs a locally running Ollama (`qwen3:1.7b` by default) — no cloud API keys required.
- Set `LANGCHAIN_TRACING_V2=true` plus `LANGCHAIN_API_KEY` to see full traces in the [LangSmith](https://smith.langchain.com) UI — `langsmith` is already a dependency and reads these vars automatically once `load_dotenv()` runs. Every example passes an explicit `run_name`/`tags` (or `@traceable(name=...)` for the ch03 raw-loop scripts) so its runs are individually identifiable in LangSmith rather than appearing as anonymous chain/agent runs.
- `ruff` is a dev dependency handling both formatting and linting; config lives in `[tool.ruff]` in `pyproject.toml`.
