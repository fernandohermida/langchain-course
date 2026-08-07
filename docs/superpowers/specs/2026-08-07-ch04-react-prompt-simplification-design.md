# ch04_react_prompt.py simplification design

## Context

`examples/ch04_react_prompt.py` is the course's hand-rolled ReAct-prompting exercise: it demonstrates that an agent's Thought/Action/Observation loop can be implemented over a plain-text completion (no `tools=` schema), by parsing the model's raw output with regex and re-injecting tool results as text. Over the course of debugging real runs against a local `qwen3:1.7b` model, the file accumulated several defensive fixes: a fallback for when the model collapses "Action:"/"Action Input:" onto one line, a fallback for call-syntax action names (`get_product_price(product="x")`), an early-stop guard for deterministically-repeated failures, and reflection-based (`inspect.signature`) auto-generation of the tool-description text in the prompt.

Each fix was individually justified at the time, but together they've made the file harder to read as a *teaching* example — a student trying to learn "what does raw ReAct parsing look like" now has to read through several layers of quirk-handling that aren't about the core concept. The goal of this change is to strip the file back down to the smallest version that still teaches the intended lesson: a plain-text ReAct loop, its tool dispatch, and the fact that regex-based parsing is fragile (a deliberate, visible property of the exercise, not something to be engineered away).

## Decisions

1. **Parsing stays minimal and visibly fragile.** Revert to a single two-line regex parse (`Action:` / `Action Input:`). No call-syntax stripping, no single-line collapse fallback, no repeated-failure early-stop guard. If the model deviates from the expected format, `parse_model_output` returns `(None, None, None)` and `run_agent_loop` prints the parse error and stops — matching the file's own existing framing that this fragility is the point of the exercise.
2. **Tool descriptions are a hardcoded string**, not generated via `inspect.signature()`/`inspect.unwrap()` reflection. Removes the `inspect` import, `get_tools_description_new`, and `_annotation_text` entirely. Trade-off (accepted): the description text must be kept in sync by hand if a tool's signature changes — acceptable for a fixed teaching example.
3. **LLM call tracing collapses to one function.** The current two-layer trace (`ollama_chat_traced` as `run_type="llm"`, wrapped by `call_model` as `run_type="chain"`, with a comment explaining why they're split) becomes a single `@traceable(run_type="llm")` function that calls `ollama.chat` directly — matching the simplicity of `ch01_hello_chain.py`. Tracing itself is kept (per `CLAUDE.md`'s repo-wide convention that every example is individually identifiable in LangSmith), just with one less layer of indirection.
4. **STRICT RULE 4 (ask the user when the discount tier is missing) is removed**, along with the `ask_user`-shaped framing in the prompt's rule list. Only one demo query remains (see below) and it never exercises this rule — keeping an untested rule in the prompt would leave a student unable to see it in action.
5. **Demo trims to one query**: `"What is the price of a laptop after applying a gold discount?"` — exercises both tools in the required order (price before discount) and STRICT RULES 1–3 (must call `get_product_price` first, never hand-calculate the discount, always use `apply_discount`), with no ambiguity to resolve.
6. **`think=False` and `num_predict=300` stay** on the Ollama call options — this is what keeps the demo from hanging (qwen3's native "thinking" mode can otherwise generate unboundedly when confused), and it's two kwargs with a one-line comment, not conceptual complexity. The comment is shortened from the current paragraph.
7. **Tool-error-to-Observation handling stays as-is** (`try/except` around tool dispatch, turning exceptions into `"Error: ..."` strings) — this is a small, valuable, on-concept lesson: tool failures become Observations the model can react to, not crashes.

## Non-goals

- No change to the two tools' actual behavior (`get_product_price`, `apply_discount`) or their signatures.
- No change to the overall loop shape (build prompt → call model → parse → dispatch tool → append Observation → repeat) or to `MAX_ITERATIONS`.
- No attempt to make the simplified parser handle every real-world model quirk — visible fragility is intentional, not a bug to re-fix here.

## Testing

`tests/test_ch04_react_prompt.py` currently has 8 tests, several of which cover the fallbacks/guard being removed (call-syntax parsing variants, the single-line-collapse full-loop test, the repeated-failure early-stop test). After this change:
- Remove tests that exercise removed behavior: the call-syntax tests, the single-line-collapse recovery test, and the repeated-failure early-stop test.
- Keep unchanged: the `think`/`num_predict` capture test (`test_call_model_disables_thinking_and_caps_output_length`) — it patches `ollama.chat` directly and calls `call_model`, which still holds after the tracing collapse in decision 3, since `call_model` still calls `ollama.chat` with the same `think`/`options` kwargs.
- Keep/add: a `parse_model_output` happy-path test (clean two-line `Action:`/`Action Input:` format), a test asserting a malformed/deviating completion now correctly returns `(None, None, None)` with no recovery attempted, and one mocked full-loop test covering the single kept demo scenario (two tool calls then a Final Answer).

## Verification

- `uv run pytest tests/test_ch04_react_prompt.py -v` — updated tests pass.
- `uv run pytest -m "not integration"` — full suite still green.
- `uv run ruff check examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py` — no new lint errors.
- `uv run pyright examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py` — no type errors.
- If Ollama is available locally: `uv run examples/ch04_react_prompt.py` and confirm the single demo reaches a Final Answer.
