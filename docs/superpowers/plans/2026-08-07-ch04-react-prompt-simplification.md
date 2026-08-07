# ch04 ReAct Prompt Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip `examples/ch04_react_prompt.py` back down to the smallest version that still teaches a plain-text ReAct loop, so a student can read it without wading through defensive fixes for real-model quirks.

**Architecture:** No structural change — same prompt template → `call_model` → `parse_model_output` → tool dispatch → scratchpad append loop. This plan removes three layers added while debugging real `qwen3:1.7b` runs (parsing fallbacks + early-stop guard, reflection-based tool-description generation, double-layered LLM tracing) and trims the prompt/demo to match.

**Tech Stack:** Python 3.12, `ollama` (Python client), `langsmith` (`@traceable`), `pytest` + `pytest-monkeypatch`, `ruff`, `pyright`, managed with `uv`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-ch04-react-prompt-simplification-design.md`
- No change to `get_product_price` / `apply_discount` behavior or signatures.
- No change to `MAX_ITERATIONS` or the overall loop shape.
- Parsing stays intentionally fragile — do not re-add recovery for model-format deviations.
- Every task must end green on: `uv run pytest tests/test_ch04_react_prompt.py -v`, `uv run ruff check examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`, `uv run pyright examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`.

---

### Task 1: Revert the parser and loop to minimal, intentionally-fragile behavior

**Files:**
- Modify: `examples/ch04_react_prompt.py:204-271` (`_CALL_SYNTAX_RE`, `_split_call_syntax`, `parse_model_output`)
- Modify: `examples/ch04_react_prompt.py:289-357` (`run_agent_loop`'s `previous_failed_call` tracking)
- Test: `tests/test_ch04_react_prompt.py` (full rewrite)

**Interfaces:**
- Consumes: nothing new — `parse_model_output(text: str) -> tuple[str | None, str | None, str | None]` and `run_agent_loop(query: str) -> str | None` keep their existing signatures.
- Produces: `parse_model_output` and `run_agent_loop` with their original (pre-session) minimal behavior, for Task 2/3 to build on unchanged.

- [ ] **Step 1: Write the two `parse_model_output` tests**

Replace the entire contents of `tests/test_ch04_react_prompt.py` with:

```python
from types import SimpleNamespace

from examples import ch04_react_prompt as ch04


def _chat_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_parse_model_output_happy_path():
    text = (
        "Thought: I need the price.\n"
        "Action: get_product_price\n"
        "Action Input: laptop"
    )

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "get_product_price"
    assert action_input == "laptop"
    assert final_answer is None


def test_parse_model_output_returns_none_when_action_input_line_missing():
    text = (
        "Thought: I need the price.\n"
        'Action: get_product_price, product="smartphone"'
    )

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action is None
    assert action_input is None
    assert final_answer is None


def test_run_agent_loop_executes_both_tools_then_returns_final_answer(monkeypatch):
    responses = iter(
        [
            _chat_response(
                "Thought: I need the laptop's price first.\n"
                "Action: get_product_price\n"
                "Action Input: laptop"
            ),
            _chat_response(
                "Thought: Now I'll apply the gold discount.\n"
                "Action: apply_discount\n"
                "Action Input: 999.99, gold"
            ),
            _chat_response(
                "Thought: I now know the final answer\n"
                "Final Answer: The laptop costs $849.99 after the gold discount."
            ),
        ]
    )
    monkeypatch.setattr(ch04.ollama, "chat", lambda **_kwargs: next(responses))

    result = ch04.run_agent_loop(
        "What is the price of a laptop after applying a gold discount?"
    )

    assert result == "The laptop costs $849.99 after the gold discount."


def test_call_model_disables_thinking_and_caps_output_length(monkeypatch):
    captured_kwargs = {}

    def fake_chat(**kwargs):
        captured_kwargs.update(kwargs)
        return _chat_response("Final Answer: done")

    monkeypatch.setattr(ch04.ollama, "chat", fake_chat)

    ch04.call_model("some prompt")

    assert captured_kwargs["think"] is False
    assert captured_kwargs["options"]["num_predict"] > 0
```

This drops the call-syntax tests, the single-line-collapse recovery test, and the
repeated-failure early-stop test (all test behavior being removed this task), and
adds `test_parse_model_output_returns_none_when_action_input_line_missing` as the
one genuinely red test — the current code recovers this input via comma-split, so
it will fail until Step 3 reverts the parser.

- [ ] **Step 2: Run tests to verify the new one fails, others pass**

Run: `uv run pytest tests/test_ch04_react_prompt.py -v`
Expected: `test_parse_model_output_returns_none_when_action_input_line_missing` FAILS
(current code returns a non-None action via the comma-split fallback). The other
three tests PASS already (they test behavior this task doesn't change).

- [ ] **Step 3: Revert `parse_model_output` to the minimal two-regex version**

In `examples/ch04_react_prompt.py`, delete `_CALL_SYNTAX_RE` and `_split_call_syntax`
(current lines 204-215), and replace the body of `parse_model_output` from
`action_match = re.search(...)` through the end of the function with:

```python
    action_match = re.search(r"Action:\s*(.+)", text)
    action_input_match = re.search(r"Action Input:\s*(.+)", text)
    if not action_match or not action_input_match:
        # Neither a Final Answer nor a well-formed Action was found — the
        # model didn't follow the prompt's format. All three come back None
        # so the caller can detect this case and stop the loop.
        return None, None, None

    return action_match.group(1).strip(), action_input_match.group(1).strip(), None
```

The function's docstring and the "CHANGE 6" / "Checked first" comments above this
block stay unchanged — they already describe this exact minimal, fragile behavior.

- [ ] **Step 4: Remove the repeated-failure early-stop guard from `run_agent_loop`**

In `examples/ch04_react_prompt.py`, delete the `previous_failed_call` setup block
(the comment + `previous_failed_call: tuple[str, str] | None = None` line, currently
just after `scratchpad = ""`), and delete the guard block after
`print(f"  [Tool Result] {observation}")`:

```python
        current_call = (action, action_input)
        if observation.startswith("Error:") and current_call == previous_failed_call:
            print(
                "ERROR: Model repeated the identical failing action "
                f"{action!r} with input {action_input!r} on consecutive "
                "iterations. temperature=0 makes this deterministic — it will "
                "never self-correct — so stopping early instead of burning "
                "the remaining iterations."
            )
            return None
        previous_failed_call = (
            current_call if observation.startswith("Error:") else None
        )
```

So that the line right after the `print(f"  [Tool Result] {observation}")` line is
directly the `# Append what the model said *and* what actually happened, ...`
comment followed by the `scratchpad +=` line.

- [ ] **Step 5: Run tests to verify all pass**

Run: `uv run pytest tests/test_ch04_react_prompt.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: only the 3 pre-existing `REACT_PROMPT_TEMPLATE` line-length warnings (fixed
in Task 2) — no new errors from this task's changes.

Run: `uv run pyright examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 7: Commit**

```bash
git add examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py
git commit -m "$(cat <<'EOF'
Revert ch04 ReAct parser and loop to minimal, intentionally-fragile behavior

Removes the call-syntax/single-line-collapse parsing fallbacks and the
repeated-failure early-stop guard added while debugging real qwen3 runs,
per docs/superpowers/specs/2026-08-07-ch04-react-prompt-simplification-design.md.
The parser's fragility on model format deviations is the intended lesson.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Hardcode tool descriptions, drop STRICT RULE 4, trim demo to one query

**Files:**
- Modify: `examples/ch04_react_prompt.py:1` (remove `import inspect`)
- Modify: `examples/ch04_react_prompt.py:66-70` (comment referencing STRICT RULE 4)
- Modify: `examples/ch04_react_prompt.py:79-127` (remove `get_tools_description_new`/`_annotation_text`, hardcode `tool_descriptions`)
- Modify: `examples/ch04_react_prompt.py:129-154` (`REACT_PROMPT_TEMPLATE`)
- Modify: `examples/ch04_react_prompt.py:363-378` (`__main__` block)

**Interfaces:**
- Consumes: `tools` dict and `tool_names` string (unchanged from before this task).
- Produces: `tool_descriptions: str` (now a plain constant instead of a function call) and `REACT_PROMPT_TEMPLATE: str` (3 rules instead of 4) — both consumed unchanged by `run_agent_loop` via `REACT_PROMPT_TEMPLATE.format(question=query)`.

- [ ] **Step 1: Remove the `inspect` import**

Delete line 1 (`import inspect`) from `examples/ch04_react_prompt.py`.

- [ ] **Step 2: Remove the reflection-based description functions and hardcode the string**

Delete `get_tools_description_new` and `_annotation_text` in full (current lines
79-115), and replace the `tools`/`tool_descriptions` block (current lines 118-127)
with:

```python
# Annotated Callable[..., float] (rather than left to inference) because the
# two tools have different arities/parameter types — run_agent_loop dispatches
# through this dict with a plain `*args` list of strings, so the dict's value
# type needs to be loose enough to cover both signatures.
tools: dict[str, Callable[..., float]] = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}
tool_names = ", ".join(tools.keys())
tool_descriptions = (
    "get_product_price(product): looks up the real price of a product.\n"
    "apply_discount(price, discount_tier): applies a discount "
    "(bronze, silver, gold, platinum, diamond) to a price."
)
```

- [ ] **Step 3: Drop STRICT RULE 4 from the prompt template**

Replace `REACT_PROMPT_TEMPLATE` (current lines 129-154) with:

```python
REACT_PROMPT_TEMPLATE = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price
   first to get the real price.
2. Only call apply_discount AFTER you have received a price from
   get_product_price. Pass the exact price returned by get_product_price —
   do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the
   apply_discount tool.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""
```

- [ ] **Step 4: Update the now-stale STRICT RULE 4 reference in `apply_discount`**

In `apply_discount`, replace:

```python
    if discount_tier not in discount_percentages:
        # Same "fail loudly" pattern as get_product_price above: an unknown
        # tier becomes an Observation the model can react to (e.g. by asking
        # the user, per STRICT RULE 4 in the prompt), instead of a raw KeyError
        # traceback that would kill the whole agent loop.
        raise ValueError(
```

with:

```python
    if discount_tier not in discount_percentages:
        # Same "fail loudly" pattern as get_product_price above: an unknown
        # tier becomes an Observation the model can react to, instead of a
        # raw KeyError traceback that would kill the whole agent loop.
        raise ValueError(
```

- [ ] **Step 5: Trim the demo to one query**

Replace the `__main__` block (current lines 363-378) with:

```python
if __name__ == "__main__":
    print("Starting the agent loop...(raw ReAct prompting)")
    print()
    run_agent_loop("What is the price of a laptop after applying a gold discount?")
```

- [ ] **Step 6: Run tests to verify nothing broke**

Run: `uv run pytest tests/test_ch04_react_prompt.py -v`
Expected: all 4 tests PASS (none of them assert on prompt/demo content, only on
`parse_model_output` and the mocked tool-call sequence).

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: `All checks passed!` (the 3 pre-existing line-length warnings are fixed by
this task's rewritten `REACT_PROMPT_TEMPLATE`).

Run: `uv run pyright examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 8: Commit**

```bash
git add examples/ch04_react_prompt.py
git commit -m "$(cat <<'EOF'
Hardcode ch04 tool descriptions, drop untested STRICT RULE 4, trim to one demo

Replaces inspect.signature()-based tool-description generation with a plain
string, removes the discount-tier-ask rule since no remaining demo exercises
it, and trims the demo to a single happy-path query, per
docs/superpowers/specs/2026-08-07-ch04-react-prompt-simplification-design.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Collapse the double-layered LLM call tracing into one function

**Files:**
- Modify: `examples/ch04_react_prompt.py:157-201` (`ollama_chat_traced` + `call_model`)

**Interfaces:**
- Consumes: `MODEL` constant, `ollama.chat` (unchanged).
- Produces: `call_model(prompt: str) -> str` — same signature and behavior as before, called unchanged by `run_agent_loop` and by `test_call_model_disables_thinking_and_caps_output_length`.

- [ ] **Step 1: Replace the two functions with one traced `call_model`**

Replace the entire block from `# CHANGE 4: Drop tools=...` through the end of
`call_model` (current lines 157-201) with:

```python
# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.
# CHANGE 5: One prompt string replaces the system/user message split from
# ch02/ch03. ollama.chat still requires a `messages` list, so the entire
# ReAct prompt — instructions, tool descriptions, and the scratchpad built up
# so far — travels as a single "user" message rather than being split across
# system/user/tool roles.
@traceable(name="call_model", run_type="llm")
def call_model(prompt: str) -> str:
    """Call the model with a plain-text completion (no `tools=` schema — the
    model can only ever be *steered* via the prompt text itself)."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        # think=False: qwen3 is a reasoning model that would otherwise emit
        # its own <think> block on top of our "Thought:" step, sometimes at
        # unbounded length.
        think=False,
        options={
            # "stop" cuts generation off as soon as the model starts writing
            # "\nObservation" itself — without it, the model could hallucinate
            # a tool result instead of waiting for the real one we inject in
            # run_agent_loop. num_predict is a hard backstop so a confused
            # completion can't run forever even if "stop" never matches.
            "stop": ["\nObservation"],
            "temperature": 0,
            "num_predict": 300,
        },
    )
    # ollama types Message.content as `str | None`; an empty completion is
    # still a valid (if useless) string for parse_model_output to fail on.
    return response.message.content or ""
```

- [ ] **Step 2: Run tests to verify all pass**

Run: `uv run pytest tests/test_ch04_react_prompt.py -v`
Expected: all 4 tests PASS, including
`test_call_model_disables_thinking_and_caps_output_length` — it patches
`ch04.ollama.chat` directly and calls `ch04.call_model`, which still calls
`ollama.chat` with the same `think`/`options` kwargs after this collapse.

- [ ] **Step 3: Lint and type-check**

Run: `uv run ruff check examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: `All checks passed!`

Run: `uv run pyright examples/ch04_react_prompt.py tests/test_ch04_react_prompt.py`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 4: Commit**

```bash
git add examples/ch04_react_prompt.py
git commit -m "$(cat <<'EOF'
Collapse ch04's double-layered LLM call tracing into one function

ollama_chat_traced (run_type=llm) wrapped by call_model (run_type=chain)
nested two LLM-looking trace spans around one real completion. Collapses
to a single @traceable(run_type="llm") call_model, matching ch01's
simplicity, per
docs/superpowers/specs/2026-08-07-ch04-react-prompt-simplification-design.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full non-integration suite**

Run: `uv run pytest -m "not integration"`
Expected: all tests pass, including the other chapters' tests (unaffected by this
change) and ch04's 4 tests.

- [ ] **Step 2: Run lint and type-check across the whole repo**

Run: `uv run ruff check .`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 3: Manually run the simplified example (if Ollama is available locally)**

Run: `uv run examples/ch04_react_prompt.py`
Expected: the single demo query reaches a `Final Answer:` after calling
`get_product_price` then `apply_discount`, without hitting
`ERROR: Max iterations reached`. If the model deviates from the expected
Action/Action Input format, the loop will print
`ERROR: Could not parse Action/Action Input from LLM output` and stop after one
iteration — that is expected, intentional behavior per this plan's spec, not a bug
to fix.

- [ ] **Step 4: Confirm no leftover references to removed code**

Run: `grep -rn "get_tools_description_new\|_annotation_text\|_split_call_syntax\|_CALL_SYNTAX_RE\|ollama_chat_traced\|previous_failed_call\|STRICT RULE 4" examples/ tests/`
Expected: no matches.
