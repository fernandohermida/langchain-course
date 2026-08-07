import inspect
import re
from collections.abc import Callable
from typing import Literal

import ollama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog."""
    print(f"Searching for the price of {product}...")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "tablet": 499.99,
    }
    normalized = product.strip().lower()
    if normalized not in prices:
        # Fail loudly instead of guessing a price — this is what feeds the agent
        # loop's try/except -> "Error: ..." tool-message path, so the model sees
        # the failure and can react instead of getting a silently made-up number.
        raise ValueError(
            f"Unknown product '{product}'. Available products: {', '.join(prices)}"
        )
    return prices[normalized]


# Valid tiers live in the type hint (not just the docstring) so the tool
# schema below can list them as an enum — the model is structurally
# discouraged from inventing a tier before the tool ever runs.
DiscountTier = Literal["bronze", "silver", "gold", "platinum", "diamond"]


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: DiscountTier) -> float:
    """Apply a discount to the price of a product."""
    print(
        f"  >> Executing apply_discount with price: {price} "
        f"and discount tier: {discount_tier}"
    )

    # Action Input arrives as plain text parsed by regex (see parse_model_output
    # / run_agent_loop below), so `price` is really a str at the call site even
    # though the type hint says float — Python doesn't enforce annotations at
    # runtime, and `str * float` would raise TypeError. Cast explicitly instead
    # of trusting the caller.
    price = float(price)

    discount_percentages = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
        "platinum": 0.20,
        "diamond": 0.25,
    }
    if discount_tier not in discount_percentages:
        # Same "fail loudly" pattern as get_product_price above: an unknown
        # tier becomes an Observation the model can react to (e.g. by asking
        # the user, per STRICT RULE 4 in the prompt), instead of a raw KeyError
        # traceback that would kill the whole agent loop.
        raise ValueError(
            f"Unknown discount tier '{discount_tier}'. "
            f"Available tiers: {', '.join(discount_percentages)}"
        )

    return round(price * (1 - discount_percentages[discount_tier]), 2)


def get_tools_description_new(tools_dict: dict[str, Callable]) -> str:
    """Generate a description of the available tools for the model prompt.

    Uses inspect.signature() instead of a hand-written description per tool,
    so the prompt can't silently drift out of sync with a tool's actual
    parameters. Each annotation is rendered via str() rather than collapsed
    to a bare type name, so a Literal/enum constraint (e.g. discount_tier:
    Literal["bronze", "silver", ...]) stays visible in the rendered text
    instead of collapsing to a bare "string".

    inspect.unwrap() strips decorators like @traceable before inspecting —
    otherwise inspect.signature() picks up the wrapper's own __signature__
    (with its injected `config` kwarg) instead of the tool's real params,
    the exact pitfall called out in ch03_agent_loop_raw_tool_calling.py's
    TOOLS-schema comment.
    """
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        signature = inspect.signature(inspect.unwrap(tool_func))
        params = ", ".join(
            f"{name}: {_annotation_text(param.annotation)}"
            for name, param in signature.parameters.items()
        )
        first_doc_line = next(iter((tool_func.__doc__ or "").strip().splitlines()), "")
        descriptions.append(f"{tool_name}({params}): {first_doc_line}")
    return "\n".join(descriptions)


def _annotation_text(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    # typing generics like Literal[...] expose a __name__ too, but it's just
    # the bare special-form name ("Literal") — str() is what actually shows
    # the constrained values, so __name__ is only used for plain classes.
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


# Annotated Callable[..., float] (rather than left to inference) because the
# two tools have different arities/parameter types — run_agent_loop dispatches
# through this dict with a plain `*args` list of strings, so the dict's value
# type needs to be loose enough to cover both signatures.
tools: dict[str, Callable[..., float]] = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}
tool_names = ", ".join(tools.keys())
tool_descriptions = get_tools_description_new(tools)

REACT_PROMPT_TEMPLATE = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price
   first to get the real price.
2. Only call apply_discount AFTER you have received a price from
   get_product_price. Pass the exact price returned by get_product_price —
   do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the
   apply_discount tool.
4. If the user does not specify a discount tier, do NOT call a tool to ask —
   there is no tool for that. Instead, skip straight to "Final Answer:" and
   ask the user which tier to use. Do NOT assume one.

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


# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    # think=False: qwen3 is a reasoning model that emits its own <think> block
    # by default, on top of the ReAct prompt's own "Thought:" step. When the
    # model gets confused (e.g. after a tool-not-found Observation), that
    # native reasoning can run on for a very long time — and since it never
    # naturally produces "\nObservation", the "stop" option below never
    # triggers either. We only want the one, plain-text ReAct Thought.
    return ollama.chat(model=model, messages=messages, think=False, options=options)


# IMPROVEMENT over the original exercise: call_model no longer carries its own
# run_type="llm" trace. It's plumbing (build messages, pick sampling options),
# not the actual network call — that already happens, and is already traced,
# inside ollama_chat_traced (CHANGE 4). Tracing both as "llm" spans would nest
# two LLM-looking runs around a single real completion, which is confusing to
# read in LangSmith. A plain "chain" trace keeps call_model visible in the
# trace tree without double-counting the LLM call itself.
@traceable(name="call_model", run_type="chain")
def call_model(prompt: str) -> str:
    """Call the model with a plain-text completion (no `tools=` schema — the
    model can only ever be *steered* via the prompt text itself)."""
    # CHANGE 5: One prompt string replaces the system/user message split from
    # ch02/ch03. ollama.chat still requires a `messages` list, so the entire
    # ReAct prompt — instructions, tool descriptions, and the scratchpad built
    # up so far — travels as a single "user" message rather than being split
    # across system/user/tool roles.
    response = ollama_chat_traced(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        # The "stop" option cuts generation off as soon as the model starts
        # writing "\nObservation" itself. Without it, a plain-text completion
        # model has nothing stopping it from *hallucinating* a tool result —
        # we need to inject the real Observation ourselves, in run_agent_loop.
        # num_predict is a hard backstop: a ReAct step is a handful of lines,
        # so nothing legitimate needs more than this — it exists purely so a
        # confused/rambling completion can't run unbounded (Ollama's default
        # is -1, i.e. no limit) even if "stop" never matches.
        options={"stop": ["\nObservation"], "temperature": 0, "num_predict": 300},
    )
    # ollama types Message.content as `str | None`; an empty completion is
    # still a valid (if useless) string for parse_model_output to fail on.
    return response.message.content or ""


def parse_model_output(text: str) -> tuple[str | None, str | None, str | None]:
    """Parse a raw completion into (action, action_input, final_answer).

    Exactly one of (action, action_input) or final_answer should be set."""
    # CHANGE 6: Parse the tool call out of raw text with regex. This is
    # fragile — it silently breaks if the model doesn't echo the exact
    # "Action: ... / Action Input: ..." wording from REACT_PROMPT_TEMPLATE —
    # but that fragility is the whole point of this exercise: it's exactly
    # what passing a structured `tools=` schema (ch02/ch03) exists to avoid.

    # Checked first: once the model has decided it's done, it stops asking for
    # tools, so a "Final Answer:" line always wins over any Action/Action Input
    # left over from earlier in the same completion.
    final_answer_match = re.search(r"Final Answer:\s*(.+)", text)
    if final_answer_match:
        return None, None, final_answer_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+)", text)
    action_input_match = re.search(r"Action Input:\s*(.+)", text)
    if not action_match or not action_input_match:
        # Neither a Final Answer nor a well-formed Action was found — the
        # model didn't follow the prompt's format. All three come back None
        # so the caller can detect this case and stop the loop.
        return None, None, None

    return action_match.group(1).strip(), action_input_match.group(1).strip(), None


@traceable(name="ch04_react_prompt")
def run_agent_loop(query: str) -> str | None:
    """Run the ReAct loop: build the scratchpad, call the model, parse its
    output, execute the requested tool, append the result as an
    "Observation:", and repeat until "Final Answer:" appears or
    MAX_ITERATIONS is hit."""
    print(f"Question: {query}")
    print("=" * 60)

    prompt = REACT_PROMPT_TEMPLATE.format(question=query)
    # CHANGE 7: History is one growing string re-sent every iteration, instead
    # of a messages list to append to (ch02/ch03). There's no per-turn `role`
    # structure here — the "conversation" is really just one completion that
    # keeps getting extended with what the model said plus what really
    # happened when we ran its requested tool.
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        output = call_model(prompt + scratchpad)
        print(f"LLM Output:\n{output}")

        action, action_input, final_answer = parse_model_output(output)

        if final_answer is not None:
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        if action is None or action_input is None:
            print("ERROR: Could not parse Action/Action Input from LLM output")
            break

        print(f"  [Tool Selected] {action} with args: {action_input}")

        # Action Input is "comma separated values" per the prompt's format
        # instructions. Some models echo "key=value" pairs instead of bare
        # values (e.g. "product=laptop") — strip an optional "key=" prefix so
        # both styles resolve to the same positional args.
        raw_args = [arg.strip() for arg in action_input.split(",")]
        args = [arg.split("=", 1)[-1].strip().strip("'\"") for arg in raw_args]

        print(f"  [Tool Executing] {action}({args})...")
        if action not in tools:
            observation = (
                f"Error: Tool '{action}' not found. "
                f"Available tools: {list(tools.keys())}"
            )
        else:
            try:
                observation = str(tools[action](*args))
            except Exception as e:
                # Turned into a plain string the model can read as an
                # Observation and react to (e.g. re-check the product name),
                # instead of an unhandled exception that kills the whole loop.
                observation = f"Error: {e}"

        print(f"  [Tool Result] {observation}")

        # Append what the model said *and* what actually happened, so the
        # next iteration's prompt continues right where this one left off.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Starting the agent loop...(raw ReAct prompting)")
    print()

    demo_queries: list[str] = [
        # Happy path: both tools get called in the required order (price
        # first, then discount), exercising STRICT RULES 1-3.
        "What is the price of a laptop after applying a gold discount?",
        # No tier specified: exercises STRICT RULE 4 — the model should ask
        # which discount tier to use rather than guessing one.
        "What is the price of a smartphone with a discount applied?",
    ]

    for i, demo_query in enumerate(demo_queries, start=1):
        print(f"\n{'=' * 70}\nDemo {i}/{len(demo_queries)}: {demo_query}\n{'=' * 70}")
        run_agent_loop(demo_query)
