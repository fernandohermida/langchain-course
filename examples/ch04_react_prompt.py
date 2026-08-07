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


# Valid tiers live in the type hint (not just the docstring) purely for the
# reader — this file has no tools= schema, so nothing here actually stops
# the model from inventing a made-up tier. apply_discount's own tier check
# below is what turns an invalid tier into a visible Observation.
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
        # tier becomes an Observation the model can react to, instead of a
        # raw KeyError traceback that would kill the whole agent loop.
        raise ValueError(
            f"Unknown discount tier '{discount_tier}'. "
            f"Available tiers: {', '.join(discount_percentages)}"
        )

    return round(price * (1 - discount_percentages[discount_tier]), 2)


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
            return None

        print(f"  [Tool Selected] {action} with args: {action_input}")

        # Action Input is "comma separated values" per the prompt's format
        # instructions.
        args = [arg.strip() for arg in action_input.split(",")]

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
    # A model this small doesn't always follow every STRICT RULE — it may
    # answer after get_product_price without ever calling apply_discount.
    # That's a real limitation of prompting a 1.7B model, not a bug in this
    # loop.
    run_agent_loop("What is the price of a laptop after applying a gold discount?")
