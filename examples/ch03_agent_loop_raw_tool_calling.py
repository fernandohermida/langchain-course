from typing import Literal, get_args

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

    discount_percentages = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
        "platinum": 0.20,
        "diamond": 0.25,
    }

    return round(price * (1 - discount_percentages[discount_tier]), 2)


# ollama's chat() can auto-derive a tool schema from a plain function's
# signature/docstring, but that derivation drops enum constraints (Literal
# types collapse to a bare "string") and, if the function is wrapped by
# something like langsmith's @traceable, picks up the wrapper's injected
# kwargs as bogus required parameters. Authoring the JSON schema by hand
# sidesteps both problems and is what "raw" tool calling means anyway: the
# model only ever sees this schema, never the Python function itself.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Lookup the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": 'Name of the product to look up, e.g. "laptop".',
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to the price of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The exact price returned by get_product_price.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "enum": list(get_args(DiscountTier)),
                        "description": "Loyalty tier to apply the discount for.",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# A dedicated run_type="llm" span so each model call shows up in LangSmith
# as its own traced step (inputs/output/token usage) nested under the agent
# loop's trace, instead of being invisible work inside a "chain" run.
@traceable(
    run_type="llm",
    name="ollama.chat",
    metadata={"ls_provider": "ollama", "ls_model_name": MODEL},
)
def call_model(messages: list[dict | ollama.Message]) -> ollama.ChatResponse:
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        options={"temperature": 0},
    )


@traceable(name="ch03_agent_loop_raw_tool_calling")
def run_agent_loop(query: str) -> str | None:
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    print(f"Starting agent loop with query: {query}")

    messages: list[dict | ollama.Message] = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": query},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration} ")
        # Passing tools=TOOLS only announces the tool schemas to the model —
        # it does not execute anything. The model can only ever *request* a
        # call; running it and feeding the result back is entirely up to
        # this loop.
        response = call_model(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # The loop's own termination condition: the model decides it's done by
        # returning zero tool calls. There's no separate "final answer" signal.
        if not tool_calls:
            final_answer = ai_message.content or ""
            print(f"\nFinal Answer: {final_answer}")
            return final_answer

        # The AI's tool-call request must go into history before the tool
        # results: unlike OpenAI-style APIs, ollama's tool messages carry no
        # call id — the model matches each result back to a request purely by
        # the order the messages appear in, so that order must be preserved.
        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                observation = f"Error: tool '{tool_name}' not found"
            else:
                try:
                    observation = tool_to_use(**tool_args)
                except Exception as e:
                    observation = f"Error: {e}"

            print(f"  [Tool Result] {observation}")
            messages.append(
                {"role": "tool", "content": str(observation), "tool_name": tool_name}
            )

    print("ERROR: Max iterations reached without a final answer")
    return None  # caller must check for None: this is a bail-out, not a real answer


if __name__ == "__main__":
    print("Starting the agent loop...(raw ollama tool calling)")
    print()

    # 1: happy path
    # 2: unknown product -> tool raises -> error fed back to model
    # 3: invalid tier -> schema/tool rejects it -> error fed back to model
    demo_queries = [
        "I want to buy a laptop and apply a gold discount.",
        "How much is a smartwatch with a platinum discount?",
        "I want a tablet with a legendary discount.",
    ]

    for i, demo_query in enumerate(demo_queries, start=1):
        print(f"\n{'=' * 70}\nDemo {i}/{len(demo_queries)}: {demo_query}\n{'=' * 70}")
        run_agent_loop(demo_query)
