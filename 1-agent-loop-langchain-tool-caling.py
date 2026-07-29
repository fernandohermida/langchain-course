from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import traceable

load_dotenv()


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog"""
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
        # loop's try/except -> "Error: ..." ToolMessage path, so the model sees
        # the failure and can react instead of getting a silently made-up number.
        raise ValueError(
            f"Unknown product '{product}'. Available products: {', '.join(prices)}"
        )
    return prices[normalized]


# Valid tiers live in the type hint (not just the docstring) so bind_tools()
# puts them in the tool's JSON schema as an enum — the model is structurally
# discouraged from inventing a tier before the tool ever runs.
DiscountTier = Literal["bronze", "silver", "gold", "platinum", "diamond"]


@tool
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


@traceable(name="agent_loop")
def run_agent_loop(query: str) -> str | None:
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # bind_tools() only announces the tool schemas (name/docstring/type hints) to
    # the model — it does not execute anything. The model can only ever *request*
    # a call; running it and feeding the result back is entirely up to this loop.
    llm = init_chat_model(MODEL, model_provider="ollama", temperature=0).bind_tools(
        tools
    )
    print(f"Starting agent loop with query: {query}")

    message = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=query),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration} ")
        ai_message = llm.invoke(message)
        tool_calls = ai_message.tool_calls

        # The loop's own termination condition: the model decides it's done by
        # returning zero tool calls. There's no separate "final answer" signal.
        if not tool_calls:
            # AIMessage.content is typed str | list[...] to support multimodal
            # content blocks across providers; this text-only chat model always
            # returns a plain string, so narrow explicitly to match -> str | None.
            final_answer = ai_message.content
            if not isinstance(final_answer, str):
                final_answer = str(final_answer)
            print(f"\nFinal Answer: {final_answer}")
            return final_answer

        # The AI's tool-call request must go into history *before* the tool
        # results: every ToolMessage.tool_call_id has to reference a tool_call
        # that already exists in a prior assistant message.
        message.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                observation = f"Error: tool '{tool_name}' not found"
            else:
                try:
                    observation = tool_to_use.invoke(tool_args)
                except Exception as e:
                    observation = f"Error: {e}"

            print(f"  [Tool Result] {observation}")
            message.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )

    print("ERROR: Max iterations reached without a final answer")
    return None  # caller must check for None: this is a bail-out, not a real answer


if __name__ == "__main__":
    print("Starting the agent loop...(.bind_tools)")
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
