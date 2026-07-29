from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


QUERY = "What's the weather in Tokyo?"


@tool
def search(query: str) -> str:
    """Search the internet for the given query.

    Args:
        query: The search query.

    Returns:
        The search result.
    """
    print(query)
    return "Tokyo weather is sunny right now."


def build_search_agent(llm: BaseChatModel):
    tools = [search]
    return create_agent(model=llm, tools=tools)


def run_query(agent, query: str):
    return agent.invoke(
        {"messages": HumanMessage(content=query)},
        config={"run_name": "ch02_search_agent", "tags": ["ch02"]},
    )


def main() -> None:
    agent = build_search_agent(ChatOpenAI(model="gpt-5"))
    result = run_query(agent, QUERY)
    print(result)


if __name__ == "__main__":
    main()
