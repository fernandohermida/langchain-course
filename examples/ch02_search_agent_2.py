import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()
QUERY = "What's the weather in Tokyo now?"


@tool
def search(query: str) -> str:
    """Tool that search over the internet.

    Args:
        query: The search to search.

    Returns:
        The search result.
    """
    print(query)
    return json.dumps(tavily.search(query=query))


def build_search_agent(llm: BaseChatModel):
    tools = [search]
    return create_agent(model=llm, tools=tools)


def run_query(agent, query: str):
    return agent.invoke(
        {"messages": HumanMessage(content=query)},
        config={"run_name": "ch02_search_agent_2", "tags": ["ch02"]},
    )


def main() -> None:
    agent = build_search_agent(ChatOpenAI(model="gpt-5", temperature=0))
    result = run_query(agent, QUERY)
    print(result)


if __name__ == "__main__":
    main()
