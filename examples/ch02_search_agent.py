from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's structured response"""

    answer: str = Field(description="The agent's answer to the query")
    sources: list[Source] = Field(
        default_factory=list, description="Sources used to generate the answer"
    )


QUERY = (
    "search for 3 job postings for an ai engineer using langchain in the bay "
    "area on linkedin and list their details?"
)


def build_search_agent(llm: BaseChatModel):
    return create_agent(
        model=llm, tools=[TavilySearch()], response_format=AgentResponse
    )


def run_query(agent, query: str) -> AgentResponse:
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["structured_response"]


def main() -> None:
    agent = build_search_agent(ChatOpenAI(model="gpt-5"))
    print(run_query(agent, QUERY))


if __name__ == "__main__":
    main()
