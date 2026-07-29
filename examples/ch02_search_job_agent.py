from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class JobPosting(BaseModel):
    """Schema for a single job posting found by the agent"""

    title: str = Field(description="The job title")
    company: str = Field(description="The hiring company")
    location: str = Field(description="The job location, e.g. city/country or 'Remote'")
    url: str = Field(description="URL to the job posting")


class JobSearchResponse(BaseModel):
    """Schema for the agent's structured response"""

    jobs: list[JobPosting] = Field(
        default_factory=list, description="Job postings matching the query"
    )


QUERY = "search 3 jobs in europe for a senior software engineer in .NET , remote"


def build_search_agent(llm: BaseChatModel):
    return create_agent(
        model=llm, tools=[TavilySearch()], response_format=JobSearchResponse
    )


def run_query(agent, query: str) -> JobSearchResponse:
    result = agent.invoke(
        {"messages": HumanMessage(content=query)},
        config={"run_name": "ch02_search_job_agent", "tags": ["ch02"]},
    )
    return result["structured_response"]


def main() -> None:
    agent = build_search_agent(ChatOpenAI(model="gpt-5-mini", temperature=0))
    result = run_query(agent, QUERY)
    print(result)


if __name__ == "__main__":
    main()
