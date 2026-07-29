import pytest
from langchain_openai import ChatOpenAI

from examples.ch02_search_job_agent import (
    QUERY,
    JobPosting,
    JobSearchResponse,
    build_search_agent,
    run_query,
)


class _StubAgent:
    def __init__(self, response: JobSearchResponse):
        self._response = response

    def invoke(self, _payload, config=None):
        return {"structured_response": self._response}


def test_run_query_extracts_structured_response():
    expected = JobSearchResponse(
        jobs=[
            JobPosting(
                title="Senior .NET Engineer",
                company="Acme Corp",
                location="Remote",
                url="https://example.com/job/1",
            )
        ]
    )

    result = run_query(_StubAgent(expected), "any query")

    assert result == expected


@pytest.mark.integration
def test_search_job_agent_with_real_apis():
    agent = build_search_agent(ChatOpenAI(model="gpt-5-mini", temperature=0))

    result = run_query(agent, QUERY)

    assert isinstance(result, JobSearchResponse)
