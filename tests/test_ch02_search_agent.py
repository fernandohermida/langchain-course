import pytest
from langchain_openai import ChatOpenAI

from examples.ch02_search_agent import (
    QUERY,
    AgentResponse,
    Source,
    build_search_agent,
    run_query,
)


class _StubAgent:
    def __init__(self, response: AgentResponse):
        self._response = response

    def invoke(self, _payload):
        return {"structured_response": self._response}


def test_run_query_extracts_structured_response():
    expected = AgentResponse(
        answer="Test answer", sources=[Source(url="https://example.com")]
    )

    result = run_query(_StubAgent(expected), "any query")

    assert result == expected


@pytest.mark.integration
def test_search_agent_with_real_apis():
    agent = build_search_agent(ChatOpenAI(model="gpt-5"))

    result = run_query(agent, QUERY)

    assert isinstance(result, AgentResponse)
