import pytest
from langchain_openai import ChatOpenAI

from examples.ch02_search_agent import QUERY, build_search_agent, run_query


class _StubAgent:
    def __init__(self, response: dict):
        self._response = response

    def invoke(self, _payload, config=None):
        return self._response


def test_run_query_returns_agent_invoke_result():
    expected = {"messages": ["Tokyo weather is sunny right now."]}

    result = run_query(_StubAgent(expected), "any query")

    assert result == expected


@pytest.mark.integration
def test_search_agent_with_real_apis():
    agent = build_search_agent(ChatOpenAI(model="gpt-5"))

    result = run_query(agent, QUERY)

    assert "messages" in result
