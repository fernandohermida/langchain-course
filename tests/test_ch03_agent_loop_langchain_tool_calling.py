import pytest
from langchain_core.messages import AIMessage

from examples import ch03_agent_loop_langchain_tool_calling as ch03


class _FakeToolCallingModel:
    def __init__(self, responses):
        self._responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self._responses)


def test_run_agent_loop_executes_tool_then_returns_final_answer(monkeypatch):
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_product_price",
                    "args": {"product": "laptop"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="The laptop costs $999.99."),
    ]
    monkeypatch.setattr(
        ch03,
        "init_chat_model",
        lambda *_args, **_kwargs: _FakeToolCallingModel(responses),
    )

    result = ch03.run_agent_loop("How much is a laptop?")

    assert result == "The laptop costs $999.99."


@pytest.mark.integration
def test_run_agent_loop_with_real_ollama():
    result = ch03.run_agent_loop("I want to buy a laptop and apply a gold discount.")

    assert isinstance(result, str)
    assert len(result) > 0
