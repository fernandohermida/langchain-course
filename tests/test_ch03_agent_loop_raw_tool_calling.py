from types import SimpleNamespace

import pytest

from examples import ch03_agent_loop_raw_tool_calling as ch03


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _chat_response(*, content="", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
    )


def test_run_agent_loop_executes_tool_then_returns_final_answer(monkeypatch):
    responses = iter(
        [
            _chat_response(
                tool_calls=[_tool_call("get_product_price", {"product": "laptop"})]
            ),
            _chat_response(content="The laptop costs $999.99."),
        ]
    )
    monkeypatch.setattr(ch03.ollama, "chat", lambda **_kwargs: next(responses))

    result = ch03.run_agent_loop("How much is a laptop?")

    assert result == "The laptop costs $999.99."


@pytest.mark.integration
def test_run_agent_loop_with_real_ollama():
    result = ch03.run_agent_loop("I want to buy a laptop and apply a gold discount.")

    assert isinstance(result, str)
    assert len(result) > 0
