from types import SimpleNamespace

from examples import ch04_react_prompt as ch04


def _chat_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_parse_model_output_happy_path():
    text = "Thought: I need the price.\nAction: get_product_price\nAction Input: laptop"

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "get_product_price"
    assert action_input == "laptop"
    assert final_answer is None


def test_parse_model_output_returns_none_when_action_input_line_missing():
    text = 'Thought: I need the price.\nAction: get_product_price, product="smartphone"'

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action is None
    assert action_input is None
    assert final_answer is None


def test_run_agent_loop_executes_both_tools_then_returns_final_answer(monkeypatch):
    responses = iter(
        [
            _chat_response(
                "Thought: I need the laptop's price first.\n"
                "Action: get_product_price\n"
                "Action Input: laptop"
            ),
            _chat_response(
                "Thought: Now I'll apply the gold discount.\n"
                "Action: apply_discount\n"
                "Action Input: 999.99, gold"
            ),
            _chat_response(
                "Thought: I now know the final answer\n"
                "Final Answer: The laptop costs $849.99 after the gold discount."
            ),
        ]
    )
    monkeypatch.setattr(ch04.ollama, "chat", lambda **_kwargs: next(responses))

    result = ch04.run_agent_loop(
        "What is the price of a laptop after applying a gold discount?"
    )

    assert result == "The laptop costs $849.99 after the gold discount."


def test_call_model_disables_thinking_and_caps_output_length(monkeypatch):
    captured_kwargs = {}

    def fake_chat(**kwargs):
        captured_kwargs.update(kwargs)
        return _chat_response("Final Answer: done")

    monkeypatch.setattr(ch04.ollama, "chat", fake_chat)

    ch04.call_model("some prompt")

    assert captured_kwargs["think"] is False
    assert captured_kwargs["options"]["num_predict"] > 0
