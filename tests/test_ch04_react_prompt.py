from types import SimpleNamespace

from examples import ch04_react_prompt as ch04


def _chat_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_parse_model_output_handles_action_and_input_on_one_line():
    text = 'Thought: I need the price.\nAction: get_product_price, product="smartphone"'

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "get_product_price"
    assert action_input == 'product="smartphone"'
    assert final_answer is None


def test_parse_model_output_strips_call_syntax_when_action_input_line_present():
    text = (
        "Thought: I need the price.\n"
        'Action: get_product_price(product="smartphone")\n'
        'Action Input: product="smartphone"'
    )

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "get_product_price"
    assert action_input == 'product="smartphone"'
    assert final_answer is None


def test_parse_model_output_handles_call_syntax_without_action_input_line():
    text = 'Thought: ...\nAction: get_product_price(product="smartphone")'

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "get_product_price"
    assert action_input == 'product="smartphone"'


def test_parse_model_output_handles_multi_arg_call_syntax_without_action_input_line():
    text = 'Action: apply_discount(price=699.99, discount_tier="gold")'

    action, action_input, final_answer = ch04.parse_model_output(text)

    assert action == "apply_discount"
    assert action_input == 'price=699.99, discount_tier="gold"'


def test_run_agent_loop_recovers_when_action_and_input_are_on_one_line(monkeypatch):
    responses = iter(
        [
            _chat_response(
                "Thought: I need the price.\n"
                'Action: get_product_price, product="smartphone"'
            ),
            _chat_response(
                "Thought: I now know the final answer\n"
                "Final Answer: The smartphone costs $699.99."
            ),
        ]
    )
    monkeypatch.setattr(ch04.ollama, "chat", lambda **_kwargs: next(responses))

    result = ch04.run_agent_loop("How much is a smartphone?")

    assert result == "The smartphone costs $699.99."


def test_run_agent_loop_recovers_when_action_line_has_call_syntax(monkeypatch):
    responses = iter(
        [
            _chat_response(
                "Thought: I need the price.\n"
                'Action: get_product_price(product="smartphone")\n'
                'Action Input: product="smartphone"'
            ),
            _chat_response(
                "Thought: I now know the final answer\n"
                "Final Answer: The smartphone costs $699.99."
            ),
        ]
    )
    monkeypatch.setattr(ch04.ollama, "chat", lambda **_kwargs: next(responses))

    result = ch04.run_agent_loop("How much is a smartphone?")

    assert result == "The smartphone costs $699.99."


def test_run_agent_loop_stops_early_on_repeated_identical_failure(monkeypatch):
    call_count = {"n": 0}

    def fake_chat(**_kwargs):
        call_count["n"] += 1
        return _chat_response(
            "Thought: I need the price.\n"
            "Action: get_product_prise\n"
            'Action Input: product="smartphone"'
        )

    monkeypatch.setattr(ch04.ollama, "chat", fake_chat)

    result = ch04.run_agent_loop("How much is a smartphone?")

    assert result is None
    assert call_count["n"] == 2


def test_call_model_disables_thinking_and_caps_output_length(monkeypatch):
    captured_kwargs = {}

    def fake_chat(**kwargs):
        captured_kwargs.update(kwargs)
        return _chat_response("Final Answer: done")

    monkeypatch.setattr(ch04.ollama, "chat", fake_chat)

    ch04.call_model("some prompt")

    assert captured_kwargs["think"] is False
    assert captured_kwargs["options"]["num_predict"] > 0
