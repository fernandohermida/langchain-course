import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from examples.ch01_hello_chain import ChatOllama, summarize_person


def test_summarize_person_invokes_chain_and_returns_content():
    fake_llm = FakeListChatModel(responses=["1. Summary. 2. Fact one. Fact two."])

    result = summarize_person(fake_llm, information="Some short bio text.")

    assert result == "1. Summary. 2. Fact one. Fact two."


@pytest.mark.integration
def test_summarize_person_with_real_ollama():
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    result = summarize_person(llm, information="Ada Lovelace was a mathematician.")

    assert isinstance(result, str)
    assert len(result) > 0
