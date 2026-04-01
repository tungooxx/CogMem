import pytest
from unittest.mock import MagicMock, patch
from cogmem.utils.llm_client import LLMClient


def test_init_ollama():
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    assert client.api_base == "http://localhost:11434/v1"


def test_init_groq():
    client = LLMClient(provider="groq", model="llama-3.1-8b-instant", api_key="test")
    assert client.api_base == "https://api.groq.com/openai/v1"


def test_init_together():
    client = LLMClient(provider="together", model="meta-llama/Llama-3.2-3B-Instruct", api_key="test")
    assert client.api_base == "https://api.together.xyz/v1"


@patch("cogmem.utils.llm_client.OpenAI")
def test_generate_calls_openai_client(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello world"))]
    )
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    result = client.generate("say hello")
    assert result == "Hello world"
    mock_client.chat.completions.create.assert_called_once()


@patch("cogmem.utils.llm_client.OpenAI")
def test_generate_retries_on_rate_limit(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    from openai import RateLimitError
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    rate_err = RateLimitError(
        message="rate limit",
        response=mock_resp,
        body=None,
    )
    mock_client.chat.completions.create.side_effect = [
        rate_err,
        MagicMock(choices=[MagicMock(message=MagicMock(content="OK"))]),
    ]
    client = LLMClient(provider="groq", model="test", api_key="k")
    client._retry_base_delay = 0.01  # fast retry for test
    result = client.generate("test")
    assert result == "OK"
    assert mock_client.chat.completions.create.call_count == 2
