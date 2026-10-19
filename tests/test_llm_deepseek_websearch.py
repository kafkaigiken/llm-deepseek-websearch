import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx2
import llm
import pytest

import llm_deepseek_websearch
from llm_deepseek_websearch import (
    DeepSeekError,
    deepseek_web_fetch,
    deepseek_web_search,
)

SEARCH_RESULT = {
    "rerank_score": 2.25,
    "title": "DeepSeek API Docs",
    "url": "https://api-docs.deepseek.com/",
    "snippet": "Short snippet",
    "summary": "Longer summary for the model",
    "content": "",
    "site_name": "DeepSeek",
    "site_icon": None,
    "date": "2026-10-01T00:00:00Z",
    "authority": 3,
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Empty llm key store and no DeepSeek environment."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path / "llm"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.fixture
def sleeps(monkeypatch):
    """Record the retry waits instead of sleeping them."""
    recorded = []
    monkeypatch.setattr(llm_deepseek_websearch.time, "sleep", recorded.append)
    return recorded


def mock_api(monkeypatch, handler):
    """Route every request to handler and record it."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    requests = []

    def recording_handler(request):
        requests.append(request)
        return handler(request)

    real_client = llm_deepseek_websearch._client
    monkeypatch.setattr(
        llm_deepseek_websearch,
        "_client",
        lambda api_key: real_client(
            api_key, transport=httpx2.MockTransport(recording_handler)
        ),
    )
    return requests


def reply(status_code=200, **response_kwargs):
    """A handler that always returns the same response."""
    return lambda request: httpx2.Response(status_code, **response_kwargs)


def test_tools_are_registered():
    tool_names = set(llm.get_tools())
    assert {"deepseek_web_search", "deepseek_web_fetch"} <= tool_names


def test_search_request_and_output(monkeypatch):
    requests = mock_api(
        monkeypatch,
        reply(200, json={"results": [SEARCH_RESULT]}),
    )
    output = deepseek_web_search("deepseek api", question="What is it?")

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "query": "deepseek api",
        "count": 5,
        "need_content": False,
        "ori_question": "What is it?",
    }
    assert output == (
        "1. DeepSeek API Docs\n"
        "   URL: https://api-docs.deepseek.com/\n"
        "   Site: DeepSeek | Date: 2026-10-01T00:00:00Z"
        " | Relevance: 2.25/3 | Authority: 3/4\n"
        "   Summary: Longer summary for the model"
    )


def test_search_clamps_count_and_question(monkeypatch):
    requests = mock_api(monkeypatch, reply(200, json={"results": []}))
    deepseek_web_search("q", count=500, question="x" * 300)
    body = json.loads(requests[0].content)
    assert body["count"] == 50
    assert len(body["ori_question"]) == 128

    deepseek_web_search("q", count=0, question="   ")
    body = json.loads(requests[1].content)
    assert body["count"] == 1
    assert "ori_question" not in body


def test_search_no_results(monkeypatch):
    mock_api(monkeypatch, reply(200, json={"results": []}))
    assert deepseek_web_search("nothing") == "No results found"


def test_search_content_is_truncated(monkeypatch):
    long_result = dict(SEARCH_RESULT, content="a" * 5000, authority=-1)
    mock_api(
        monkeypatch,
        reply(200, json={"results": [long_result]}),
    )
    output = deepseek_web_search("q", need_content=True)
    assert "a" * 3000 + " [truncated, use deepseek_web_fetch]" in output
    assert "a" * 3001 not in output
    assert "Authority" not in output


def test_fetch_request_and_output(monkeypatch):
    requests = mock_api(
        monkeypatch,
        reply(200, json={"title": "Example", "content": "# Hello\n\nWorld"}),
    )
    output = deepseek_web_fetch("https://example.com/")
    assert str(requests[0].url) == "https://api.deepseek.com/v1/tools/fetch"
    assert json.loads(requests[0].content) == {"url": "https://example.com/"}
    expected = "Title: Example\nURL: https://example.com/\n\n"
    assert output == expected + "# Hello\n\nWorld"


def test_fetch_empty_content(monkeypatch):
    mock_api(
        monkeypatch,
        reply(200, json={"title": "", "content": ""}),
    )
    output = deepseek_web_fetch("https://example.com/")
    assert output == "No content could be fetched from https://example.com/"


def test_missing_key():
    with pytest.raises(llm.NeedsKeyException, match="DEEPSEEK_API_KEY"):
        deepseek_web_search("q")


def test_stored_key_is_used(monkeypatch, tmp_path):
    key_file = tmp_path / "llm" / "keys.json"
    key_file.parent.mkdir(parents=True)
    key_file.write_text(json.dumps({"deepseek": "sk-stored"}))
    requests = mock_api(monkeypatch, reply(200, json={"results": []}))
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    deepseek_web_search("q")
    assert requests[0].headers["Authorization"] == "Bearer sk-stored"


def test_client_error_is_not_retried(monkeypatch, sleeps):
    requests = mock_api(
        monkeypatch,
        reply(
            404,
            json={"error": {"message": "No usable content", "code": "x"}},
        ),
    )
    with pytest.raises(DeepSeekError, match="404: No usable content"):
        deepseek_web_fetch("https://example.com/missing")
    assert len(requests) == 1
    assert sleeps == []


def test_rate_limit_is_retried(monkeypatch, sleeps):
    responses = [
        reply(429, headers={"Retry-After": "3"}, text="slow down"),
        reply(503, text="busy"),
        reply(json={"results": [SEARCH_RESULT]}),
    ]
    requests = mock_api(monkeypatch, lambda request: responses.pop(0)(request))
    output = deepseek_web_search("q")
    assert output.startswith("1. DeepSeek API Docs")
    assert len(requests) == 3
    assert sleeps == [3.0, 2]


def test_retries_give_up(monkeypatch, sleeps):
    requests = mock_api(monkeypatch, reply(500, text="boom"))
    with pytest.raises(DeepSeekError, match="500: boom"):
        deepseek_web_search("q")
    assert len(requests) == 3


def test_transport_error(monkeypatch, sleeps):
    def handler(request):
        raise httpx2.ConnectError("no network", request=request)

    mock_api(monkeypatch, handler)
    with pytest.raises(DeepSeekError, match="request failed: no network"):
        deepseek_web_fetch("https://example.com/")
    assert sleeps == [1, 2]


def test_retry_after_http_date_is_honoured(monkeypatch, sleeps):
    # A date far in the future is parsed and capped, which is independent of
    # how long the test itself takes.
    when = datetime.now(timezone.utc) + timedelta(seconds=120)
    header = format_datetime(when, usegmt=True)
    responses = [
        reply(429, headers={"Retry-After": header}, text="slow down"),
        reply(json={"results": []}),
    ]
    mock_api(monkeypatch, lambda request: responses.pop(0)(request))
    deepseek_web_search("q")
    assert sleeps == [10.0]


@pytest.mark.parametrize("header", ["nan", "inf", "-5", "garbage"])
def test_unusable_retry_after_falls_back(monkeypatch, sleeps, header):
    responses = [
        reply(429, headers={"Retry-After": header}, text="slow down"),
        reply(json={"results": []}),
    ]
    mock_api(monkeypatch, lambda request: responses.pop(0)(request))
    deepseek_web_search("q")
    assert sleeps == [1]


@pytest.mark.parametrize("body", ["a string", [1, 2], None, 3])
def test_non_object_json_body_is_a_tool_error(monkeypatch, body):
    mock_api(monkeypatch, lambda request: httpx2.Response(200, text=json.dumps(body)))
    with pytest.raises(DeepSeekError, match="unexpected JSON"):
        deepseek_web_fetch("https://example.com/")


def test_non_object_results_are_skipped(monkeypatch):
    mock_api(monkeypatch, reply(json={"results": ["junk", SEARCH_RESULT, None]}))
    output = deepseek_web_search("q")
    assert output.startswith("1. DeepSeek API Docs")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/",
        "file:///etc/passwd",
        "https://user:pass@example.com/",
        "https:///nohost",
        "",
        "   ",
    ],
)
def test_bad_url_is_a_tool_error(monkeypatch, url):
    requests = mock_api(monkeypatch, reply(json={"content": "x"}))
    with pytest.raises(DeepSeekError):
        deepseek_web_fetch(url)
    assert requests == []


def test_too_long_url_is_a_tool_error(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"content": "x"}))
    with pytest.raises(DeepSeekError, match="longer than 8192"):
        deepseek_web_fetch("https://example.com/" + "a" * 8192)
    assert requests == []


def test_url_is_stripped_before_it_is_sent(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"content": "x"}))
    deepseek_web_fetch("  https://example.com/  ")
    assert json.loads(requests[0].content) == {"url": "https://example.com/"}


def test_bad_count_is_a_tool_error(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    with pytest.raises(DeepSeekError, match="whole number"):
        deepseek_web_search("q", count="many")
    assert requests == []


def test_null_url_and_score_do_not_crash(monkeypatch):
    mock_api(
        monkeypatch,
        reply(json={"results": [dict(SEARCH_RESULT, url=None, rerank_score="high")]}),
    )
    output = deepseek_web_search("q")
    assert "URL: \n" in output
    assert "Relevance" not in output


@pytest.mark.parametrize("status", [501, 505, 507, 511, 599])
def test_every_5xx_is_retried(monkeypatch, sleeps, status):
    requests = mock_api(monkeypatch, reply(status, text="boom"))
    with pytest.raises(DeepSeekError, match=str(status)):
        deepseek_web_search("q")
    assert len(requests) == 3
    assert sleeps == [1, 2]


@pytest.mark.parametrize("query", [None, 42, ["q"], "", "   ", "x" * 129])
def test_bad_query_is_a_tool_error(monkeypatch, query):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    with pytest.raises(DeepSeekError):
        deepseek_web_search(query)
    assert requests == []


def test_query_is_stripped_before_it_is_sent(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    deepseek_web_search("  deepseek api  ")
    assert json.loads(requests[0].content)["query"] == "deepseek api"


def test_non_string_question_is_a_tool_error(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    with pytest.raises(DeepSeekError, match="question must be a string"):
        deepseek_web_search("q", question=42)
    assert requests == []


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), (False, False), (None, False), (0, False), (1, True),
     ("false", False), ("True", True), ("no", False), ("off", False), ("", False)],
)
def test_need_content_accepts_plain_spellings(monkeypatch, value, expected):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    deepseek_web_search("q", need_content=value)
    assert json.loads(requests[0].content)["need_content"] is expected


def test_null_question_is_treated_as_absent(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    deepseek_web_search("q", question=None)
    assert "ori_question" not in json.loads(requests[0].content)


def test_bad_need_content_is_a_tool_error(monkeypatch):
    requests = mock_api(monkeypatch, reply(json={"results": []}))
    with pytest.raises(DeepSeekError, match="need_content must be true or false"):
        deepseek_web_search("q", need_content="sometimes")
    assert requests == []


def test_redirect_is_an_error(monkeypatch):
    requests = mock_api(
        monkeypatch,
        reply(302, text="", headers={"Location": "https://api.deepseek.com/v2/"}),
    )
    with pytest.raises(
        DeepSeekError, match="redirected to https://api.deepseek.com/v2/"
    ):
        deepseek_web_fetch("https://example.com/")
    assert len(requests) == 1


def test_redirect_body_is_not_returned_as_content(monkeypatch):
    mock_api(
        monkeypatch,
        reply(307, json={"title": "t", "content": "redirect body"}),
    )
    with pytest.raises(DeepSeekError, match="redirected"):
        deepseek_web_fetch("https://example.com/")


def test_error_body_is_capped(monkeypatch, sleeps):
    html = "<html>" + "x" * 200_000 + "</html>"
    mock_api(monkeypatch, reply(502, text=html))
    with pytest.raises(DeepSeekError) as excinfo:
        deepseek_web_search("q")
    message = str(excinfo.value)
    assert "[...]" in message
    assert len(message) < llm_deepseek_websearch.MAX_ERROR_CHARS + 100


def test_corrupt_gzip_body_is_retried_then_reported(monkeypatch, sleeps):
    mock_api(
        monkeypatch,
        lambda request: httpx2.Response(
            200, content=b"not gzip", headers={"content-encoding": "gzip"}
        ),
    )
    with pytest.raises(DeepSeekError, match="request failed"):
        deepseek_web_search("q")
    assert sleeps == [1, 2]


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [SEARCH_RESULT]},
        {"results": {"items": [SEARCH_RESULT]}},
        {"results": "junk"},
    ],
)
def test_unexpected_search_payload_is_an_error(monkeypatch, payload):
    mock_api(monkeypatch, reply(json=payload))
    with pytest.raises(DeepSeekError, match="API may have changed"):
        deepseek_web_search("q")


def test_unexpected_fetch_content_is_an_error(monkeypatch):
    mock_api(monkeypatch, reply(json={"title": "t", "content": 12345}))
    with pytest.raises(DeepSeekError, match="expected text"):
        deepseek_web_fetch("https://example.com/")
