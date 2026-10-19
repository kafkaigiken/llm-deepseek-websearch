"""llm plugin: search and fetch the web with the DeepSeek Search and Fetch
APIs.

Docs: https://api-docs.deepseek.com/guides/search/

Two tools:
  - `deepseek_web_search`: POST {API_ROOT}search
        body={"query": ..., "count": ..., "need_content": ...}
  - `deepseek_web_fetch`: POST {API_ROOT}fetch
        body={"url": ...}

The API key comes from the DEEPSEEK_API_KEY environment variable or from
`llm keys set deepseek`, the same key that the llm-deepseek model plugin uses.

Rate limits are 5 requests per second per account for each API. The plugin
retries 408, 429 and 5xx responses and network failures, as the docs advise,
and fails at once on other errors.
"""

import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx2
import llm

API_ROOT = "https://api.deepseek.com/v1/tools/"
ENV_KEY = "DEEPSEEK_API_KEY"
KEY_ALIAS = "deepseek"
HTTP_TIMEOUT = 60  # seconds
MAX_ATTEMPTS = 3
RETRY_STATUSES = {408, 429}  # plus every 5xx, see _is_retryable()
MAX_RETRY_WAIT = 10  # seconds
DEFAULT_COUNT = 5
MAX_COUNT = 50
MAX_QUERY_LENGTH = 128  # the API rejects a longer query
MAX_QUESTION_LENGTH = 128  # the API rejects a longer ori_question
MAX_URL_LENGTH = 8192  # the API rejects a longer url
MAX_ERROR_CHARS = 1000  # cap an error body before it reaches the model
# Search results with need_content=True can be whole pages. Cap each one so
# that a 50-result search does not fill the context window. Fetch output is
# never cut.
MAX_SEARCH_CONTENT_CHARS = 3000


class DeepSeekError(Exception):
    pass


@llm.hookimpl
def register_tools(register):
    register(deepseek_web_search)
    register(deepseek_web_fetch)


def _api_key():
    api_key = llm.get_key(alias=KEY_ALIAS, env=ENV_KEY)
    if not api_key:
        # Same wording llm uses for a missing model key.
        raise llm.NeedsKeyException(
            f"No key found - add one using 'llm keys set {KEY_ALIAS}' or set the "
            f"{ENV_KEY} environment variable"
        )
    return api_key


def _client(api_key, transport=None):
    return httpx2.Client(
        base_url=API_ROOT,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=HTTP_TIMEOUT,
        transport=transport,
    )


def _is_retryable(status_code):
    """True for the statuses the API docs say to retry: 408, 429 and any 5xx."""
    return status_code in RETRY_STATUSES or 500 <= status_code <= 599


def _error_message(response):
    """Return a short, bounded description of an error response."""
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        # Not an OpenAI-style body, for example a proxy's HTML error page.
        message = response.text.strip() or response.reason_phrase
    if not isinstance(message, str):
        message = str(message)
    message = message.strip() or response.reason_phrase
    if len(message) > MAX_ERROR_CHARS:
        message = message[:MAX_ERROR_CHARS] + " [...]"
    return message


def _retry_after(response):
    """Return the seconds in a Retry-After header, or None when unusable.

    The header is either a number of seconds or an HTTP date.
    """
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_WAIT)


def _retry_wait(attempt, response=None):
    """Seconds to wait before retrying, honouring Retry-After when it is usable."""
    wait_seconds = None if response is None else _retry_after(response)
    if wait_seconds is None:
        wait_seconds = 2 ** (attempt - 1)
    return min(max(wait_seconds, 0), MAX_RETRY_WAIT)


def _validate_url(url):
    """Return url stripped, after checking the scheme, the length and credentials."""
    if not isinstance(url, str):
        raise DeepSeekError(f"url must be a string, got {type(url).__name__}")
    url = url.strip()
    if not url:
        raise DeepSeekError("url must not be empty")
    if len(url) > MAX_URL_LENGTH:
        raise DeepSeekError(f"url is longer than {MAX_URL_LENGTH} characters")
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError as error:
        raise DeepSeekError(f"url is not valid: {error}") from error
    if parts.scheme not in ("http", "https"):
        raise DeepSeekError(f"url must start with http:// or https://, got {url!r}")
    if not hostname:
        raise DeepSeekError(f"url has no host: {url!r}")
    if parts.username or parts.password:
        raise DeepSeekError("url must not contain a user name or password")
    return url


def _as_bool(value, name):
    """Return value as a bool, accepting the usual spellings of true and false."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    raise DeepSeekError(f"{name} must be true or false, got {value!r}")


def _post(endpoint, body):
    """POST body to the endpoint and return the decoded JSON object."""
    label = endpoint.capitalize()
    with _client(_api_key()) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.post(endpoint, json=body)
            except (httpx2.RequestError, httpx2.StreamError) as error:
                if attempt == MAX_ATTEMPTS:
                    raise DeepSeekError(
                        f"DeepSeek {label} request failed: {error}"
                    ) from error
                time.sleep(_retry_wait(attempt))
                continue
            if _is_retryable(response.status_code) and attempt < MAX_ATTEMPTS:
                time.sleep(_retry_wait(attempt, response))
                continue
            if response.is_redirect:
                location = response.headers.get("Location") or "an unknown location"
                raise DeepSeekError(
                    f"DeepSeek {label} API error {response.status_code}: "
                    f"redirected to {location}"
                )
            if not response.is_success:
                raise DeepSeekError(
                    f"DeepSeek {label} API error {response.status_code}: "
                    f"{_error_message(response)}"
                )
            try:
                data = response.json()
            except ValueError as error:
                raise DeepSeekError(
                    f"DeepSeek {label} returned invalid JSON: {error}"
                ) from error
            if not isinstance(data, dict):
                raise DeepSeekError(
                    f"DeepSeek {label} returned unexpected JSON: "
                    f"{type(data).__name__}. The API may have changed."
                )
            return data


def _text(value):
    """Return value as a stripped string, or "" when it is not a string."""
    return value.strip() if isinstance(value, str) else ""


def _format_result(position, result):
    lines = [
        f"{position}. {_text(result.get('title')) or '(no title)'}",
        f"   URL: {_text(result.get('url'))}",
    ]
    details = []
    if site_name := _text(result.get("site_name")):
        details.append(f"Site: {site_name}")
    if date := _text(result.get("date")):
        details.append(f"Date: {date}")
    score = result.get("rerank_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        details.append(f"Relevance: {score:.2f}/3")
    authority = result.get("authority")
    if isinstance(authority, int) and not isinstance(authority, bool):
        if authority >= 0:
            details.append(f"Authority: {authority}/4")
    if details:
        lines.append("   " + " | ".join(details))
    summary = _text(result.get("summary")) or _text(result.get("snippet"))
    if summary:
        lines.append(f"   Summary: {summary}")
    content = _text(result.get("content"))
    if content:
        if len(content) > MAX_SEARCH_CONTENT_CHARS:
            content = (
                content[:MAX_SEARCH_CONTENT_CHARS]
                + " [truncated, use deepseek_web_fetch]"
            )
        lines.append(f"   Content: {content}")
    return "\n".join(lines)


def deepseek_web_search(
    query: str,
    count: int = DEFAULT_COUNT,
    question: str = "",
    need_content: bool = False,
) -> str:
    """
    Search the web with the DeepSeek Search API.

    The query is 1 to 128 characters and supports these operators, which can
    be combined: site:example.com (several with OR), -site:example.com, -term,
    "exact phrase", -"excluded phrase", lang:en or lang:zh,en, after:2025 or
    after:2025-01-31, before:2026.

    Args:
        query: The search query, 1 to 128 characters. Rejected when empty or
            longer.
        count: Number of results, 1 to 50 (default: 5)
        question: Optional. The user's original question, used to rank the
            results. Cut to 128 characters.
        need_content: Optional. Also return the page content of each result,
            up to 3000 characters each (default: False). Accepts true/false,
            yes/no, on/off and 1/0. Use deepseek_web_fetch for a full page.

    Returns:
        Numbered results with title, URL, site, date, relevance (0-3, 1.5 or
        more is good), authority (0-4, 3 or more is highly authoritative) and
        a summary
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise DeepSeekError(f"count must be a whole number, got {count!r}") from None
    if not isinstance(query, str):
        raise DeepSeekError(f"query must be a string, got {type(query).__name__}")
    if question is not None and not isinstance(question, str):
        raise DeepSeekError(f"question must be a string, got {type(question).__name__}")
    query = query.strip()
    if not query:
        raise DeepSeekError("query must not be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise DeepSeekError(f"query is longer than {MAX_QUERY_LENGTH} characters")
    body = {
        "query": query,
        "count": min(max(count, 1), MAX_COUNT),
        "need_content": _as_bool(need_content, "need_content"),
    }
    question = (question or "").strip()[:MAX_QUESTION_LENGTH]
    if question:
        body["ori_question"] = question

    payload = _post("search", body)
    if "results" not in payload:
        raise DeepSeekError(
            "DeepSeek Search returned no results field. The API may have changed."
        )
    raw_results = payload["results"]
    if raw_results is None:
        raw_results = []
    if not isinstance(raw_results, list):
        raise DeepSeekError(
            f"DeepSeek Search returned {type(raw_results).__name__} results, "
            "expected a list. The API may have changed."
        )
    results = [item for item in raw_results if isinstance(item, dict)]
    if raw_results and not results:
        raise DeepSeekError(
            "DeepSeek Search returned results that are not objects. "
            "The API may have changed."
        )
    if not results:
        return "No results found"
    return "\n\n".join(
        _format_result(position, result)
        for position, result in enumerate(results, start=1)
    )


def deepseek_web_fetch(url: str) -> str:
    """
    Fetch a web page with the DeepSeek Fetch API and return it as Markdown.

    Args:
        url: The http or https URL to fetch, at most 8192 characters, with no
            user name or password in it

    Returns:
        The page title and the page content as Markdown
    """
    url = _validate_url(url)
    page = _post("fetch", {"url": url})
    raw_content = page.get("content")
    if raw_content is None or (
        isinstance(raw_content, str) and not raw_content.strip()
    ):
        return f"No content could be fetched from {url}"
    content = _text(raw_content)
    if not content:
        raise DeepSeekError(
            f"DeepSeek Fetch returned {type(raw_content).__name__} content, "
            "expected text. The API may have changed."
        )
    title = _text(page.get("title")) or "(no title)"
    return f"Title: {title}\nURL: {url}\n\n{content}"
