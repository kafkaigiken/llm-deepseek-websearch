# llm-deepseek-websearch

[![PyPI](https://img.shields.io/pypi/v/llm-deepseek-websearch.svg)](https://pypi.org/project/llm-deepseek-websearch/)
[![Changelog](https://img.shields.io/github/v/release/kafkaigiken/llm-deepseek-websearch?include_prereleases&label=changelog)](https://github.com/kafkaigiken/llm-deepseek-websearch/releases)
[![Tests](https://github.com/kafkaigiken/llm-deepseek-websearch/actions/workflows/test.yml/badge.svg)](https://github.com/kafkaigiken/llm-deepseek-websearch/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/kafkaigiken/llm-deepseek-websearch/blob/main/LICENSE)

An [llm](https://llm.datasette.io/) plugin that gives models two web tools, built on the [DeepSeek Search and Fetch APIs](https://api-docs.deepseek.com/guides/search/):

- `deepseek_web_search`: search the web and return ranked results with summaries.
- `deepseek_web_fetch`: fetch one page and return its content as Markdown.

The tools work with any model that supports tools in `llm`, not only DeepSeek models.

We use this plugin for research as part of our article writing process for [Kafkai Blog](https://kafkai.com/en/blog/), [Kafkai AI Media](https://kafkai.ai/) and [PuchiDen Blog](https://puchiden.app/blog/).

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
You need LLM 0.33 or newer and Python 3.10 or newer.

```bash
llm install llm-deepseek-websearch
```

To install the latest code from GitHub before a release reaches PyPI:

```bash
llm install https://github.com/kafkaigiken/llm-deepseek-websearch/archive/refs/heads/main.zip
```

## API key

Create a key on the [DeepSeek platform](https://platform.deepseek.com/api_keys), then
store it:

```bash
llm keys set deepseek     # paste the key
```

If no stored key exists, the plugin uses the `DEEPSEEK_API_KEY` environment variable.
A stored key takes priority over the environment variable, so remove it with
`llm keys remove deepseek` before relying on the variable. The `deepseek` key alias is
the same one that the `llm-deepseek` model plugin uses, so one stored key serves both
plugins.

## Usage

Search:

```bash
llm -T deepseek_web_search "What changed in the latest Python release?" --td
```

Search, then read the best pages:

```bash
llm -T deepseek_web_search -T deepseek_web_fetch \
  "Find the DeepSeek API rate limits and quote the official page." --td
```

Fetch a page you already know:

```bash
llm -T deepseek_web_fetch "Summarize https://llm.datasette.io/en/stable/tools.html"
```

`--td` (`--tools-debug`) prints every tool call and its result, so you can see what the
model searched for.

### `deepseek_web_search`

| Argument | Default | Description |
|---|---|---|
| `query` | required | The search query, 1 to 128 characters. The plugin rejects an empty, non-string or longer query. |
| `count` | `5` | Number of results. Values outside 1 to 50 are clamped to that range. |
| `question` | `""` | The user's original question. DeepSeek uses it to rank the results. The plugin cuts it to 128 characters. |
| `need_content` | `False` | Also return the page content of each result. Accepts `true`/`false`, `yes`/`no`, `on`/`off` and `1`/`0`. The plugin cuts each result's content to 3,000 characters. Use `deepseek_web_fetch` to read a full page. |

Each result shows the title, URL, site name, date, relevance score (0 to 3, 1.5 or more
is a good match), authority tier (0 to 4, 3 or more is highly authoritative) and a
summary of about 600 to 800 characters.

The query supports these operators, and you can combine them:

| Operator | Example |
|---|---|
| Restrict to a site | `site:python.org`, `site:a.com OR site:b.com` |
| Exclude a site | `-site:pinterest.com` |
| Exclude a term | `-beta` |
| Exact phrase | `"release notes"` |
| Exclude a phrase | `-"release candidate"` |
| Language | `lang:en`, `lang:ja,en` |
| Date range | `after:2025`, `after:2025-06-01 before:2026` |

The tool docstring lists the operators, so models use them without extra prompting.

### `deepseek_web_fetch`

| Argument | Default | Description |
|---|---|---|
| `url` | required | An `http` or `https` URL, at most 8,192 characters, with no user name or password in it. |

It returns the page title, the URL and the page content as Markdown. The content is not
cut.

## Errors and retries

- The plugin makes up to 3 attempts. It retries HTTP 408, 429 and 5xx responses, and
  also network failures. Other HTTP errors fail at once.
- Between attempts it waits for the `Retry-After` header, given as seconds or as an HTTP
  date, and never longer than 10 seconds. Without a usable header it backs off from 1
  second.
- The model sees the HTTP status and DeepSeek's error message, for example
  `DeepSeek Fetch API error 404: ...` when a page has no content that DeepSeek can
  extract. The plugin cuts a long error body to 1,000 characters.
- A redirect fails with the new location in the message. The plugin never follows a
  redirect, so an API that moves a path is reported rather than silently used.
- A bad `url` fails before any request is sent, with a message naming the problem, for
  example `url must start with http:// or https://`.
- A malformed 200 response fails with a message saying the API may have changed, rather
  than looking like an empty search or an empty page.
- HTTP 402 means the account balance is too low. Add funds on the DeepSeek platform.

## Pricing and limits

From the [DeepSeek documentation](https://api-docs.deepseek.com/guides/search/) on
2026-10-19:

- Search and Fetch each cost USD 0.75 per 1,000 successful requests. Errors, empty
  results and timeouts are free. `count` does not change the price.
- Each API allows 5 requests per second per account. Fetch also has a limit per domain.

## Development

To set up this plugin locally, check out the code, then create a virtual environment:

```bash
cd llm-deepseek-websearch
python -m venv venv
source venv/bin/activate
```

Install the dependencies and the test dependencies:

```bash
llm install -e '.[test]'
```

Run the tests:

```bash
python -m pytest
```

The tests use a mock HTTP transport, so they never call the real API and need no key.

## Releasing

The `Publish Python Package` workflow publishes to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so the repository
stores no PyPI token.

To release a new version:

1. Change `version` in `pyproject.toml`, then commit and push.
2. On GitHub, create a release with a new tag, for example `0.2`.
3. The workflow runs the tests on Python 3.10 to 3.13, builds the package and uploads
   it to PyPI.

To check the package before a release, build it in a separate virtual environment:

```bash
python -m venv /tmp/buildenv
/tmp/buildenv/bin/pip install build twine
/tmp/buildenv/bin/python -m build --outdir /tmp/dist
/tmp/buildenv/bin/twine check --strict /tmp/dist/*
```
