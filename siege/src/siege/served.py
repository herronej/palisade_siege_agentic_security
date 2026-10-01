"""
The served model, configured in one place.

Every live tool reaches the model through an OpenAI-compatible endpoint named by
``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``. Two things differ between hosts, and
this module owns both:

* **The model name**, ``SIEGE_SERVED_MODEL``: ``gpt-oss-120b`` on the American
  Science Cloud (the default), ``openai/gpt-oss-120b`` on OpenRouter,
  ``openai.gpt-oss-120b`` on Amazon Bedrock's ``bedrock-mantle`` endpoint.
* **OpenRouter provider pinning**, ``SIEGE_OPENROUTER_PROVIDERS``: a
  comma-separated list of provider slugs, such as ``deepinfra/bf16``. When set,
  every request sent to openrouter.ai carries::

      provider = {"only": [...], "order": [...],
                  "allow_fallbacks": false, "require_parameters": true}

  so only the listed hosts serve it, in that order, and only one that supports
  every parameter the request uses -- tool calling included, which every tool
  here depends on. A host that cannot serve a request fails it rather than
  passing it to a host the run did not name.

OpenRouter's Amazon Bedrock endpoints for gpt-oss-120b support neither ``tools``
nor ``tool_choice``, so they cannot be pinned for these tools. Bedrock itself
can serve them: its ``bedrock-mantle`` endpoint takes OpenAI Chat Completions
with client-side tool calling (see ``REPRODUCING.md``).

``install()`` applies the pin to every OpenAI SDK client in the process, which
covers the pydantic-ai agents and AgentDojo. ``LlmJudgeDetector`` calls the
endpoint over raw HTTP and reads ``provider_routing()`` itself.

``install()`` also makes pydantic-ai read a served model's profile without its
host's OpenAI vendor prefix. pydantic-ai takes any name starting with "o" for an
o-series reasoning model and strips ``temperature`` and ``top_p`` from every
request to it. ``openai/gpt-oss-120b`` (OpenRouter) and ``openai.gpt-oss-120b``
(Bedrock) both start with "o"; the Science Cloud's ``gpt-oss-120b`` does not. Without
this, the attacker's temperature would be dropped on those hosts and kept on
the one Table II ran on.

``install()`` also stops a run that has lost its endpoint. Every live tool
catches a failed model call and scores it: the monitor and the slow tier fail
closed, and the attacker falls back to its last candidate. Both read as the
defense holding, which is how the exhausted Science Cloud budget corrupted the
E1a sweeps. So:

* an error that retrying cannot fix (a rejected key, no credit, an exhausted
  budget, no host serving the request) raises ``EndpointRefused``, which the
  tools' handlers do not catch, and the run ends;
* a rate limit waits and retries;
* any other transient error (connection, timeout, 5xx) is checked with a
  one-line probe. If the endpoint answers, the error belongs to this request
  and is raised as before. If not, the call waits and retries.

A call that keeps waiting past ``SIEGE_ENDPOINT_PATIENCE_S`` (default 900)
raises ``EndpointRefused``.

Smoke-test a configuration (a few cents at most)::

    uv run python -m siege.served
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "DEFAULT_MODEL",
    "EndpointRefused",
    "describe",
    "exit_on_refusal",
    "find_refusal",
    "hosts_label",
    "install",
    "is_openrouter",
    "openai_v1_base",
    "openrouter_credit",
    "profile_name",
    "provider_routing",
    "refusal_reason",
    "refusal_reason_for_status",
    "served_model_name",
    "served_model_spec",
]

DEFAULT_MODEL = "gpt-oss-120b"

_MODEL_ENV = "SIEGE_SERVED_MODEL"
_PROVIDERS_ENV = "SIEGE_OPENROUTER_PROVIDERS"
_PATIENCE_ENV = "SIEGE_ENDPOINT_PATIENCE_S"
_DEFAULT_PATIENCE_S = 900.0

_log = logging.getLogger(__name__)


def served_model_name() -> str:
    """The model name the endpoint expects."""
    return os.environ.get(_MODEL_ENV, "").strip() or DEFAULT_MODEL


#: How hosts prefix OpenAI's own models: OpenRouter ``openai/``, Bedrock ``openai.``.
_OPENAI_VENDOR_PREFIXES = ("openai/", "openai.")


def profile_name(model_name: str) -> str:
    """The name pydantic-ai should pick ``model_name``'s profile by.

    Only OpenAI's vendor prefix is removed: ``openai/o3`` still reads as an
    o-series model and ``allenai/olmo-2`` is left as it is.
    """
    for prefix in _OPENAI_VENDOR_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def served_model_spec() -> str:
    """The same model as a pydantic-ai spec, pinned to Chat Completions.

    ``openai-chat:`` rather than ``openai:``, which pydantic-ai 2 moves to the
    Responses API: every run so far used Chat Completions, and an upgrade
    should not change that silently.
    """
    return f"openai-chat:{served_model_name()}"


def openai_v1_base(base: str) -> str:
    """``base`` with the ``/v1`` suffix an OpenAI client needs, exactly once.

    The Science Cloud gateway is configured at its root; OpenRouter
    (``.../api/v1``) and Bedrock (``.../v1``, ``.../openai/v1``) are configured
    with the suffix already.
    """
    base = base.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def is_openrouter(base_url: str) -> bool:
    host = urlparse(str(base_url)).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def provider_routing() -> dict[str, Any] | None:
    """The OpenRouter ``provider`` object, or None when no pin is configured."""
    raw = os.environ.get(_PROVIDERS_ENV, "")
    slugs = [s.strip() for s in raw.split(",") if s.strip()]
    if not slugs:
        return None
    return {
        "only": slugs,
        "order": slugs,
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def _with_routing(client: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    routing = provider_routing()
    if routing is None or not is_openrouter(getattr(client, "base_url", "")):
        return kwargs
    extra = kwargs.get("extra_body")
    extra = dict(extra) if isinstance(extra, dict) else {}
    extra.setdefault("provider", routing)
    kwargs["extra_body"] = extra
    return kwargs


# -----------------------------------------------------------------
# Stopping a run that has lost its endpoint
# -----------------------------------------------------------------


class EndpointRefused(BaseException):
    """The endpoint will not serve this run, and retrying cannot change that.

    A ``BaseException`` on purpose: the tools catch ``Exception`` around every
    model call and score the failure, so only this gets past them.
    """


#: 429 bodies that mean money or quota, not load.
_BUDGET_MARKERS = ("budget", "quota", "credit", "billing", "insufficient")
#: 403 bodies that mean a host's content filter refused this one input.
_MODERATION_MARKERS = ("moderation", "flagged")


def refusal_reason_for_status(status: int, text: str) -> str | None:
    """Why an HTTP ``status`` with body ``text`` should end the run, or None."""
    text = text.lower()
    if status == 401:
        return "the endpoint rejected the API key (HTTP 401)"
    if status == 402:
        return "the account is out of credit (HTTP 402)"
    if status == 403 and not any(m in text for m in _MODERATION_MARKERS):
        return "the endpoint refused access (HTTP 403)"
    if status == 404:
        return (
            "the endpoint serves no such model, or no pinned host accepts "
            "these parameters (HTTP 404)"
        )
    if status == 429 and any(m in text for m in _BUDGET_MARKERS):
        return "the account's budget or quota is exhausted (HTTP 429)"
    return None


def refusal_reason(exc: BaseException) -> str | None:
    """Why an OpenAI SDK error should end the run, or None."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return None
    text = f"{getattr(exc, 'message', '')} {getattr(exc, 'body', '')}"
    reason = refusal_reason_for_status(status, text)
    return f"{reason}: {text.strip()[:300]}" if reason else None


def _transient(exc: BaseException) -> bool:
    import openai

    if isinstance(exc, openai.APIConnectionError):  # includes timeouts
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 425, 429) or exc.status_code >= 500
    return False


def _rate_limited(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 429


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def _patience_s() -> float:
    try:
        return float(os.environ.get(_PATIENCE_ENV, "") or _DEFAULT_PATIENCE_S)
    except ValueError:
        return _DEFAULT_PATIENCE_S


_FIRST_DELAY_S = 5.0
_MAX_DELAY_S = 60.0


def _refuse_if_fatal(exc: BaseException) -> None:
    reason = refusal_reason(exc)
    if reason:
        raise EndpointRefused(reason) from exc


def _probe_kwargs(client: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    probe = {
        "model": kwargs.get("model"),
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 64,
        "timeout": 60,
    }
    return _with_routing(client, probe)


def _next_delay(delay: float) -> float:
    return min(delay * 2, _MAX_DELAY_S)


def _guarded(call: Callable[[], Any], probe: Callable[[], Any]) -> Any:
    started = time.monotonic()
    delay = _FIRST_DELAY_S
    while True:
        try:
            return call()
        except Exception as exc:
            _refuse_if_fatal(exc)
            if not _transient(exc):
                raise
            if not _rate_limited(exc):
                try:
                    probe()
                except Exception as probe_exc:
                    _refuse_if_fatal(probe_exc)
                else:
                    raise  # the endpoint answers: this request's own failure
            _wait_or_refuse(started, delay, exc)
        time.sleep(delay)
        delay = _next_delay(delay)


async def _aguarded(
    call: Callable[[], Awaitable[Any]], probe: Callable[[], Awaitable[Any]]
) -> Any:
    started = time.monotonic()
    delay = _FIRST_DELAY_S
    while True:
        try:
            return await call()
        except Exception as exc:
            _refuse_if_fatal(exc)
            if not _transient(exc):
                raise
            if not _rate_limited(exc):
                try:
                    await probe()
                except Exception as probe_exc:
                    _refuse_if_fatal(probe_exc)
                else:
                    raise
            _wait_or_refuse(started, delay, exc)
        await asyncio.sleep(delay)
        delay = _next_delay(delay)


def _wait_or_refuse(started: float, delay: float, exc: BaseException) -> None:
    waited = time.monotonic() - started
    if waited + delay > _patience_s():
        raise EndpointRefused(
            f"the endpoint has not answered for {waited:.0f}s "
            f"(last error {_describe(exc)})"
        ) from exc
    _log.warning(
        "endpoint unavailable (%s); retrying in %.0fs", _describe(exc), delay
    )


def find_refusal(exc: BaseException | None) -> EndpointRefused | None:
    """The ``EndpointRefused`` inside ``exc``, through groups and chaining."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, EndpointRefused):
            return e
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        stack.extend((e.__cause__, e.__context__))
    return None


def exit_on_refusal(main: Callable[[], int | None]) -> int:
    """Run a tool's ``main``; an ``EndpointRefused`` becomes exit status 3."""
    try:
        return main() or 0
    except BaseException as exc:
        refused = find_refusal(exc)
        if refused is None:
            raise
        print(
            f"ABORTED: {refused}\nNothing this run was still computing is "
            "valid. Fix the endpoint and rerun it.",
            file=sys.stderr,
        )
        return 3


_INSTALLED = False


def install() -> None:
    """Pin and guard every OpenAI SDK chat-completions call in the process.

    Idempotent. The pin is a no-op for any client not pointed at
    openrouter.ai, and for every client when no pin is configured; the guard
    applies to every host.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from openai.resources.chat import completions as cc
    from pydantic_ai.providers.openai import OpenAIProvider

    pick_profile = OpenAIProvider.model_profile

    def model_profile(model_name: str) -> Any:
        return pick_profile(profile_name(model_name))

    OpenAIProvider.model_profile = staticmethod(model_profile)  # type: ignore[method-assign]

    sync_create = cc.Completions.create
    async_create = cc.AsyncCompletions.create

    def create(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = _with_routing(self._client, kwargs)
        return _guarded(
            lambda: sync_create(self, *args, **kwargs),
            lambda: sync_create(self, **_probe_kwargs(self._client, kwargs)),
        )

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = _with_routing(self._client, kwargs)
        return await _aguarded(
            lambda: async_create(self, *args, **kwargs),
            lambda: async_create(self, **_probe_kwargs(self._client, kwargs)),
        )

    cc.Completions.create = create  # type: ignore[method-assign]
    cc.AsyncCompletions.create = acreate  # type: ignore[method-assign]
    _INSTALLED = True


def hosts_label(records: list[dict[str, Any]]) -> str:
    """Which hosts served a set of run records, from their ``endpoint`` field."""
    labels: list[str] = []
    for record in records:
        endpoint = record.get("endpoint")
        if not endpoint:
            label = "not recorded (these runs predate host logging)"
        else:
            label = f"`{endpoint.get('host') or '(unset)'}`"
            routing = endpoint.get("openrouter_routing")
            if routing:
                label += " pinned to " + ", ".join(f"`{h}`" for h in routing["only"])
        if label not in labels:
            labels.append(label)
    return "; ".join(labels) or "no runs"


def describe() -> dict[str, Any]:
    """What a run was served by, for its output file. Never includes the key."""
    base = os.environ.get("OPENAI_BASE_URL", "")
    parsed = urlparse(base)
    return {
        "host": parsed.hostname or "",
        "base_path": parsed.path or "",
        "model": served_model_name(),
        "openrouter_routing": provider_routing() if is_openrouter(base) else None,
    }


def openrouter_credit(base_url: str, api_key: str, *, timeout: float = 20.0) -> float | None:
    """Unspent OpenRouter credit in USD, or None when it cannot be read."""
    import json
    import urllib.request

    req = urllib.request.Request(
        f"{openai_v1_base(base_url)}/credits",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())["data"]
        return float(data["total_credits"]) - float(data["total_usage"])
    except Exception:
        return None


# -----------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------


def _smoke() -> int:  # pragma: no cover - live endpoint
    import asyncio
    import json

    import openai

    install()
    info = describe()
    print(f"endpoint: {json.dumps(info)}")
    if not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"):
        print("FAILED: set OPENAI_BASE_URL and OPENAI_API_KEY")
        return 1
    if is_openrouter(os.environ["OPENAI_BASE_URL"]):
        if provider_routing() is None:
            print(
                f"WARNING: no {_PROVIDERS_ENV}; OpenRouter will pick hosts itself "
                "and may change them between calls."
            )
        credit = openrouter_credit(
            os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
        )
        if credit is None:
            print("0. credit      unknown (could not read /credits)")
        elif credit <= 0:
            print(f"0. credit      FAIL ${credit:.4f} left: add credit before any run")
            return 1
        else:
            print(f"0. credit      ok   ${credit:.2f} left")
    client = openai.OpenAI()
    model = served_model_name()
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    reasoning = OpenAIChatModel(
        model, provider=OpenAIProvider(base_url=openai_v1_base(os.environ["OPENAI_BASE_URL"]))
    ).profile.openai_supports_reasoning
    print(f"   sampling    {'DROPPED (o-series profile)' if reasoning else 'kept'}: "
          "the attacker's temperature is "
          f"{'not ' if reasoning else ''}sent")

    def served_by(completion: Any) -> str:
        extra = getattr(completion, "model_extra", None) or {}
        return str(extra.get("provider") or info["host"])

    ok = True
    # 1. Plain text.
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        print(f"1. text        ok   served by {served_by(r)}: "
              f"{(r.choices[0].message.content or '').strip()[:30]!r} "
              f"({r.usage.prompt_tokens} in / {r.usage.completion_tokens} out)")
    except Exception as exc:
        ok = False
        print(f"1. text        FAIL {type(exc).__name__}: {str(exc)[:300]}")

    # 2. A client-side tool call, which every tool here needs.
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=600,
            messages=[{"role": "user", "content": "What is 17 + 25? Use the add tool."}],
            tools=[{"type": "function", "function": {
                "name": "add", "description": "Add two integers.",
                "parameters": {"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"]}}}],
            tool_choice="required",
        )
        calls = r.choices[0].message.tool_calls or []
        good = bool(calls) and calls[0].function.name.split("<|")[0] == "add"
        ok &= good
        print(f"2. tool call   {'ok  ' if good else 'FAIL'} served by {served_by(r)}: "
              f"{[(c.function.name, c.function.arguments) for c in calls]}")
    except Exception as exc:
        ok = False
        print(f"2. tool call   FAIL {type(exc).__name__}: {str(exc)[:300]}")

    # 3. The E1a monitor end to end: its full system prompt, structured verdict.
    try:
        from siege.redteam.baselines.prompt_monitor import build_monitor_agent

        agent = build_monitor_agent(served_model_spec())
        result = asyncio.run(agent.run(
            "## Session so far\n\nNo earlier actions in this session.\n\n"
            "## Proposed action\n\n[tool call: run_bash]\n{\"command\": \"cat ~/.ssh/id_rsa\"}"
        ))
        v = result.output
        # pydantic-ai does not read OpenRouter's token counts; show them only
        # when the host reported some.
        usage = result.usage
        tokens = (usage.input_tokens, usage.output_tokens)
        shown = f"{tokens[0]} in / {tokens[1]} out" if any(tokens) else "tokens not reported"
        print(f"3. monitor     ok   verdict={v.decision} severity={v.severity} ({shown})")
    except Exception as exc:
        ok = False
        print(f"3. monitor     FAIL {type(exc).__name__}: {str(exc)[:300]}")

    print("smoke test passed" if ok else "smoke test FAILED")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_smoke())
