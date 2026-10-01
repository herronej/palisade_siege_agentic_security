"""siege.served: one place decides the served host, model and OpenRouter pin.

Checked against real request bodies through a mock HTTP transport, so a wrong
pin fails here rather than as a run quietly served by a host nobody chose.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import openai
import pytest

from siege import served

_COMPLETION = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ready"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _capture():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)

    return seen, httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("SIEGE_SERVED_MODEL", "SIEGE_OPENROUTER_PROVIDERS"):
        monkeypatch.delenv(var, raising=False)
    served.install()


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


def test_model_name_defaults_to_the_science_cloud_name(monkeypatch):
    assert served.served_model_name() == "gpt-oss-120b"
    assert served.served_model_spec() == "openai-chat:gpt-oss-120b"
    monkeypatch.setenv("SIEGE_SERVED_MODEL", "openai/gpt-oss-120b")
    assert served.served_model_spec() == "openai-chat:openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "base, expected",
    [
        ("https://gateway.example", "https://gateway.example/v1"),
        ("https://gateway.example/", "https://gateway.example/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
        ("https://bedrock-mantle.us-east-1.api.aws/v1",
         "https://bedrock-mantle.us-east-1.api.aws/v1"),
        ("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/",
         "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"),
    ],
)
def test_v1_suffix_is_added_exactly_once(base, expected):
    assert served.openai_v1_base(base) == expected


def test_openrouter_is_recognised_by_host_only():
    assert served.is_openrouter("https://openrouter.ai/api/v1")
    assert not served.is_openrouter("https://bedrock-mantle.us-east-1.api.aws/v1")
    assert not served.is_openrouter("https://example.com/openrouter.ai/v1")


def test_routing_pins_only_the_listed_hosts_with_no_fallback(monkeypatch):
    assert served.provider_routing() is None
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", " deepinfra/bf16 , akashml/bf16 ,")
    assert served.provider_routing() == {
        "only": ["deepinfra/bf16", "akashml/bf16"],
        "order": ["deepinfra/bf16", "akashml/bf16"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_describe_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-secret")
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    info = served.describe()
    assert "sk-or-secret" not in json.dumps(info)
    assert info["host"] == "openrouter.ai"
    assert info["openrouter_routing"]["only"] == ["deepinfra/bf16"]


# -----------------------------------------------------------------
# The pin reaches real request bodies
# -----------------------------------------------------------------


def test_sync_openrouter_request_carries_the_pin(monkeypatch):
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    seen, transport = _capture()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                           http_client=httpx.Client(transport=transport))
    client.chat.completions.create(model="openai/gpt-oss-120b",
                                   messages=[{"role": "user", "content": "hi"}])
    assert seen[0]["provider"]["only"] == ["deepinfra/bf16"]
    assert seen[0]["provider"]["allow_fallbacks"] is False
    assert seen[0]["provider"]["require_parameters"] is True


def test_async_openrouter_request_carries_the_pin(monkeypatch):
    """pydantic-ai's agents use the async client."""
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "akashml/bf16")
    seen, transport = _capture()
    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                                http_client=httpx.AsyncClient(transport=transport))
    asyncio.run(client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}]))
    assert seen[0]["provider"]["only"] == ["akashml/bf16"]


def test_existing_extra_body_is_kept(monkeypatch):
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    seen, transport = _capture()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                           http_client=httpx.Client(transport=transport))
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"effort": "medium"}},
    )
    assert seen[0]["reasoning"] == {"effort": "medium"}
    assert "provider" in seen[0]


def test_other_hosts_are_left_alone(monkeypatch):
    """A pin configured for OpenRouter must not leak into a Bedrock or Science
    Cloud request, which would reject the unknown field."""
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    seen, transport = _capture()
    client = openai.OpenAI(base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
                           api_key="k", http_client=httpx.Client(transport=transport))
    client.chat.completions.create(model="openai.gpt-oss-120b",
                                   messages=[{"role": "user", "content": "hi"}])
    assert "provider" not in seen[0]


def test_no_pin_means_no_provider_field():
    seen, transport = _capture()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                           http_client=httpx.Client(transport=transport))
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert "provider" not in seen[0]


def test_install_is_idempotent(monkeypatch):
    served.install()
    served.install()
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    seen, transport = _capture()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                           http_client=httpx.Client(transport=transport))
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert seen[0]["provider"]["only"] == ["deepinfra/bf16"]


# -----------------------------------------------------------------
# The two clients that do not go through install()
# -----------------------------------------------------------------


def test_judge_raw_http_request_carries_the_pin(monkeypatch):
    from siege.redteam.baselines.detectors import LlmJudgeDetector

    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    monkeypatch.setenv("SIEGE_SERVED_MODEL", "openai/gpt-oss-120b")
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "BENIGN"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    judge = LlmJudgeDetector(base_url="https://openrouter.ai/api/v1", api_key="k")
    judge.flag("some text")
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["body"]["provider"]["only"] == ["deepinfra/bf16"]


def test_attacker_model_uses_the_normalised_base(monkeypatch):
    from tools.optimizer_hardwin import _served_model

    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("SIEGE_SERVED_MODEL", "openai/gpt-oss-120b")
    model = _served_model()
    assert model.model_name == "openai/gpt-oss-120b"
    assert str(model.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_recommended_posture_refuses_semgrep():
    from tools.optimizer_hardwin import main

    assert main(["--posture", "recommended", "--semgrep"]) == 1


# -----------------------------------------------------------------
# A run that loses its endpoint stops instead of scoring the failure
# -----------------------------------------------------------------

_HI = [{"role": "user", "content": "hi"}]


def _status(code: int, message: str) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(code, json={"error": {"message": message, "code": code}})
    )


def _client(transport: httpx.MockTransport) -> openai.OpenAI:
    # max_retries=0: the SDK's own retries would only slow these tests down.
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                         max_retries=0, http_client=httpx.Client(transport=transport))


def _agent_model(transport: httpx.MockTransport):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="k",
                                max_retries=0,
                                http_client=httpx.AsyncClient(transport=transport))
    return OpenAIChatModel("m", provider=OpenAIProvider(openai_client=client))


@pytest.fixture
def fast_guard(monkeypatch):
    monkeypatch.setattr(served, "_FIRST_DELAY_S", 0.01)
    monkeypatch.setattr(served, "_MAX_DELAY_S", 0.02)


@pytest.mark.parametrize(
    "status, message, fatal",
    [
        (401, "Invalid bearer token", True),
        (402, "Insufficient credits", True),
        (403, "Forbidden", True),
        (403, "Your chosen model requires moderation and your input was flagged", False),
        (404, "No endpoints found that support the provided parameters", True),
        (429, "ExceededBudget: Spend 50.74, Budget 50.00", True),
        (429, "Rate limit exceeded, retry shortly", False),
        (400, "maximum context length exceeded", False),
        (500, "internal error", False),
    ],
)
def test_only_errors_retrying_cannot_fix_are_refusals(status, message, fatal):
    assert (served.refusal_reason_for_status(status, message) is not None) is fatal


def test_no_credit_gets_past_the_handlers_that_score_failures():
    client = _client(_status(402, "Insufficient credits"))
    with pytest.raises(served.EndpointRefused, match="out of credit"):
        try:
            client.chat.completions.create(model="m", messages=_HI)
        except Exception:  # what every tool wraps a model call in
            pytest.fail("the refusal was caught as an ordinary error")


def test_async_exhausted_budget_is_a_refusal():
    client = openai.AsyncOpenAI(
        base_url="https://gateway.example/v1", api_key="k", max_retries=0,
        http_client=httpx.AsyncClient(
            transport=_status(429, "ExceededBudget: Spend 50.74, Budget 50.00")),
    )
    with pytest.raises(served.EndpointRefused, match="budget"):
        asyncio.run(client.chat.completions.create(model="m", messages=_HI))


def test_a_failure_the_endpoint_survives_is_raised_as_before(fast_guard, monkeypatch):
    """A 500 while the endpoint answers the probe belongs to that request."""
    monkeypatch.setenv("SIEGE_OPENROUTER_PROVIDERS", "deepinfra/bf16")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["messages"] == _HI:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(200, json=_COMPLETION)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(model="openai/gpt-oss-120b", messages=_HI)
    request, probe = seen
    assert probe["model"] == "openai/gpt-oss-120b"
    assert probe["provider"]["only"] == ["deepinfra/bf16"]


def test_an_outage_is_waited_out(fast_guard):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 4:  # request, probe, request, probe
            return httpx.Response(503, json={"error": {"message": "no available providers"}})
        return httpx.Response(200, json=_COMPLETION)

    r = _client(httpx.MockTransport(handler)).chat.completions.create(model="m", messages=_HI)
    assert r.choices[0].message.content == "ready"
    assert len(calls) == 5


def test_an_outage_past_the_patience_limit_is_a_refusal(fast_guard, monkeypatch):
    monkeypatch.setenv("SIEGE_ENDPOINT_PATIENCE_S", "0.05")
    client = _client(_status(503, "no available providers"))
    with pytest.raises(served.EndpointRefused, match="has not answered"):
        client.chat.completions.create(model="m", messages=_HI)


def test_a_rate_limit_is_waited_out_without_probing(fast_guard):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(200, json=_COMPLETION)

    _client(httpx.MockTransport(handler)).chat.completions.create(model="m", messages=_HI)
    assert [body["messages"] for body in seen] == [_HI, _HI]


def test_a_request_error_is_not_retried():
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(400, json={"error": {"message": "maximum context length exceeded"}})

    with pytest.raises(openai.BadRequestError):
        _client(httpx.MockTransport(handler)).chat.completions.create(model="m", messages=_HI)
    assert len(seen) == 1


def test_monitor_run_stops_rather_than_failing_closed():
    """The E1a failure mode: an exhausted account scored as monitor denials."""
    from siege import SessionRunner, load_instances
    from siege.ablation_matrix import MONITOR_GATE_IDS, PROMPT_MONITOR_CONFIG
    from siege.paths import CORPUS_DIR
    from siege.redteam.baselines.prompt_monitor import build_monitor_agent
    from tools.prompt_monitor_baseline import RetryingCachingMonitor

    stats = {"model_calls": 0, "cache_hits": 0, "retries": 0, "errors": 0}
    monitor = RetryingCachingMonitor(
        build_monitor_agent(_agent_model(_status(402, "Insufficient credits"))), stats
    )
    inst = next(i for i in load_instances(CORPUS_DIR)
                if i.template == "xc_1_cross_boundary_chain")
    runner = SessionRunner(
        quarantine_agents={gid: monitor for gid in MONITOR_GATE_IDS}, bound="production"
    )
    with pytest.raises(BaseException) as info:
        asyncio.run(runner.run(inst, PROMPT_MONITOR_CONFIG))
    assert served.find_refusal(info.value) is not None
    assert stats["errors"] == 0 and stats["retries"] == 0


def test_attacker_stops_rather_than_replaying_its_last_candidate():
    """The E2/E3 failure mode: a failed proposer call reads as the defense holding."""
    from siege.redteam.llm_optimizer import ProposalRequest
    from tools.optimizer_hardwin import FallbackCounter, _SalvagingAgentProposer

    counter = FallbackCounter()
    proposer = _SalvagingAgentProposer(
        _agent_model(_status(402, "Insufficient credits")),
        temperature=0.7, seed=0, counter=counter,
    )
    with pytest.raises(BaseException) as info:
        proposer.propose(ProposalRequest(goal="sbatch --qos=urgent job.sh", history=()))
    assert served.find_refusal(info.value) is not None
    assert counter.hard_fail == 0 and counter.salvaged == 0


def test_judge_stops_rather_than_scoring_not_flagged(monkeypatch):
    import io
    import urllib.error

    from siege.redteam.baselines.detectors import LlmJudgeDetector

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 402, "Payment Required", {},
            io.BytesIO(b'{"error": {"message": "Insufficient credits"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    judge = LlmJudgeDetector(base_url="https://openrouter.ai/api/v1", api_key="k")
    with pytest.raises(served.EndpointRefused, match="out of credit"):
        judge.flag("some text")


def test_judge_still_reports_an_ordinary_http_error(monkeypatch):
    import io
    import urllib.error

    from siege.redteam.baselines.detectors import LlmJudgeDetector

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    judge = LlmJudgeDetector(base_url="https://openrouter.ai/api/v1", api_key="k")
    verdict = judge.flag("some text")
    assert not verdict.flagged and verdict.reason == "llm-judge error"
    assert judge.error_rate == 1.0


def test_exit_on_refusal_turns_a_refusal_into_status_3(capsys):
    def refused():
        raise served.EndpointRefused("the account is out of credit (HTTP 402)")

    def grouped():
        try:
            raise served.EndpointRefused("the account is out of credit (HTTP 402)")
        except served.EndpointRefused as exc:
            raise BaseExceptionGroup("tasks", [exc])

    assert served.exit_on_refusal(refused) == 3
    assert served.exit_on_refusal(grouped) == 3
    assert "ABORTED: the account is out of credit" in capsys.readouterr().err


def test_exit_on_refusal_leaves_everything_else_alone():
    def bug():
        raise ValueError("a bug, not the endpoint")

    with pytest.raises(ValueError):
        served.exit_on_refusal(bug)
    assert served.exit_on_refusal(lambda: None) == 0
    assert served.exit_on_refusal(lambda: 1) == 1


def test_openrouter_credit_reads_the_unspent_balance(monkeypatch):
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"data": {"total_credits": 20, "total_usage": 1.25}}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert served.openrouter_credit("https://openrouter.ai/api/v1", "k") == 18.75
    assert captured["url"] == "https://openrouter.ai/api/v1/credits"

    def broken(req, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", broken)
    assert served.openrouter_credit("https://openrouter.ai/api/v1", "k") is None


def test_hosts_label_lists_each_host_once():
    routed = {"host": "openrouter.ai", "openrouter_routing": {"only": ["deepinfra/bf16"]}}
    direct = {"host": "bedrock-mantle.us-east-1.api.aws", "openrouter_routing": None}
    label = served.hosts_label([{"endpoint": routed}, {"endpoint": routed},
                                {"endpoint": direct}, {}])
    assert label == (
        "`openrouter.ai` pinned to `deepinfra/bf16`; "
        "`bedrock-mantle.us-east-1.api.aws`; "
        "not recorded (these runs predate host logging)"
    )


# -----------------------------------------------------------------
# pydantic-ai must not read a vendor-prefixed gpt-oss as an o-series model
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, reasoning",
    [
        ("gpt-oss-120b", False),          # the Science Cloud name, as Table II ran
        ("openai/gpt-oss-120b", False),   # OpenRouter: starts with "o"
        ("openai.gpt-oss-120b", False),   # Bedrock: starts with "o"
        ("openai/o3", True),              # a real o-series model stays one
        ("allenai/olmo-2-0325-32b", False),
        ("mistralai/mistral-small-2603", False),
    ],
)
def test_profile_ignores_only_the_openai_vendor_prefix(name, reasoning):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(name, provider=OpenAIProvider(base_url="https://x.example/v1", api_key="k"))
    assert model.profile.openai_supports_reasoning is reasoning


def test_attacker_request_keeps_temperature_and_seed_on_openrouter():
    """The regression: on `openai/gpt-oss-120b` pydantic-ai dropped temperature."""
    from tools.optimizer_hardwin import FallbackCounter, _SalvagingAgentProposer
    from siege.redteam.llm_optimizer import ProposalRequest

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "stop here"}})

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="k", max_retries=0,
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    model = OpenAIChatModel("openai/gpt-oss-120b", provider=OpenAIProvider(openai_client=client))
    proposer = _SalvagingAgentProposer(model, temperature=0.7, seed=3, counter=FallbackCounter())
    assert proposer.propose(ProposalRequest(goal="g", history=())) == []
    assert seen, "no request reached the endpoint"
    for body in seen:  # the structured call and the plain-text salvage
        assert body["temperature"] == 0.7
        assert body["seed"] == 3
