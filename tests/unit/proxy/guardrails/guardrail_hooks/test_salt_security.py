from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final, cast

import httpx
import pytest
import respx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.salt_security import DEFAULT_TIMEOUT_SECONDS, SaltSecurityGuardrail
from litellm.proxy.guardrails.guardrail_hooks.salt_security.salt_security import BLOCKED_MESSAGE
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams
from litellm.types.utils import GenericGuardrailAPIInputs

ENDPOINT: Final = "https://ai.example/beta/litellm_basic_guardrail_api"
FAKE_TOKEN: Final = "fake-token-do-not-use-123"


@pytest.fixture(autouse=True)
def _isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])


def _load(api_base: str | None = "https://ai.example", **extras: object) -> SaltSecurityGuardrail:
    params: Final = LitellmParams(
        guardrail="salt_security", mode="pre_call", api_base=api_base, api_key=FAKE_TOKEN, **extras
    )
    guardrail: Final = Guardrail(guardrail_name="salt-security", litellm_params=params)
    assert InMemoryGuardrailHandler().initialize_guardrail(guardrail) is not None
    (callback,) = litellm.callbacks
    assert isinstance(callback, SaltSecurityGuardrail)
    return callback


def test_loads_from_config_with_defaults() -> None:
    callback: Final = _load()

    assert callback.endpoint_url == ENDPOINT
    assert callback.timeout == DEFAULT_TIMEOUT_SECONDS
    assert callback.enforcement_mode == "monitor"
    assert callback.block_tool_results is False
    assert callback.verify_tls is True
    assert callback.event_hook == "pre_call"
    assert callback.guardrail_name == "salt-security"


def test_trailing_slash_on_api_base_yields_one_slash() -> None:
    assert _load(api_base="https://ai.example/").endpoint_url == ENDPOINT


def test_supported_hooks() -> None:
    expected: Final = [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    assert SaltSecurityGuardrail.get_supported_event_hooks() == expected
    assert _load().supported_event_hooks == expected


def test_extras_are_read_from_config() -> None:
    callback: Final = _load(enforcement_mode="block", block_tool_results=True, verify_tls=False, timeout=1.5)

    assert callback.enforcement_mode == "block"
    assert callback.block_tool_results is True
    assert callback.verify_tls is False
    assert callback.timeout == 1.5


@pytest.mark.parametrize("value", [None, "Block", "bogus", ""])
def test_enforcement_mode_other_than_exact_block_is_monitor(value: object) -> None:
    assert _load(enforcement_mode=value).enforcement_mode == "monitor"


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("False", False), ("true", True), ("TRUE", True)])
def test_bool_settings_accept_strings(value: str, expected: bool) -> None:
    callback: Final = _load(verify_tls=value, block_tool_results=value)

    assert callback.verify_tls is expected
    assert callback.block_tool_results is expected


@pytest.mark.parametrize("key", ["verify_tls", "block_tool_results"])
@pytest.mark.parametrize("value", ["maybe", 1, None])
def test_bad_bool_setting_fails_at_load_naming_the_key(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        _load(**{key: value})


@pytest.mark.parametrize("api_base", [None, ""])
def test_missing_api_base_fails_at_load(api_base: str | None) -> None:
    with pytest.raises(ValueError, match="api_base"):
        _load(api_base=api_base)


def _sent_body(route: respx.Route) -> dict[str, object]:
    assert route.call_count == 1
    return cast("dict[str, object]", json.loads(route.calls.last.request.content))


@pytest.fixture
def salt_route() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"action": "NONE"}))
        yield router


async def test_posts_wire_body(salt_route: respx.MockRouter) -> None:
    messages: Final = [{"role": "user", "content": "hi"}]
    inputs: Final = cast(
        "GenericGuardrailAPIInputs", {"texts": ["hi"], "structured_messages": messages, "model": "gpt-4o"}
    )

    await _load().apply_guardrail(
        inputs,
        {},
        "request",
        cast("object", SimpleNamespace(litellm_call_id="call-1")),  # pyright: ignore[reportArgumentType]  # minimal logging stub
    )

    assert _sent_body(salt_route.routes[0]) == {
        "input_type": "request",
        "litellm_call_id": "call-1",
        "texts": ["hi"],
        "structured_messages": messages,
        "model": "gpt-4o",
        "additional_provider_specific_params": {"enforcement_mode": "monitor"},
    }


async def test_sends_token_header(salt_route: respx.MockRouter) -> None:
    await _load().apply_guardrail(cast("GenericGuardrailAPIInputs", {"texts": ["hi"]}), {}, "request")

    headers: Final = salt_route.routes[0].calls.last.request.headers
    assert headers["x-api-key"] == FAKE_TOKEN
    assert headers["content-type"] == "application/json"


@pytest.mark.parametrize(("configured", "requested"), [("block", "monitor"), ("monitor", "block")])
async def test_caller_cannot_override_mode(salt_route: respx.MockRouter, configured: str, requested: str) -> None:
    request_data: Final[dict[str, object]] = {
        "body": {"guardrails": {"salt-security": {"extra_body": {"enforcement_mode": requested}}}}
    }

    await _load(enforcement_mode=configured).apply_guardrail(
        cast("GenericGuardrailAPIInputs", {"texts": ["hi"]}), request_data, "request"
    )

    params: Final = cast("dict[str, object]", _sent_body(salt_route.routes[0])["additional_provider_specific_params"])
    assert params == {"enforcement_mode": configured}


async def test_missing_inputs_do_not_raise(salt_route: respx.MockRouter) -> None:
    inputs: Final = cast("GenericGuardrailAPIInputs", {})

    result: Final = await _load().apply_guardrail(inputs, {}, "response", None)

    body: Final = _sent_body(salt_route.routes[0])
    assert body["texts"] == []
    assert body["structured_messages"] == []
    assert body["litellm_call_id"] is None
    assert body["input_type"] == "response"
    assert result == inputs


SENTINEL_PROMPT: Final = "sentinel-prompt-text-7f3a"
SENTINEL_BODY: Final = "sentinel-response-body-9c1d"
INPUTS: Final = cast("GenericGuardrailAPIInputs", {"texts": [SENTINEL_PROMPT]})


def _answer(router: respx.MockRouter, response: httpx.Response | Exception) -> None:
    route: Final = router.routes[0]
    if isinstance(response, Exception):
        route.mock(side_effect=response)
    else:
        route.mock(return_value=response)


def _recorded(request_data: dict[str, object]) -> list[dict[str, object]]:
    metadata: Final = cast("dict[str, object]", request_data.get("metadata", {}))
    entries: Final = cast("list[object]", metadata.get("standard_logging_guardrail_information", []))
    return [dict(entry) for entry in entries]  # pyright: ignore[reportCallIssue,reportArgumentType]  # TypedDict entries


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []  # mutable-ok: collects emitted records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def test_blocked_raises_with_reason(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED", "blocked_reason": "policy X"}))

    with pytest.raises(GuardrailRaisedException) as raised:
        await _load().apply_guardrail(INPUTS, {}, "request")

    assert raised.value.message == "policy X"
    assert raised.value.blocked_content is True
    assert raised.value.guardrail_name == "salt-security"


@pytest.mark.parametrize(
    "answer",
    [{"action": "BLOCKED"}, {"action": "BLOCKED", "blocked_reason": ""}, {"action": "BLOCKED", "blocked_reason": 123}],
)
async def test_blocked_without_usable_reason_uses_default(
    salt_route: respx.MockRouter, answer: dict[str, object]
) -> None:
    _answer(salt_route, httpx.Response(200, json=answer))

    with pytest.raises(GuardrailRaisedException) as raised:
        await _load().apply_guardrail(INPUTS, {}, "request")

    assert raised.value.message == BLOCKED_MESSAGE


async def test_none_allows(salt_route: respx.MockRouter) -> None:
    assert await _load().apply_guardrail(INPUTS, {}, "request") == INPUTS


async def test_flagged_is_recorded(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "NONE", "flagged": True}))
    request_data: Final[dict[str, object]] = {}

    result: Final = await _load().apply_guardrail(INPUTS, request_data, "request")

    assert result == INPUTS
    (entry,) = _recorded(request_data)
    assert entry["guardrail_status"] == "guardrail_flagged"
    assert entry["guardrail_response"] == {"guardrail_flagged": True}


@pytest.mark.parametrize("flagged", ["yes", 1, None, False])
async def test_non_true_flagged_is_not_recorded_as_flagged(salt_route: respx.MockRouter, flagged: object) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "NONE", "flagged": flagged}))
    request_data: Final[dict[str, object]] = {}

    await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    assert all(entry["guardrail_status"] != "guardrail_flagged" for entry in _recorded(request_data))


@pytest.mark.parametrize("action", ["blocked", "ALLOW", None, ["BLOCKED"]])
async def test_unknown_action_allows(salt_route: respx.MockRouter, action: object) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": action, "blocked_reason": "policy X"}))

    assert await _load().apply_guardrail(INPUTS, {}, "request") == INPUTS


FAILURES: Final = [
    pytest.param(httpx.ReadTimeout("slow"), id="read-timeout"),
    pytest.param(httpx.ConnectError("down"), id="connect-error"),
    pytest.param(httpx.Response(401, json={"action": "BLOCKED"}), id="401"),
    pytest.param(httpx.Response(413, json={"action": "BLOCKED"}), id="413"),
    pytest.param(httpx.Response(500, json={"action": "BLOCKED"}), id="500"),
    pytest.param(httpx.Response(503, json={"action": "BLOCKED"}), id="503"),
    pytest.param(httpx.Response(200, content=b"not json"), id="not-json"),
    pytest.param(httpx.Response(200, json=[1]), id="list-body"),
    pytest.param(httpx.Response(200, json=None), id="null-body"),
]


@pytest.mark.parametrize("failure", FAILURES)
async def test_fails_open(salt_route: respx.MockRouter, failure: httpx.Response | Exception) -> None:
    _answer(salt_route, failure)

    assert await _load().apply_guardrail(INPUTS, {}, "request") == INPUTS


async def test_timeout_fails_open_with_configured_timeout(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.ReadTimeout("slow"))

    result: Final = await _load(timeout=1.5).apply_guardrail(INPUTS, {}, "request")

    assert result == INPUTS
    assert salt_route.routes[0].call_count == 1
    sent_timeout: Final = salt_route.routes[0].calls.last.request.extensions["timeout"]
    assert sent_timeout == {"connect": 1.5, "read": 1.5, "write": 1.5, "pool": 1.5}


async def test_failure_logs_no_secrets(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(500, content=SENTINEL_BODY.encode()))
    handler: Final = _ListHandler()
    previous_level: Final = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        await _load().apply_guardrail(INPUTS, {}, "request")
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(previous_level)

    assert handler.records
    text: Final = "\n".join(record.getMessage() for record in handler.records)
    for secret in (FAKE_TOKEN, SENTINEL_PROMPT, SENTINEL_BODY):
        assert secret not in text


async def test_failure_records_no_secrets(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(500, content=SENTINEL_BODY.encode()))
    request_data: Final[dict[str, object]] = {}

    await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    dumped: Final = json.dumps(request_data, default=str)
    for secret in (FAKE_TOKEN, SENTINEL_PROMPT, SENTINEL_BODY):
        assert secret not in dumped
