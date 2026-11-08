from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Final, cast

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy import proxy_server
from litellm.proxy.guardrails.guardrail_hooks.salt_security import DEFAULT_TIMEOUT_SECONDS, SaltSecurityGuardrail
from litellm.proxy.guardrails.guardrail_hooks.salt_security.salt_security import BLOCKED_MESSAGE
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.salt_security import (
    SaltSecurityGuardrailConfigModelOptionalParams,
)
from litellm.types.utils import GenericGuardrailAPIInputs

ENDPOINT: Final = "https://ai.example/beta/litellm_basic_guardrail_api"
FAKE_TOKEN: Final = "fake-token-do-not-use-123"
COMPANY_ID: Final = "c-test"


@pytest.fixture(autouse=True)
def isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())


def _load_with(
    extras: Mapping[str, object],
    api_base: str | None = "https://ai.example",
    api_key: str | None = FAKE_TOKEN,
    mode: str | list[str] = "pre_call",
    company_id: str | None = COMPANY_ID,
) -> SaltSecurityGuardrail:
    company: Final = {} if company_id is None else {"company_id": company_id}
    params: Final = LitellmParams(
        guardrail="salt_security", mode=mode, api_base=api_base, api_key=api_key, **company, **extras
    )
    guardrail: Final = Guardrail(guardrail_name="salt-security", litellm_params=params)
    assert InMemoryGuardrailHandler().initialize_guardrail(guardrail) is not None
    (callback,) = litellm.callbacks
    assert isinstance(callback, SaltSecurityGuardrail)
    return callback


def _load(
    api_base: str | None = "https://ai.example",
    api_key: str | None = FAKE_TOKEN,
    mode: str | list[str] = "pre_call",
    company_id: str | None = COMPANY_ID,
    **extras: object,
) -> SaltSecurityGuardrail:
    return _load_with(extras, api_base, api_key, mode, company_id)


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


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("False", False), ("true", True), ("TRUE", True)])
def test_bool_settings_accept_strings(value: str, expected: bool) -> None:
    callback: Final = _load(verify_tls=value, block_tool_results=value)

    assert callback.verify_tls is expected
    assert callback.block_tool_results is expected


@pytest.mark.parametrize("key", ["verify_tls", "block_tool_results"])
@pytest.mark.parametrize("value", ["maybe", 1, None])
def test_bad_bool_setting_fails_at_load_naming_the_key(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        _load_with({key: value})


@pytest.mark.parametrize("api_base", [None, ""])
def test_missing_api_base_fails_at_load(api_base: str | None) -> None:
    with pytest.raises(ValueError, match="api_base"):
        _load(api_base=api_base)


def test_list_mode_is_loaded_as_event_hooks() -> None:
    callback: Final = _load(mode=["pre_call", "post_call"])

    assert callback.event_hook == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


def test_unknown_mode_fails_at_load() -> None:
    with pytest.raises(ValueError, match="mode"):
        _load(mode="pre_cal")


@pytest.mark.parametrize("api_key", [None, ""])
def test_missing_api_key_fails_at_load(api_key: str | None) -> None:
    with pytest.raises(ValueError, match="api_key"):
        _load(api_key=api_key)


@pytest.mark.parametrize("company_id", [None, "", "   "])
def test_missing_company_id_fails_at_load(company_id: str | None) -> None:
    with pytest.raises(ValueError, match="company_id"):
        _load(company_id=company_id)


@pytest.mark.parametrize("timeout", [0, 0.0, -1])
def test_non_positive_timeout_fails_at_load(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _load(timeout=timeout)


@pytest.mark.parametrize("field", ["verify_tls", "block_tool_results"])
def test_config_model_rejects_null_booleans(field: str) -> None:
    with pytest.raises(ValidationError):
        SaltSecurityGuardrailConfigModelOptionalParams.model_validate({field: None})


def test_config_model_accepts_the_modes_the_loader_accepts() -> None:
    assert SaltSecurityGuardrailConfigModelOptionalParams(enforcement_mode="BLOCK").enforcement_mode == "BLOCK"


def test_config_model_timeout_default_is_the_loader_default() -> None:
    assert SaltSecurityGuardrailConfigModelOptionalParams().timeout == DEFAULT_TIMEOUT_SECONDS


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
    assert headers["authorization"] == _basic(COMPANY_ID, FAKE_TOKEN)
    assert "x-api-key" not in headers
    assert headers["content-type"] == "application/json"


def _basic(company_id: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{company_id}:{token}".encode()).decode()


@pytest.mark.parametrize("token", ["abc:def", "a:b:c:", ":lead", "plain"])
async def test_basic_header_is_company_id_colon_token(salt_route: respx.MockRouter, token: str) -> None:
    await _load(api_key=token, company_id="acme").apply_guardrail(INPUTS, {}, "request")

    assert salt_route.routes[0].calls.last.request.headers["authorization"] == _basic("acme", token)
    decoded: Final = base64.b64decode(salt_route.routes[0].calls.last.request.headers["authorization"][6:]).decode()
    assert decoded.split(":", 1) == ["acme", token]


@pytest.fixture
def premium_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", True)


def _dynamic_params(guardrail: SaltSecurityGuardrail, request_data: Mapping[str, object]) -> Mapping[str, object]:
    return cast(
        "Mapping[str, object]",
        guardrail.get_guardrail_dynamic_request_body_params(dict(request_data)),  # pyright: ignore[reportUnknownMemberType]  # upstream returns a bare dict
    )


def _caller_extra_body(**extra_body: object) -> dict[str, object]:
    return {"guardrails": [{"salt-security": {"extra_body": extra_body}}]}


def test_token_is_absent_from_repr_and_str() -> None:
    guardrail: Final = _load()

    encoded: Final = base64.b64encode(f"{COMPANY_ID}:{FAKE_TOKEN}".encode()).decode()
    for rendered in (repr(guardrail), str(guardrail)):
        assert FAKE_TOKEN not in rendered
        assert encoded not in rendered


async def test_token_header_is_sent_on_every_call(salt_route: respx.MockRouter) -> None:
    guardrail: Final = _load()
    sent: Final[list[str]] = []

    for direction in ("request", "response"):
        await guardrail.apply_guardrail(INPUTS, {}, direction)
        sent.append(salt_route.routes[0].calls.last.request.headers["authorization"])

    assert sent == [_basic(COMPANY_ID, FAKE_TOKEN)] * 2
    assert salt_route.routes[0].call_count == 2


@pytest.mark.parametrize(("configured", "requested"), [("block", "monitor"), ("monitor", "block")])
async def test_caller_cannot_override_mode(
    salt_route: respx.MockRouter, premium_proxy: None, configured: str, requested: str
) -> None:
    request_data: Final = _caller_extra_body(enforcement_mode=requested)
    guardrail: Final = _load(enforcement_mode=configured)
    assert _dynamic_params(guardrail, request_data) == {"enforcement_mode": requested}

    await guardrail.apply_guardrail(cast("GenericGuardrailAPIInputs", {"texts": ["hi"]}), request_data, "request")

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("block", "block"),
        ("BLOCK", "block"),
        (" block ", "block"),
        ("monitor", "monitor"),
        ("", "monitor"),
        (None, "monitor"),
        ("enforce", "monitor"),
        (5, "monitor"),
    ],
)
async def test_mode_normalisation(salt_route: respx.MockRouter, value: object, expected: str) -> None:
    await _load(enforcement_mode=value).apply_guardrail(INPUTS, {}, "request")

    params: Final = cast("dict[str, object]", _sent_body(salt_route.routes[0])["additional_provider_specific_params"])
    assert params == {"enforcement_mode": expected}


def test_default_mode_is_pre_call() -> None:
    guardrail: Final = SaltSecurityGuardrail(
        api_base="https://ai.example",
        company_id=COMPANY_ID,
        api_key=FAKE_TOKEN,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        enforcement_mode="monitor",
        block_tool_results=False,
        verify_tls=True,
        guardrail_name="salt-security",
    )

    assert guardrail.event_hook == GuardrailEventHooks.pre_call


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
    return [dict(cast("Mapping[str, object]", entry)) for entry in entries]


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


async def test_flagged_is_recorded_once_with_keyword_call(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "NONE", "flagged": True}))
    request_data: Final[dict[str, object]] = {}

    result: Final = await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    assert result == INPUTS
    (entry,) = _recorded(request_data)
    assert entry["guardrail_status"] == "guardrail_flagged"


async def test_block_wins_over_flagged(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED", "flagged": True}))
    request_data: Final[dict[str, object]] = {}

    with pytest.raises(GuardrailRaisedException):
        await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    assert all(entry["guardrail_status"] != "guardrail_flagged" for entry in _recorded(request_data))


def _with_newest(role: str) -> GenericGuardrailAPIInputs:
    messages: Final = [{"role": "user", "content": "hi"}, {"role": role, "content": "result"}]
    return cast("GenericGuardrailAPIInputs", {"texts": ["result"], "structured_messages": messages})


async def test_tool_role_block_is_not_enforced_by_default(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED", "blocked_reason": "policy X"}))
    inputs: Final = _with_newest("tool")
    request_data: Final[dict[str, object]] = {}

    result: Final = await _load().apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")

    assert result == inputs
    (entry,) = _recorded(request_data)
    assert entry["guardrail_status"] == "guardrail_flagged"
    params: Final = cast("dict[str, object]", _sent_body(salt_route.routes[0])["additional_provider_specific_params"])
    assert params == {"enforcement_mode": "monitor"}


async def test_caller_cannot_enable_tool_block(salt_route: respx.MockRouter, premium_proxy: None) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED"}))
    inputs: Final = _with_newest("tool")
    request_data: Final = _caller_extra_body(block_tool_results=True)
    guardrail: Final = _load()
    assert _dynamic_params(guardrail, request_data) == {"block_tool_results": True}

    result: Final = await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")

    assert result == inputs


@pytest.mark.parametrize(
    "messages",
    [["tool"], [None], [{"content": "no role"}], [{"role": None}], []],
    ids=["str", "none", "no-role", "role-none", "empty"],
)
async def test_malformed_newest_message_still_blocks(salt_route: respx.MockRouter, messages: list[object]) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED"}))
    inputs: Final = cast("GenericGuardrailAPIInputs", {"texts": ["hi"], "structured_messages": messages})

    with pytest.raises(GuardrailRaisedException):
        await _load().apply_guardrail(inputs, {}, "request")


async def test_tool_role_block_enforced_when_switch_on(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED", "blocked_reason": "policy X"}))

    with pytest.raises(GuardrailRaisedException) as raised:
        await _load(block_tool_results=True).apply_guardrail(_with_newest("tool"), {}, "request")

    assert raised.value.message == "policy X"


@pytest.mark.parametrize("inputs", [_with_newest("user"), cast("GenericGuardrailAPIInputs", {"texts": ["hi"]})])
async def test_non_tool_block_ignores_switch(salt_route: respx.MockRouter, inputs: GenericGuardrailAPIInputs) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED"}))

    with pytest.raises(GuardrailRaisedException):
        await _load().apply_guardrail(inputs, {}, "request")


async def test_tool_message_before_newest_user_message_still_blocks(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": "BLOCKED"}))
    messages: Final = [{"role": "tool", "content": "old"}, {"role": "user", "content": "now"}]
    inputs: Final = cast("GenericGuardrailAPIInputs", {"structured_messages": messages})

    with pytest.raises(GuardrailRaisedException):
        await _load().apply_guardrail(inputs, {}, "request")


@pytest.mark.parametrize("action", ["BLOCKED", "NONE"])
async def test_response_direction(salt_route: respx.MockRouter, action: str) -> None:
    _answer(salt_route, httpx.Response(200, json={"action": action}))
    guardrail: Final = _load()

    if action == "BLOCKED":
        with pytest.raises(GuardrailRaisedException):
            await guardrail.apply_guardrail(INPUTS, {}, "response")
    else:
        assert await guardrail.apply_guardrail(INPUTS, {}, "response") == INPUTS

    assert _sent_body(salt_route.routes[0])["input_type"] == "response"


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


@pytest.mark.parametrize("failure", FAILURES)
async def test_fail_open_is_recorded_as_failed_to_respond(
    salt_route: respx.MockRouter, failure: httpx.Response | Exception
) -> None:
    _answer(salt_route, failure)
    request_data: Final[dict[str, object]] = {}

    result: Final = await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    assert result == INPUTS
    (entry,) = _recorded(request_data)
    assert entry["guardrail_status"] == "guardrail_failed_to_respond"


async def test_normal_allow_is_recorded_as_success(salt_route: respx.MockRouter) -> None:
    request_data: Final[dict[str, object]] = {"model": "gpt-4o"}

    await _load().apply_guardrail(inputs=INPUTS, request_data=request_data, input_type="request")

    (entry,) = _recorded(request_data)
    assert entry["guardrail_status"] == "success"


async def test_timeout_fails_open_with_configured_timeout(salt_route: respx.MockRouter) -> None:
    _answer(salt_route, httpx.ReadTimeout("slow"))

    result: Final = await _load(timeout=1.5).apply_guardrail(INPUTS, {}, "request")

    assert result == INPUTS
    assert salt_route.routes[0].call_count == 1
    sent_timeout: Final = cast("dict[str, float]", salt_route.routes[0].calls.last.request.extensions["timeout"])
    assert sent_timeout == {"connect": 1.5, "read": 1.5, "write": 1.5, "pool": 1.5}


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.Response(500, content=SENTINEL_BODY.encode()), id="500-body"),
        pytest.param(httpx.ConnectError(f"cannot reach {FAKE_TOKEN} {SENTINEL_PROMPT}"), id="exception-text"),
        pytest.param(httpx.ReadTimeout(f"slow {FAKE_TOKEN} {SENTINEL_PROMPT} {SENTINEL_BODY}"), id="timeout-text"),
    ],
)
async def test_failure_logs_no_secrets(salt_route: respx.MockRouter, failure: httpx.Response | Exception) -> None:
    _answer(salt_route, failure)
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


@pytest.mark.parametrize("verify_tls", [True, False])
async def test_verify_tls_setting_reaches_the_client(verify_tls: bool) -> None:
    guardrail: Final = _load(verify_tls=verify_tls)

    assert guardrail._client.ssl_verify is verify_tls  # pyright: ignore[reportPrivateUsage]  # asserting the shared client's configuration


async def test_timeout_setting_reaches_the_client() -> None:
    guardrail: Final = _load(timeout=1.5)

    assert guardrail._client.timeout == 1.5  # pyright: ignore[reportPrivateUsage]  # asserting the shared client's configuration


def _build(verify_tls: bool) -> SaltSecurityGuardrail:
    return SaltSecurityGuardrail(
        api_base="https://ai.example",
        company_id=COMPANY_ID,
        api_key=FAKE_TOKEN,
        timeout=2.0,
        enforcement_mode="monitor",
        block_tool_results=False,
        verify_tls=verify_tls,
        guardrail_name="salt-security",
    )


async def test_guardrails_with_different_settings_get_different_clients() -> None:
    strict: Final = _build(verify_tls=True)
    lenient: Final = _build(verify_tls=False)

    assert strict._client is not lenient._client  # pyright: ignore[reportPrivateUsage]  # comparing client identity


def _warnings_while_loading(extras: Mapping[str, object]) -> list[str]:
    handler: Final = _ListHandler()
    previous_level: Final = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        _load_with(extras)
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(previous_level)
    return [record.getMessage() for record in handler.records if record.levelno == logging.WARNING]


def test_verify_disabled_warns_once_without_secrets() -> None:
    warnings: Final = _warnings_while_loading({"verify_tls": False})

    assert len(warnings) == 1
    assert "verify_tls" in warnings[0]
    assert FAKE_TOKEN not in warnings[0]
    assert "ai.example" not in warnings[0]


def test_verify_enabled_does_not_warn() -> None:
    assert _warnings_while_loading({"verify_tls": True}) == []
