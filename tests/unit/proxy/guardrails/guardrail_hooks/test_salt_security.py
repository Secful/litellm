from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final, cast

import httpx
import pytest
import respx

import litellm
from litellm.proxy.guardrails.guardrail_hooks.salt_security import DEFAULT_TIMEOUT_SECONDS, SaltSecurityGuardrail
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams
from litellm.types.utils import GenericGuardrailAPIInputs

ENDPOINT: Final = "https://ai.example/beta/litellm_basic_guardrail_api"


@pytest.fixture(autouse=True)
def _isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])


def _load(api_base: str | None = "https://ai.example", **extras: object) -> SaltSecurityGuardrail:
    params: Final = LitellmParams(
        guardrail="salt_security", mode="pre_call", api_base=api_base, api_key="tok", **extras
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
    assert headers["x-api-key"] == "tok"
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
