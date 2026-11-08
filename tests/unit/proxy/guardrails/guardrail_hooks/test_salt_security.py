from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm.proxy.guardrails.guardrail_hooks.salt_security import DEFAULT_TIMEOUT_SECONDS, SaltSecurityGuardrail
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams

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
