from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.salt_security import DEFAULT_TIMEOUT_SECONDS

from .salt_security import ENDPOINT_PATH, EventHook, SaltSecurityGuardrail

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})
_EVENT_HOOK_ADAPTER: Final[TypeAdapter[EventHook]] = TypeAdapter(EventHook)
_BOOL_STRINGS: Final = MappingProxyType({"true": True, "false": False})


def _bool_setting(extras: Mapping[str, object], key: str, default: bool) -> bool:
    value: Final = extras.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _BOOL_STRINGS:
        return _BOOL_STRINGS[value.lower()]
    raise ValueError(f"salt_security: {key} must be true or false")


def _extras_of(litellm_params: LitellmParams) -> Mapping[str, object]:
    return litellm_params.model_extra or _NO_EXTRAS


def _name_of(guardrail: Mapping[str, object]) -> str:
    name: Final = guardrail.get("guardrail_name", "")
    return name if isinstance(name, str) else ""


def extras_company_id(litellm_params: LitellmParams) -> str:
    value: Final = _extras_of(litellm_params).get("company_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("salt_security: company_id is required")
    return value.strip()


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> CustomGuardrail:
    import litellm

    if not litellm_params.api_base:
        raise ValueError("salt_security: api_base is required")
    company_id: Final = extras_company_id(litellm_params)
    if not litellm_params.api_key:
        raise ValueError("salt_security: api_key is required")
    timeout: Final = DEFAULT_TIMEOUT_SECONDS if litellm_params.timeout is None else litellm_params.timeout
    if timeout <= 0:
        raise ValueError("salt_security: timeout must be greater than 0")
    extras: Final = _extras_of(litellm_params)
    _callback: Final = SaltSecurityGuardrail(
        api_base=litellm_params.api_base,
        company_id=company_id,
        api_key=litellm_params.api_key,
        timeout=timeout,
        enforcement_mode=extras.get("enforcement_mode"),
        block_tool_results=_bool_setting(extras, "block_tool_results", False),
        verify_tls=_bool_setting(extras, "verify_tls", True),
        guardrail_name=_name_of(guardrail),
        event_hook=_EVENT_HOOK_ADAPTER.validate_python(litellm_params.mode),
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.SALT_SECURITY.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.SALT_SECURITY.value: SaltSecurityGuardrail,
}

__all__ = ("DEFAULT_TIMEOUT_SECONDS", "ENDPOINT_PATH", "SaltSecurityGuardrail", "initialize_guardrail")
