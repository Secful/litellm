from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .salt_security import DEFAULT_TIMEOUT_SECONDS, ENDPOINT_PATH, SaltSecurityGuardrail

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})
_BOOL_STRINGS: Final = MappingProxyType({"true": True, "false": False})


def _bool_setting(extras: Mapping[str, object], key: str, default: bool) -> bool:
    value: Final = extras.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _BOOL_STRINGS:
        return _BOOL_STRINGS[value.lower()]
    raise ValueError(f"salt_security: {key} must be true or false")


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> CustomGuardrail:
    import litellm

    if not litellm_params.api_base:
        raise ValueError("salt_security: api_base is required")
    extras: Final = litellm_params.model_extra or _NO_EXTRAS
    _callback: Final = SaltSecurityGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS if litellm_params.timeout is None else litellm_params.timeout,
        enforcement_mode="block" if extras.get("enforcement_mode") == "block" else "monitor",
        block_tool_results=_bool_setting(extras, "block_tool_results", False),
        verify_tls=_bool_setting(extras, "verify_tls", True),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
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
