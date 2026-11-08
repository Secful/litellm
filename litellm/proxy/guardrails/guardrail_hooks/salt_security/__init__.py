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


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> CustomGuardrail:
    import litellm

    extras: Final = litellm_params.model_extra or _NO_EXTRAS
    _callback: Final = SaltSecurityGuardrail(
        api_base=litellm_params.api_base or "",
        api_key=litellm_params.api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS if litellm_params.timeout is None else litellm_params.timeout,
        enforcement_mode=str(extras.get("enforcement_mode", "monitor")),
        block_tool_results=extras.get("block_tool_results", False) is True,
        verify_tls=extras.get("verify_tls", True) is not False,
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
