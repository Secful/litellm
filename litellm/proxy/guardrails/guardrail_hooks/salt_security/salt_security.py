from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.salt_security import (
    SaltSecurityGuardrailConfigModel,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

ENDPOINT_PATH: Final = "/beta/litellm_basic_guardrail_api"
EventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode
BLOCKED_MESSAGE: Final = "Request blocked by security policy."
_ANSWER_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _CheckFailed:
    reason: str


def normalise_mode(value: object) -> Literal["monitor", "block"]:
    if isinstance(value, str) and value.strip().lower() == "block":
        return "block"
    return "monitor"


def _is_tool_message(message: object) -> bool:
    if not isinstance(message, Mapping):
        return False
    fields: Final = cast("Mapping[str, object]", message)  # cast-ok: isinstance leaves Mapping[Unknown, Unknown]
    return fields.get("role") == "tool"


def _newest_is_tool_message(inputs: GenericGuardrailAPIInputs) -> bool:
    messages: Final[Sequence[object] | None] = inputs.get("structured_messages")
    return bool(messages) and _is_tool_message(messages[-1])


def request_payload(
    inputs: GenericGuardrailAPIInputs,
    input_type: Literal["request", "response"],
    call_id: str | None,
    enforcement_mode: str,
) -> Mapping[str, object]:
    return {
        "input_type": input_type,
        "litellm_call_id": call_id,
        "texts": inputs.get("texts", []),
        "structured_messages": inputs.get("structured_messages", []),
        "model": inputs.get("model"),
        "additional_provider_specific_params": {"enforcement_mode": enforcement_mode},
    }


class SaltSecurityGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        timeout: float,
        enforcement_mode: object,
        block_tool_results: bool,
        verify_tls: bool,
        event_hook: EventHook | None = None,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail
    ) -> None:
        self.endpoint_url: Final = api_base.rstrip("/") + ENDPOINT_PATH
        self._headers: Final = MappingProxyType({"content-type": "application/json", "x-api-key": api_key})
        self.enforcement_mode: Final = normalise_mode(enforcement_mode)
        self.block_tool_results: Final = block_tool_results
        self.verify_tls: Final = verify_tls
        self._client: httpx.AsyncClient | None = None
        if not verify_tls:
            verbose_proxy_logger.warning(
                "salt_security: verify_tls is false, TLS certificates are not verified for the guardrail endpoint"
            )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # parent takes untyped **kwargs
            timeout=timeout,
            event_hook=GuardrailEventHooks.pre_call if event_hook is None else event_hook,
            supported_event_hooks=self.get_supported_event_hooks(),
            **kwargs,  # pyright: ignore[reportArgumentType]  # CustomGuardrail.__init__ kwargs are untyped
        )

    @staticmethod
    def get_config_model() -> type[SaltSecurityGuardrailConfigModel]:
        return SaltSecurityGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: overrides a base method typed list
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, verify=self.verify_tls)
        return self._client

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail.apply_guardrail contract
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        call_id: Final = getattr(logging_obj, "litellm_call_id", None)
        body: Final = request_payload(inputs, input_type, call_id, self.enforcement_mode)
        answer: Final = await self._ask(body)
        if isinstance(answer, _CheckFailed):
            self._record(request_data, {"error": answer.reason}, "guardrail_failed_to_respond")
            return inputs
        blocked: Final = answer.get("action") == "BLOCKED"
        downgraded: Final = blocked and not self.block_tool_results and _newest_is_tool_message(inputs)
        if blocked and not downgraded:
            reason: Final = answer.get("blocked_reason")
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=reason if isinstance(reason, str) and reason else BLOCKED_MESSAGE,
                should_wrap_with_default_message=False,
                blocked_content=True,
            )
        if downgraded or answer.get("flagged") is True:
            self._record(request_data, {"guardrail_flagged": True}, "guardrail_flagged")
        return inputs

    def _record(
        self,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail contract
        response: Mapping[str, object],
        status: GuardrailStatus,
    ) -> None:
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]  # the method's request_data param is an untyped dict upstream
            guardrail_json_response=response,
            request_data=request_data,
            guardrail_status=status,
        )

    async def _ask(self, body: Mapping[str, object]) -> Mapping[str, object] | _CheckFailed:
        try:
            response: Final = await self._http_client().post(self.endpoint_url, json=body, headers=self._headers)
            if not response.is_success:
                verbose_proxy_logger.warning(
                    "salt_security: allowing request, endpoint returned %s", response.status_code
                )
                return _CheckFailed(f"http_{response.status_code}")
            return MappingProxyType(_ANSWER_ADAPTER.validate_json(response.content))
        except Exception as exc:
            verbose_proxy_logger.warning("salt_security: allowing request, check failed with %s", type(exc).__name__)
            return _CheckFailed(type(exc).__name__)
