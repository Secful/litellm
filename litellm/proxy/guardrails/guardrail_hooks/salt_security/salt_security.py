from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # upstream params annotation is a bare dict
    httpxSpecialProvider,
)
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


def basic_authorization(company_id: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{company_id}:{token}".encode()).decode()


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
        company_id: str,
        api_key: str,
        timeout: float,
        enforcement_mode: object,
        verify_tls: bool,
        event_hook: EventHook | None = None,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail
    ) -> None:
        self.endpoint_url: Final = api_base.rstrip("/") + ENDPOINT_PATH
        self._headers: Final = MappingProxyType(
            {"content-type": "application/json", "authorization": basic_authorization(company_id, api_key)}
        )
        self.enforcement_mode: Final = normalise_mode(enforcement_mode)
        self.verify_tls: Final = verify_tls
        self._client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": timeout, "ssl_verify": verify_tls},
        )
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
        if blocked:
            reason: Final = answer.get("blocked_reason")
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=reason if isinstance(reason, str) and reason else BLOCKED_MESSAGE,
                should_wrap_with_default_message=False,
                blocked_content=True,
            )
        if answer.get("flagged") is True:
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
            response: Final = await self._client.post(  # pyright: ignore[reportUnknownMemberType]  # upstream params are bare dicts
                self.endpoint_url, json=dict(body), headers=dict(self._headers)
            )
            return MappingProxyType(_ANSWER_ADAPTER.validate_json(response.content))
        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.warning(
                "salt_security: allowing request, endpoint returned %s", exc.response.status_code
            )
            return _CheckFailed(f"http_{exc.response.status_code}")
        except Exception as exc:
            verbose_proxy_logger.warning("salt_security: allowing request, check failed with %s", type(exc).__name__)
            return _CheckFailed(type(exc).__name__)
