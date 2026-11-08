from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.salt_security import SaltSecurityGuardrailConfigModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs

DEFAULT_TIMEOUT_SECONDS: Final = 3.0
ENDPOINT_PATH: Final = "/beta/litellm_basic_guardrail_api"
BLOCKED_MESSAGE: Final = "Request blocked by security policy."
_NO_ANSWER: Final[Mapping[str, object]] = MappingProxyType({})


def normalise_mode(value: object) -> Literal["monitor", "block"]:
    if isinstance(value, str) and value.strip().lower() == "block":
        return "block"
    return "monitor"


def _newest_is_tool_message(inputs: GenericGuardrailAPIInputs) -> bool:
    messages: Final = inputs.get("structured_messages")
    if not messages:
        return False
    newest: Final[object] = messages[-1]
    return isinstance(newest, Mapping) and newest.get("role") == "tool"


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
        api_key: str | None,
        timeout: float,
        enforcement_mode: object,
        block_tool_results: bool,
        verify_tls: bool,
        event_hook: object = None,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail
    ) -> None:
        self.endpoint_url: Final = api_base.rstrip("/") + ENDPOINT_PATH
        self.api_key: Final = api_key
        self.enforcement_mode: Final = normalise_mode(enforcement_mode)
        self.block_tool_results: Final = block_tool_results
        self.verify_tls: Final = verify_tls
        self._client: httpx.AsyncClient | None = None
        super().__init__(
            timeout=timeout,
            event_hook=GuardrailEventHooks.pre_call if event_hook is None else event_hook,  # pyright: ignore[reportArgumentType]  # CustomGuardrail event_hook union is wider than object
            supported_event_hooks=self.get_supported_event_hooks(),
            **kwargs,  # pyright: ignore[reportArgumentType]  # CustomGuardrail.__init__ kwargs are untyped
        )

    @staticmethod
    def get_config_model() -> type[SaltSecurityGuardrailConfigModel]:
        return SaltSecurityGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, verify=self.verify_tls)
        return self._client

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail.apply_guardrail contract
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        call_id: Final = getattr(logging_obj, "litellm_call_id", None)
        body: Final = request_payload(inputs, input_type, call_id, self.enforcement_mode)
        headers: Final = {"content-type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        answer: Final = await self._ask(body, headers)
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
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={"guardrail_flagged": True},
                request_data=request_data,
                guardrail_status="guardrail_flagged",
            )
        return inputs

    async def _ask(self, body: Mapping[str, object], headers: Mapping[str, str]) -> Mapping[str, object]:
        try:
            response: Final = await self._http_client().post(self.endpoint_url, json=body, headers=dict(headers))
            if not response.is_success:
                verbose_proxy_logger.warning(
                    "salt_security: allowing request, endpoint returned %s", response.status_code
                )
                return _NO_ANSWER
            answer: Final = response.json()
        except Exception as exc:
            verbose_proxy_logger.warning("salt_security: allowing request, check failed with %s", type(exc).__name__)
            return _NO_ANSWER
        if not isinstance(answer, dict):
            verbose_proxy_logger.warning("salt_security: allowing request, endpoint returned a non-object body")
            return _NO_ANSWER
        return cast("Mapping[str, object]", answer)
