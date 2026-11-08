from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal

import httpx

from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.salt_security import SaltSecurityGuardrailConfigModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs

DEFAULT_TIMEOUT_SECONDS: Final = 3.0
ENDPOINT_PATH: Final = "/beta/litellm_basic_guardrail_api"


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
        enforcement_mode: str,
        block_tool_results: bool,
        verify_tls: bool,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail
    ) -> None:
        self.endpoint_url: Final = api_base.rstrip("/") + ENDPOINT_PATH
        self.api_key: Final = api_key
        self.enforcement_mode: Final = enforcement_mode
        self.block_tool_results: Final = block_tool_results
        self.verify_tls: Final = verify_tls
        self._client: httpx.AsyncClient | None = None
        super().__init__(
            timeout=timeout,
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
        await self._http_client().post(self.endpoint_url, json=body, headers=headers)
        return inputs
