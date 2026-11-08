from __future__ import annotations

from typing import Literal

from pydantic import Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel


class SaltSecurityGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    timeout: float | None = Field(
        default=3.0,
        gt=0.0,
        description="Timeout in seconds for the Salt Security check. On timeout the request is allowed.",
    )
    enforcement_mode: Literal["monitor", "block"] | None = Field(
        default="monitor",
        description="monitor records findings and allows the request. block rejects requests Salt Security flags.",
    )
    block_tool_results: bool | None = Field(
        default=False,
        description="Also block when the finding comes from a tool result in the conversation.",
    )
    verify_tls: bool | None = Field(
        default=True,
        description="Verify the TLS certificate of the Salt Security endpoint.",
    )


class SaltSecurityGuardrailConfigModel(GuardrailConfigModel[SaltSecurityGuardrailConfigModelOptionalParams]):
    api_key: str = Field(
        min_length=1,
        description="Salt Security API token, sent as the x-api-key header.",
    )
    api_base: str = Field(
        min_length=1,
        description="Salt Security ai-front base URL. Requests go to <api_base>/beta/litellm_basic_guardrail_api.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Salt Security"
