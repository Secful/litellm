from __future__ import annotations

from typing import Final

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel

DEFAULT_TIMEOUT_SECONDS: Final = 3.0


class SaltSecurityGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    timeout: float | None = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0.0,
        description="Timeout in seconds for the Salt Security check. On timeout the request is allowed.",
    )
    enforcement_mode: str | None = Field(
        default="monitor",
        description=(
            "monitor records findings and allows the request. block rejects requests Salt Security flags. "
            "Matched case-insensitively, any other value means monitor."
        ),
    )
    verify_tls: bool = Field(
        default=True,
        description="Verify the TLS certificate of the Salt Security endpoint.",
    )


class SaltSecurityGuardrailConfigModel(GuardrailConfigModel[SaltSecurityGuardrailConfigModelOptionalParams]):
    company_id: str = Field(
        min_length=1,
        description="Salt Security company id, the username of the HTTP Basic credential.",
    )
    api_key: str = Field(
        min_length=1,
        description="Salt Security API token, the password of the HTTP Basic credential.",
    )
    api_base: str = Field(
        min_length=1,
        description="Salt Security ai-front base URL. Requests go to <api_base>/beta/litellm_basic_guardrail_api.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Salt Security"
