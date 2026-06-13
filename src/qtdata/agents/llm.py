"""Thin Anthropic client seam + usage/cost telemetry.

Tests inject `client=FakeAnthropic()`; production lazily constructs the real
client. Model calls follow the opus-4-8 contract: adaptive thinking, NO
temperature/top_p/top_k (they 400), prompt caching via cache_control on the
system block (set by callers).
"""

from __future__ import annotations

from dataclasses import dataclass

from qtdata.config import Settings

# $/Mtok — used for the per-case cost line, not billing truth
PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
}


@dataclass
class UsageMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    def add(self, usage: object) -> None:
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.calls += 1

    def cost_usd(self, model: str) -> float:
        p = PRICING_PER_MTOK.get(model, PRICING_PER_MTOK["claude-opus-4-8"])
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read_tokens * p["cache_read"]
            + self.cache_creation_tokens * p["cache_write"]
        ) / 1_000_000

    def summary_line(self, model: str) -> str:
        return (
            f"{self.calls} llamadas | in={self.input_tokens:,} out={self.output_tokens:,} "
            f"cache_read={self.cache_read_tokens:,} cache_write={self.cache_creation_tokens:,} "
            f"| ~${self.cost_usd(model):.4f}"
        )


class LLMClient:
    """Mockable seam around the anthropic SDK (lazy import, injected in tests)."""

    def __init__(self, settings: Settings, client: object | None = None):
        self.settings = settings
        self.model = settings.agent_model
        self.meter = UsageMeter()
        self._client = client

    @property
    def client(self) -> object:
        if self._client is None:
            if self.settings.agent_use_subscription:
                # Claude Pro/Max subscription via OAuth — no API key needed.
                from qtdata.agents.claude_subscription import build_subscription_client

                self._client = build_subscription_client(
                    self.settings.agent_credentials_path
                )
            else:
                import anthropic  # lazy: agents are optional at runtime

                key = self.settings.anthropic_api_key
                self._client = anthropic.Anthropic(
                    api_key=key.get_secret_value() if key else None  # None -> env var
                )
        return self._client

    def _prepare(self, kwargs: dict) -> dict:
        """Apply subscription OAuth transforms when billing the Claude plan."""
        if self.settings.agent_use_subscription:
            from qtdata.agents.claude_subscription import apply_subscription_transforms

            apply_subscription_transforms(kwargs)
        return kwargs

    def create(self, **kwargs):
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("max_tokens", 16000)
        kwargs.setdefault("thinking", {"type": "adaptive"})
        response = self.client.messages.create(**self._prepare(kwargs))
        self.meter.add(getattr(response, "usage", None))
        return response

    def parse(self, **kwargs):
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("max_tokens", 4000)
        kwargs.setdefault("thinking", {"type": "adaptive"})
        response = self.client.messages.parse(**self._prepare(kwargs))
        self.meter.add(getattr(response, "usage", None))
        return response
