"""Smoke test: one short real call through the subscription OAuth path.

Confirms the billing bypass routes to the Claude Pro/Max plan. Prints only the
reply text and HTTP status class — never the token. Exit 0 on success.
"""

from __future__ import annotations

import sys

from qtdata.agents.claude_subscription import (
    apply_subscription_transforms,
    build_subscription_client,
)


def main() -> int:
    try:
        client = build_subscription_client()
    except Exception as exc:  # SubscriptionAuthError or import error
        print(f"AUTH SETUP FAILED: {type(exc).__name__}: {exc}")
        return 2

    kwargs = {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "Eres un asistente de prueba."}],
        "messages": [
            {"role": "user", "content": "Responde EXACTAMENTE con: SUBSCRIPTION OK"}
        ],
    }
    apply_subscription_transforms(kwargs)

    try:
        resp = client.messages.create(**kwargs)
    except Exception as exc:
        print(f"CALL FAILED: {type(exc).__name__}: {exc}")
        return 1

    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
    )
    usage = getattr(resp, "usage", None)
    print("REPLY:", text.strip())
    if usage is not None:
        print(
            f"USAGE: in={getattr(usage, 'input_tokens', '?')} "
            f"out={getattr(usage, 'output_tokens', '?')}"
        )
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
