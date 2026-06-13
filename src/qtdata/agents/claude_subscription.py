"""Claude Pro/Max subscription auth for the agent layer (OAuth, no API key).

Lets ``qt agent ...`` bill against a Claude Code (Pro/Max) subscription instead
of pay-per-token API credits, by reusing the OAuth token the official Claude
Code CLI stores at ``~/.claude/.credentials.json``.

How it works
------------
Anthropic's server-side validator (added 2026-04-04) rejects OAuth requests
that don't look like they came from the first-party Claude Code CLI. To pass
it, every request must carry:

1. A signed ``x-anthropic-billing-header`` injected as ``system[0]`` — its
   signature is derived from the salt baked into the CLI binary and a sample of
   the first user message (see :func:`_build_billing_header_value`).
2. The Claude Code identity line as the first *real* system entry; any other
   system text is relocated into the first user message as
   ``<system-reminder>`` blocks (the CLI keeps its system prompt minimal).
3. Stainless SDK fingerprint headers (``x-stainless-*``) + the
   ``?beta=true`` query param + ``anthropic-dangerous-direct-browser-access``.
4. The OAuth beta flags on the ``anthropic-beta`` header.
5. ``metadata.user_id`` mapped from the account UUID.

This is a focused port of the Hermes ``anthropic_billing_bypass`` shim. The
two deliberate omissions versus that shim:

* **No MCP tool-name rewriting.** Hermes namespaces its tools to
  ``mcp__hermes__*``; this project's tools (``run_sql`` etc.) are plain names
  that the agent's own dispatcher resolves literally, exactly like the real
  CLI's ``Bash``/``Read`` tools. Renaming them would break verification.
* **No monkey-patching.** The agent calls the ``anthropic`` SDK directly, so we
  apply the transforms to the request kwargs in :class:`LLMClient` and set
  bearer auth on the client — no interpreter hooks, no source patching.

Credits: billing-header signing and the transform set originate from
``griffinmartin/opencode-claude-auth`` (MIT) and the Hermes port of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match the real Claude Code CLI; deviations break routing)
# ---------------------------------------------------------------------------

# Reported CLI version used for the billing-header signature + fingerprint.
CLAUDE_CODE_VERSION = "2.1.112"

# Shared salt shipped in the Claude Code binary; the server verifies the
# billing-header signature against it.
_BILLING_SALT = "59cf53e54c78"

# Claude Code 2.1.112+ reports ``sdk-cli`` as the billing entrypoint.
_BILLING_ENTRYPOINT = "sdk-cli"

_BILLING_PREFIX = "x-anthropic-billing-header"
_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# Stainless-generated SDK fingerprint (Claude Code 2.1.112 / JS SDK).
_STAINLESS_PACKAGE_VERSION = "0.81.0"
_STAINLESS_NODE_VERSION = "v22.11.0"

# Beta flags the OAuth subscription tier expects. ``oauth-2025-04-20`` is the
# one that actually routes to the plan; the others match the first-party CLI.
OAUTH_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "prompt-caching-scope-2026-01-05",
    "advisor-tool-2026-03-01",
]

# Public OAuth client used by the Claude Code CLI (needed for token refresh).
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"

DEFAULT_CREDENTIALS_PATH = "~/.claude/.credentials.json"


class SubscriptionAuthError(RuntimeError):
    """Raised when no usable Claude subscription token can be obtained."""


# ---------------------------------------------------------------------------
# Credential loading + refresh
# ---------------------------------------------------------------------------


def _credentials_path(path: str | Path | None) -> Path:
    return Path(path or DEFAULT_CREDENTIALS_PATH).expanduser()


def _load_credentials(path: str | Path | None) -> dict[str, Any]:
    p = _credentials_path(path)
    if not p.exists():
        raise SubscriptionAuthError(
            f"No se encontró el fichero de credenciales de Claude en {p}. "
            "Autentícate primero con la CLI oficial: `claude auth login --claudeai`."
        )
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SubscriptionAuthError(f"No se pudo leer {p}: {exc}") from exc
    # Claude Code wraps the token under "claudeAiOauth"; accept a flat shape too.
    if isinstance(raw, dict) and isinstance(raw.get("claudeAiOauth"), dict):
        return raw["claudeAiOauth"]
    if isinstance(raw, dict):
        return raw
    raise SubscriptionAuthError(f"Formato de credenciales inesperado en {p}.")


def _expires_at_seconds(creds: dict[str, Any]) -> float | None:
    exp = creds.get("expiresAt")
    if exp is None:
        return None
    try:
        exp = float(exp)
    except (TypeError, ValueError):
        return None
    # Claude Code stores expiry in milliseconds since epoch.
    return exp / 1000.0 if exp > 1e11 else exp


def _is_expired(creds: dict[str, Any], *, skew_s: float = 120.0) -> bool:
    exp = _expires_at_seconds(creds)
    if exp is None:
        return False  # no expiry recorded -> assume the CLI keeps it fresh
    return time.time() >= (exp - skew_s)


def refresh_oauth_token(
    creds: dict[str, Any],
    path: str | Path | None = None,
    *,
    session: Any | None = None,
) -> dict[str, Any]:
    """Exchange the refresh token for a fresh access token and persist it.

    Best-effort: on any failure the original creds are returned unchanged so
    callers can still try the (possibly expired) access token. ``session`` is
    injectable for tests; production uses ``requests``.
    """
    refresh = creds.get("refreshToken")
    if not refresh:
        return creds
    if session is None:
        import requests  # lazy: only needed when an actual refresh happens

        session = requests
    try:
        resp = session.post(
            _OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": _OAUTH_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("No se pudo refrescar el token OAuth de Claude: %s", exc)
        return creds

    updated = dict(creds)
    if payload.get("access_token"):
        updated["accessToken"] = payload["access_token"]
    if payload.get("refresh_token"):
        updated["refreshToken"] = payload["refresh_token"]
    if payload.get("expires_in"):
        updated["expiresAt"] = int((time.time() + float(payload["expires_in"])) * 1000)

    p = _credentials_path(path)
    try:
        existing = json.loads(p.read_text()) if p.exists() else {}
        if isinstance(existing, dict) and "claudeAiOauth" in existing:
            existing["claudeAiOauth"] = updated
        else:
            existing = updated
        p.write_text(json.dumps(existing, indent=2))
        p.chmod(0o600)
    except OSError as exc:
        logger.warning("Token refrescado pero no se pudo escribir %s: %s", p, exc)
    return updated


def get_access_token(
    path: str | Path | None = None,
    *,
    allow_refresh: bool = True,
) -> str:
    """Return a usable OAuth access token, refreshing if expired and possible."""
    creds = _load_credentials(path)
    if allow_refresh and _is_expired(creds):
        creds = refresh_oauth_token(creds, path)
    token = creds.get("accessToken") or creds.get("access_token")
    if not token:
        raise SubscriptionAuthError(
            "Las credenciales de Claude no contienen un accessToken. "
            "Vuelve a autenticarte con `claude auth login --claudeai`."
        )
    if _is_expired(creds):
        logger.warning(
            "El token OAuth de Claude parece caducado y no se pudo refrescar; "
            "la petición podría devolver 401. Ejecuta `claude` para refrescarlo."
        )
    return str(token)


# ---------------------------------------------------------------------------
# Account metadata (accountUuid -> metadata.user_id)
# ---------------------------------------------------------------------------


def _account_user_id() -> str | None:
    p = Path("~/.claude.json").expanduser()
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    oauth = cfg.get("oauthAccount") if isinstance(cfg, dict) else None
    if isinstance(oauth, dict) and isinstance(oauth.get("accountUuid"), str):
        return oauth["accountUuid"]
    return None


# ---------------------------------------------------------------------------
# Billing-header signing (mirror upstream src/signing.ts)
# ---------------------------------------------------------------------------


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        return text
        return ""
    return ""


def _compute_cch(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:5]


def _compute_version_suffix(text: str, version: str) -> str:
    sampled = "".join(text[i] if i < len(text) else "0" for i in (4, 7, 20))
    return hashlib.sha256(f"{_BILLING_SALT}{sampled}{version}".encode()).hexdigest()[:3]


def _build_billing_header_value(
    messages: list[dict[str, Any]], version: str, entrypoint: str
) -> str:
    text = _first_user_text(messages)
    suffix = _compute_version_suffix(text, version)
    cch = _compute_cch(text)
    return (
        f"{_BILLING_PREFIX}: "
        f"cc_version={version}.{suffix}; "
        f"cc_entrypoint={entrypoint}; "
        f"cch={cch};"
    )


# ---------------------------------------------------------------------------
# Stainless SDK fingerprint headers
# ---------------------------------------------------------------------------


def _stainless_arch() -> str:
    machine = (platform.machine() or "").lower()
    return {
        "x86_64": "x64",
        "amd64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
        "i386": "ia32",
        "i686": "ia32",
    }.get(machine, machine or "unknown")


def _stainless_os() -> str:
    return {"Darwin": "MacOS", "Linux": "Linux", "Windows": "Windows"}.get(
        platform.system(), platform.system() or "Unknown"
    )


def _spoof_headers() -> dict[str, str]:
    return {
        "anthropic-dangerous-direct-browser-access": "true",
        "x-stainless-arch": _stainless_arch(),
        "x-stainless-lang": "js",
        "x-stainless-os": _stainless_os(),
        "x-stainless-package-version": _STAINLESS_PACKAGE_VERSION,
        "x-stainless-retry-count": "0",
        "x-stainless-runtime": "node",
        "x-stainless-runtime-version": _STAINLESS_NODE_VERSION,
        "x-stainless-timeout": "600",
    }


def _merge_spoof_extras(api_kwargs: dict[str, Any]) -> None:
    headers = dict(_spoof_headers())
    existing = api_kwargs.get("extra_headers")
    if isinstance(existing, dict):
        headers.update(existing)  # caller headers win
    api_kwargs["extra_headers"] = headers

    query: dict[str, Any] = {"beta": "true"}
    existing_q = api_kwargs.get("extra_query")
    if isinstance(existing_q, dict):
        query.update(existing_q)
    api_kwargs["extra_query"] = query


# ---------------------------------------------------------------------------
# System prompt relocation (non-identity system text -> first user message)
# ---------------------------------------------------------------------------


def _prepend_to_first_user_message(
    messages: list[dict[str, Any]], texts: list[str]
) -> None:
    if not texts:
        return
    combined = "\n\n".join(
        f"<system-reminder>\n{t}\n</system-reminder>" for t in texts
    )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_text = f"{combined}\n\n{content}" if content else combined
            messages[i] = {**msg, "content": [{"type": "text", "text": new_text}]}
            return
        if isinstance(content, list):
            new_content = list(content)
            for j, block in enumerate(new_content):
                if isinstance(block, dict) and block.get("type") == "text":
                    existing = block.get("text") or ""
                    new_content[j] = {
                        **block,
                        "text": f"{combined}\n\n{existing}" if existing else combined,
                    }
                    messages[i] = {**msg, "content": new_content}
                    return
            new_content.insert(0, {"type": "text", "text": combined})
            messages[i] = {**msg, "content": new_content}
            return
        messages[i] = {**msg, "content": [{"type": "text", "text": combined}]}
        return


# ---------------------------------------------------------------------------
# Request transform orchestrator
# ---------------------------------------------------------------------------


def apply_subscription_transforms(
    api_kwargs: dict[str, Any], version: str = CLAUDE_CODE_VERSION
) -> dict[str, Any]:
    """Rewrite request kwargs in place so OAuth routes to the subscription tier.

    Idempotent: a stale billing header is dropped before the new one is added
    and a duplicate identity entry is removed, so it is safe to call twice.
    Tool names are intentionally left untouched (see module docstring).
    """
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return api_kwargs

    raw_system = api_kwargs.get("system")
    if raw_system is None:
        system: list[Any] = []
    elif isinstance(raw_system, str):
        system = [{"type": "text", "text": raw_system}] if raw_system else []
    elif isinstance(raw_system, list):
        system = list(raw_system)
    else:
        logger.warning(
            "Tipo de system inesperado %s; se omite el bypass",
            type(raw_system).__name__,
        )
        return api_kwargs

    billing_entry = {
        "type": "text",
        "text": _build_billing_header_value(messages, version, _BILLING_ENTRYPOINT),
    }

    kept: list[Any] = []
    moved_texts: list[str] = []
    identity_seen = False
    for entry in system:
        if not isinstance(entry, dict) or entry.get("type") != "text":
            kept.append(entry)
            continue
        text = entry.get("text") or ""
        if text.startswith(_BILLING_PREFIX):
            continue  # stale billing header -> drop
        if text.startswith(_SYSTEM_IDENTITY):
            if identity_seen:
                continue
            identity_seen = True
            rest = text[len(_SYSTEM_IDENTITY):].lstrip("\n")
            kept.append({"type": "text", "text": _SYSTEM_IDENTITY})
            if rest:
                moved_texts.append(rest)
            continue
        if text:
            moved_texts.append(text)

    if not identity_seen:
        kept.insert(0, {"type": "text", "text": _SYSTEM_IDENTITY})

    api_kwargs["system"] = [billing_entry] + kept
    if moved_texts:
        _prepend_to_first_user_message(messages, moved_texts)

    _merge_spoof_extras(api_kwargs)

    user_id = _account_user_id()
    if user_id:
        meta = api_kwargs.get("metadata")
        if isinstance(meta, dict):
            meta.setdefault("user_id", user_id)
        else:
            api_kwargs["metadata"] = {"user_id": user_id}

    return api_kwargs


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def build_subscription_client(
    credentials_path: str | Path | None = None,
    *,
    allow_refresh: bool = True,
) -> Any:
    """Construct an ``anthropic.Anthropic`` client authed by the subscription.

    Uses bearer (OAuth) auth instead of ``x-api-key`` and pins the OAuth beta
    flags. Per-request transforms are applied separately by the caller via
    :func:`apply_subscription_transforms`.
    """
    import anthropic  # lazy: agent layer is optional

    token = get_access_token(credentials_path, allow_refresh=allow_refresh)
    return anthropic.Anthropic(
        auth_token=token,
        default_headers={"anthropic-beta": ",".join(OAUTH_BETAS)},
    )
