"""Offline tests for the Claude subscription (OAuth) auth layer.

No network, no real credentials: token loading, refresh, and request transforms
are all exercised against fakes/fixtures.
"""

from __future__ import annotations

import json
import time

import pytest

from qtdata.agents import claude_subscription as cs
from qtdata.agents.llm import LLMClient
from qtdata.config import Settings
from tests.fake_anthropic import FakeAnthropic, response, text_block, tool_use

# --------------------------------------------------------------------------- #
# Billing-header signing
# --------------------------------------------------------------------------- #


def test_billing_header_is_deterministic_and_well_formed():
    messages = [{"role": "user", "content": "Mandato: calidad con sentimiento"}]
    v1 = cs._build_billing_header_value(messages, cs.CLAUDE_CODE_VERSION, "sdk-cli")
    v2 = cs._build_billing_header_value(messages, cs.CLAUDE_CODE_VERSION, "sdk-cli")
    assert v1 == v2  # deterministic
    assert v1.startswith("x-anthropic-billing-header: ")
    assert "cc_entrypoint=sdk-cli;" in v1
    assert "cch=" in v1 and "cc_version=" in v1


def test_billing_header_changes_with_message_text():
    a = cs._build_billing_header_value(
        [{"role": "user", "content": "aaaaaaaaaaaaaaaaaaaaaaaa"}],
        cs.CLAUDE_CODE_VERSION,
        "sdk-cli",
    )
    b = cs._build_billing_header_value(
        [{"role": "user", "content": "bbbbbbbbbbbbbbbbbbbbbbbb"}],
        cs.CLAUDE_CODE_VERSION,
        "sdk-cli",
    )
    assert a != b


def test_version_suffix_pads_short_messages():
    # Must not IndexError on messages shorter than the sampled indices (4,7,20).
    suffix = cs._compute_version_suffix("hi", cs.CLAUDE_CODE_VERSION)
    assert len(suffix) == 3


def test_first_user_text_handles_block_content():
    messages = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": [{"type": "text", "text": "hola"}]},
    ]
    assert cs._first_user_text(messages) == "hola"


# --------------------------------------------------------------------------- #
# Request transforms
# --------------------------------------------------------------------------- #


def _basic_kwargs():
    return {
        "model": "claude-opus-4-8",
        "system": [{"type": "text", "text": "Eres un screener cuantitativo."}],
        "messages": [{"role": "user", "content": "Mandato: X"}],
        "tools": [
            {"name": "run_sql", "input_schema": {"type": "object"}},
            {"name": "submit_proposal", "input_schema": {"type": "object"}},
        ],
    }


def test_transforms_inject_billing_header_as_first_system_entry():
    kw = cs.apply_subscription_transforms(_basic_kwargs())
    assert kw["system"][0]["text"].startswith("x-anthropic-billing-header:")
    # Claude Code identity present as a real system entry.
    assert any(
        e.get("text") == cs._SYSTEM_IDENTITY for e in kw["system"] if isinstance(e, dict)
    )


def test_transforms_relocate_non_identity_system_to_user_message():
    kw = cs.apply_subscription_transforms(_basic_kwargs())
    first_user = kw["messages"][0]["content"]
    text = first_user if isinstance(first_user, str) else first_user[0]["text"]
    assert "<system-reminder>" in text
    assert "screener cuantitativo" in text
    # The original system text is no longer a bare system entry.
    assert all(
        "screener cuantitativo" not in (e.get("text") or "")
        for e in kw["system"]
        if isinstance(e, dict)
    )


def test_transforms_add_stainless_headers_and_beta_query():
    kw = cs.apply_subscription_transforms(_basic_kwargs())
    headers = kw["extra_headers"]
    assert headers["x-stainless-runtime"] == "node"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert kw["extra_query"]["beta"] == "true"


def test_transforms_do_not_rename_tools():
    """Critical: this project's tools must keep plain names (no mcp__ namespace)."""
    kw = cs.apply_subscription_transforms(_basic_kwargs())
    names = {t["name"] for t in kw["tools"]}
    assert names == {"run_sql", "submit_proposal"}
    assert not any(n.startswith("mcp__") for n in names)


def test_transforms_are_idempotent():
    kw = cs.apply_subscription_transforms(_basic_kwargs())
    billing_entries = [
        e for e in kw["system"] if isinstance(e, dict)
        and (e.get("text") or "").startswith("x-anthropic-billing-header:")
    ]
    kw2 = cs.apply_subscription_transforms(kw)
    billing_entries2 = [
        e for e in kw2["system"] if isinstance(e, dict)
        and (e.get("text") or "").startswith("x-anthropic-billing-header:")
    ]
    assert len(billing_entries) == 1
    assert len(billing_entries2) == 1  # no duplicate billing header
    identities = [
        e for e in kw2["system"] if isinstance(e, dict)
        and e.get("text") == cs._SYSTEM_IDENTITY
    ]
    assert len(identities) == 1  # no duplicate identity


def test_transforms_noop_without_messages():
    kw = {"system": "x", "messages": []}
    assert cs.apply_subscription_transforms(dict(kw)) == kw


# --------------------------------------------------------------------------- #
# Credential loading + refresh
# --------------------------------------------------------------------------- #


def _write_creds(tmp_path, **over):
    creds = {
        "claudeAiOauth": {
            "accessToken": "tok-access",
            "refreshToken": "tok-refresh",
            "expiresAt": int((time.time() + 3600) * 1000),
            **over,
        }
    }
    p = tmp_path / "creds.json"
    p.write_text(json.dumps(creds))
    return p


def test_get_access_token_reads_nested_oauth(tmp_path):
    p = _write_creds(tmp_path)
    assert cs.get_access_token(p, allow_refresh=False) == "tok-access"


def test_get_access_token_missing_file_raises(tmp_path):
    with pytest.raises(cs.SubscriptionAuthError):
        cs.get_access_token(tmp_path / "nope.json")


def test_expired_token_triggers_refresh(tmp_path):
    p = _write_creds(tmp_path, expiresAt=int((time.time() - 10) * 1000))

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
                "expires_in": 3600,
            }

    class FakeSession:
        def __init__(self):
            self.posted = None

        def post(self, url, **kwargs):
            self.posted = (url, kwargs)
            return FakeResp()

    creds = cs._load_credentials(p)
    sess = FakeSession()
    updated = cs.refresh_oauth_token(creds, p, session=sess)
    assert updated["accessToken"] == "fresh-access"
    assert sess.posted[0] == cs._OAUTH_TOKEN_URL
    # Persisted back to disk under the nested key.
    on_disk = json.loads(p.read_text())["claudeAiOauth"]
    assert on_disk["accessToken"] == "fresh-access"


def test_refresh_without_refresh_token_is_noop(tmp_path):
    creds = {"accessToken": "x"}
    assert cs.refresh_oauth_token(creds, tmp_path / "c.json") == creds


# --------------------------------------------------------------------------- #
# LLMClient integration
# --------------------------------------------------------------------------- #


def test_llmclient_applies_transforms_when_subscription_enabled():
    settings = Settings(agent_use_subscription=True)
    fake = FakeAnthropic()
    fake.messages.create_queue.append(
        response([text_block("hola"), tool_use("refuse", {"reason": "test"})])
    )
    llm = LLMClient(settings, client=fake)
    llm.create(
        system=[{"type": "text", "text": "system text"}],
        messages=[{"role": "user", "content": "Mandato: X"}],
        tools=[{"name": "run_sql", "input_schema": {"type": "object"}}],
    )
    sent = fake.messages.create_calls[0]
    # Billing header injected; tools untouched.
    assert sent["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert sent["tools"][0]["name"] == "run_sql"
    assert "x-stainless-runtime" in sent["extra_headers"]


def test_llmclient_skips_transforms_when_subscription_disabled():
    settings = Settings(agent_use_subscription=False)
    fake = FakeAnthropic()
    fake.messages.create_queue.append(
        response([tool_use("refuse", {"reason": "test"})])
    )
    llm = LLMClient(settings, client=fake)
    llm.create(
        system=[{"type": "text", "text": "system text"}],
        messages=[{"role": "user", "content": "Mandato: X"}],
    )
    sent = fake.messages.create_calls[0]
    # No billing header, no spoof headers.
    assert sent["system"][0]["text"] == "system text"
    assert "extra_headers" not in sent
