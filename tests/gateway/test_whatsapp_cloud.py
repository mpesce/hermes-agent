"""Tests for the WhatsApp Cloud API adapter (Phase 2).

Covers the outbound Graph API send path and the inbound verify-token
handshake. The webhook POST path is currently a stub (Phase 3 will add
signature verification + dispatch); we just confirm it accepts a body
and returns 200 here.

All tests are fixture-driven — no live network. httpx is patched so the
adapter never reaches graph.facebook.com, and the aiohttp server is
exercised with synthetic ``Request`` objects.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(**overrides):
    """Build a WhatsAppCloudAdapter with test attributes (bypass __init__).

    Mirrors the pattern in tests/gateway/test_whatsapp_*.py.
    """
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter.platform = Platform.WHATSAPP_CLOUD
    adapter.config = MagicMock()
    adapter.config.extra = {}

    # Cloud-API-specific attributes
    adapter._phone_number_id = overrides.pop("phone_number_id", "1234567890")
    adapter._access_token = overrides.pop("access_token", "test-token")
    adapter._app_id = overrides.pop("app_id", "")
    adapter._app_secret = overrides.pop("app_secret", "")
    adapter._waba_id = overrides.pop("waba_id", "")
    adapter._verify_token = overrides.pop("verify_token", "")
    adapter._webhook_host = "127.0.0.1"
    adapter._webhook_port = 8090
    adapter._webhook_path = "/whatsapp/webhook"
    adapter._health_path = "/health"
    adapter._api_version = overrides.pop("api_version", "v20.0")
    adapter._runner = None
    adapter._http_client = None

    # Behavior-mixin contract
    adapter._reply_prefix = None
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []

    # BasePlatformAdapter contract — minimum to keep send/lifecycle happy
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()

    # Apply any leftover overrides directly
    for key, value in overrides.items():
        setattr(adapter, key, value)
    return adapter


def _mock_httpx_response(status_code: int, json_body: dict):
    """Build an httpx-Response-like mock the adapter's ``send`` will accept."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body)
    resp.text = json.dumps(json_body)
    return resp


# ---------------------------------------------------------------------------
# Outbound send via Graph API
# ---------------------------------------------------------------------------

class TestSendText:
    """Outbound text-message path."""

    @pytest.mark.asyncio
    async def test_send_builds_correct_url(self):
        adapter = _make_adapter(phone_number_id="9999", api_version="v20.0")
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hello")

        called_url = adapter._http_client.post.call_args.args[0]
        assert called_url == "https://graph.facebook.com/v20.0/9999/messages"

    @pytest.mark.asyncio
    async def test_send_includes_bearer_auth(self):
        adapter = _make_adapter(access_token="my-secret-token")
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hi")

        headers = adapter._http_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_payload_shape(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hello world")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["messaging_product"] == "whatsapp"
        assert payload["recipient_type"] == "individual"
        assert payload["to"] == "15551234567"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "hello world"
        assert payload["text"]["preview_url"] is True

    @pytest.mark.asyncio
    async def test_send_returns_wamid(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.HBgL...="}]}
            )
        )

        result = await adapter.send("15551234567", "hi")

        assert result.success is True
        assert result.message_id == "wamid.HBgL...="

    @pytest.mark.asyncio
    async def test_send_applies_markdown_conversion(self):
        """Mixin's format_message should run before send."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        await adapter.send("15551234567", "**bold** text")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["text"]["body"] == "*bold* text"

    @pytest.mark.asyncio
    async def test_send_reply_to_attaches_context_first_chunk_only(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        await adapter.send("15551234567", "short reply", reply_to="wamid.original")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["context"] == {"message_id": "wamid.original"}

    @pytest.mark.asyncio
    async def test_send_long_message_chunked(self):
        """Messages over the chunk limit are split into multiple POSTs."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        # MAX_MESSAGE_LENGTH = 4096 from the mixin. 8500 chars forces 2+ chunks.
        long_text = "a" * 8500
        await adapter.send("15551234567", long_text)

        # At least 2 POST calls
        assert adapter._http_client.post.call_count >= 2
        # Second call should NOT have context (only first chunk gets reply_to)
        first_call = adapter._http_client.post.call_args_list[0]
        second_call = adapter._http_client.post.call_args_list[1]
        # No reply_to passed → no context anywhere, but verify structure anyway
        assert "context" not in second_call.kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_graph_error_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                400,
                {
                    "error": {
                        "message": "Invalid parameter",
                        "type": "OAuthException",
                        "code": 100,
                        "fbtrace_id": "abc",
                    }
                },
            )
        )

        result = await adapter.send("15551234567", "hi")

        assert result.success is False
        assert "graph error 100" in result.error
        assert "Invalid parameter" in result.error

    @pytest.mark.asyncio
    async def test_send_empty_content_no_request(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock()

        result = await adapter.send("15551234567", "")
        assert result.success is True
        assert result.message_id is None
        adapter._http_client.post.assert_not_called()

        result = await adapter.send("15551234567", "   \n  ")
        assert result.success is True
        adapter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = None

        result = await adapter.send("15551234567", "hi")
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_network_exception_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=RuntimeError("boom"))

        result = await adapter.send("15551234567", "hi")
        assert result.success is False
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# Inbound webhook verify (GET) handshake
# ---------------------------------------------------------------------------

def _verify_request(query: dict):
    """Build a minimal aiohttp.web.Request stub for verify tests."""
    request = MagicMock()
    request.query = query
    return request


class TestWebhookVerify:
    """GET <webhook>?hub.mode=...&hub.verify_token=...&hub.challenge=..."""

    @pytest.mark.asyncio
    async def test_verify_echoes_challenge_on_match(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "shared-secret-123",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 200
        assert response.text == "abc-12345"
        assert response.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_verify_rejects_token_mismatch(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_verify_rejects_wrong_mode(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "unsubscribe",
            "hub.verify_token": "shared-secret-123",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_verify_rejects_missing_challenge(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "shared-secret-123",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_verify_refuses_when_token_unconfigured(self):
        """An empty verify_token must NOT match an empty incoming token —
        otherwise an attacker who guesses the misconfiguration could
        subscribe their own webhook URL.
        """
        adapter = _make_adapter(verify_token="")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "abc",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 503  # service refuses to perform handshake


# ---------------------------------------------------------------------------
# Inbound webhook POST stub (Phase 2 — Phase 3 will add real verification)
# ---------------------------------------------------------------------------

class TestWebhookPostStub:
    """The POST stub should accept payloads but not crash. Phase 3 wires
    in signature verification + dispatch."""

    @pytest.mark.asyncio
    async def test_post_accepts_small_body(self):
        adapter = _make_adapter()
        request = MagicMock()
        request.read = AsyncMock(return_value=b'{"object":"whatsapp_business_account"}')

        response = await adapter._handle_webhook(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_post_rejects_oversize_body(self):
        adapter = _make_adapter()
        request = MagicMock()
        # 4MB > 3MB limit per Meta docs.
        request.read = AsyncMock(return_value=b"x" * (4 * 1024 * 1024))

        response = await adapter._handle_webhook(request)

        assert response.status == 413

    @pytest.mark.asyncio
    async def test_post_handles_unreadable_body(self):
        adapter = _make_adapter()
        request = MagicMock()
        request.read = AsyncMock(side_effect=RuntimeError("read failed"))

        response = await adapter._handle_webhook(request)

        assert response.status == 400


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_reports_config_visibility(self):
        adapter = _make_adapter(
            phone_number_id="555",
            verify_token="secret",
            app_secret="signing-key",
        )
        request = MagicMock()

        response = await adapter._handle_health(request)

        # web.json_response stores the dict on .text as JSON
        body = json.loads(response.text)
        assert body["status"] == "ok"
        assert body["platform"] == "whatsapp_cloud"
        assert body["phone_number_id"] == "555"
        assert body["verify_token_configured"] is True
        assert body["app_secret_configured"] is True

    @pytest.mark.asyncio
    async def test_health_flags_missing_secrets(self):
        adapter = _make_adapter(verify_token="", app_secret="")
        request = MagicMock()

        response = await adapter._handle_health(request)
        body = json.loads(response.text)
        assert body["verify_token_configured"] is False
        assert body["app_secret_configured"] is False


# ---------------------------------------------------------------------------
# Mixin contract — gating still works on the cloud adapter
# ---------------------------------------------------------------------------

class TestMixinInherited:
    """Sanity-check: the Cloud adapter inherits the same gating behavior
    as the Baileys adapter via WhatsAppBehaviorMixin.
    """

    def test_format_message_converts_markdown(self):
        adapter = _make_adapter()
        assert adapter.format_message("**bold**") == "*bold*"
        assert adapter.format_message("# Title") == "*Title*"

    def test_should_process_message_dm_open(self):
        adapter = _make_adapter()
        adapter._dm_policy = "open"
        assert adapter._should_process_message({
            "chatId": "15551234567@c.us",
            "senderId": "15551234567@c.us",
            "isGroup": False,
            "body": "hi",
        }) is True

    def test_should_process_message_dm_disabled(self):
        adapter = _make_adapter()
        adapter._dm_policy = "disabled"
        assert adapter._should_process_message({
            "chatId": "15551234567@c.us",
            "senderId": "15551234567@c.us",
            "isGroup": False,
            "body": "hi",
        }) is False

    def test_broadcast_chats_filtered(self):
        adapter = _make_adapter()
        assert adapter._should_process_message({
            "chatId": "status@broadcast",
            "isGroup": False,
            "body": "x",
        }) is False
