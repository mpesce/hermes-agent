"""
WhatsApp Cloud API adapter — official Meta WhatsApp Business Platform.

This adapter is a *complement* to ``whatsapp.py`` (the Baileys bridge), not
a replacement. The two are independent:

- ``whatsapp.py``      — unofficial Baileys bridge, personal accounts, no
                         public URL needed, account-ban risk.
- ``whatsapp_cloud.py`` (this file) — official Meta Cloud API, Business
                         account required, public webhook URL required,
                         token-based auth.

Both share gating / mention / formatting behavior via ``WhatsAppBehaviorMixin``.

Phase scope (this file evolves across phases):
- Phase 2 — outbound text via Graph API + webhook server with verify-token
            handshake. POST endpoint accepts payloads but does NOT yet
            verify signatures (insecure; Phase 3 fixes this).
- Phase 3 — X-Hub-Signature-256 HMAC verification + replay protection.
- Phase 4 — media upload + send (image/video/audio/document).
- Phase 5 — 24-hour conversation window + template fallback.

Required env vars to enable the adapter:
- WHATSAPP_CLOUD_PHONE_NUMBER_ID  (the Graph URL path component)
- WHATSAPP_CLOUD_ACCESS_TOKEN     (System User permanent token)

Optional / Phase-3+:
- WHATSAPP_CLOUD_APP_ID
- WHATSAPP_CLOUD_APP_SECRET       (HMAC key for X-Hub-Signature-256)
- WHATSAPP_CLOUD_WABA_ID          (analytics / future use)
- WHATSAPP_CLOUD_VERIFY_TOKEN     (hub.verify_token shared secret)
- WHATSAPP_CLOUD_WEBHOOK_HOST     (default 0.0.0.0)
- WHATSAPP_CLOUD_WEBHOOK_PORT     (default 8090)
- WHATSAPP_CLOUD_WEBHOOK_PATH     (default /whatsapp/webhook)
- WHATSAPP_CLOUD_API_VERSION      (default v20.0)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

logger = logging.getLogger(__name__)


DEFAULT_API_VERSION = "v20.0"
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8090
DEFAULT_WEBHOOK_PATH = "/whatsapp/webhook"
GRAPH_API_BASE = "https://graph.facebook.com"


def check_whatsapp_cloud_requirements() -> bool:
    """Return whether transport dependencies are available.

    aiohttp is needed for the webhook server (inbound). httpx is needed
    for Graph API calls (outbound). Both ship with hermes-agent's default
    dependency set, so this should always be True in normal installs.
    """
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


class WhatsAppCloudAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """WhatsApp Business Cloud API adapter.

    Outbound: HTTPS POST to ``graph.facebook.com/<api_version>/<phone_id>/messages``.
    Inbound: aiohttp server accepting Meta's webhook payloads.

    The mixin must come first in the bases list so its ``format_message``
    overrides ``BasePlatformAdapter.format_message`` (the base provides a
    generic implementation that does not convert markdown to WhatsApp
    syntax). The Baileys adapter does the same.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP_CLOUD)
        extra = config.extra or {}

        # Required
        self._phone_number_id: str = str(extra.get("phone_number_id", "")).strip()
        self._access_token: str = str(extra.get("access_token", "")).strip()

        # Optional / used in later phases
        self._app_id: str = str(extra.get("app_id", "")).strip()
        self._app_secret: str = str(extra.get("app_secret", "")).strip()
        self._waba_id: str = str(extra.get("waba_id", "")).strip()
        self._verify_token: str = str(extra.get("verify_token", "")).strip()

        # Webhook server config
        self._webhook_host: str = str(extra.get("webhook_host", DEFAULT_WEBHOOK_HOST))
        self._webhook_port: int = int(extra.get("webhook_port", DEFAULT_WEBHOOK_PORT))
        self._webhook_path: str = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path: str = self._normalize_path(
            extra.get("health_path", "/health")
        )

        # Graph API
        self._api_version: str = str(extra.get("api_version", DEFAULT_API_VERSION))

        # Behavior-mixin contract: these names are read by the mixin's
        # gating methods. Derived from env / config the same way the
        # Baileys adapter derives them.
        import os

        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        self._dm_policy: str = str(
            extra.get("dm_policy") or os.getenv("WHATSAPP_DM_POLICY", "open")
        ).strip().lower()
        self._allow_from: set[str] = self._coerce_allow_list(
            extra.get("allow_from") or extra.get("allowFrom")
        )
        self._group_policy: str = str(
            extra.get("group_policy") or os.getenv("WHATSAPP_GROUP_POLICY", "open")
        ).strip().lower()
        self._group_allow_from: set[str] = self._coerce_allow_list(
            extra.get("group_allow_from") or extra.get("groupAllowFrom")
        )
        self._mention_patterns = self._compile_mention_patterns()

        # Runtime
        self._runner = None
        self._http_client: Optional["httpx.AsyncClient"] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    def _graph_url(self, path: str) -> str:
        """Build a Graph API URL for this adapter's phone-number scope."""
        if path.startswith("/"):
            path = path[1:]
        return f"{GRAPH_API_BASE}/{self._api_version}/{self._phone_number_id}/{path}"

    def _effective_reply_prefix(self) -> str:
        """Cloud API has no self-chat concept — never prepend a reply prefix.

        Override the mixin default which keys off WHATSAPP_MODE=self-chat
        (a Baileys-only setting).
        """
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        return ""

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> bool:
        if not check_whatsapp_cloud_requirements():
            self._set_fatal_error(
                "whatsapp_cloud_deps_missing",
                "aiohttp and httpx are required for whatsapp_cloud — "
                "reinstall hermes-agent.",
                retryable=False,
            )
            return False
        if not self._phone_number_id or not self._access_token:
            self._set_fatal_error(
                "whatsapp_cloud_unconfigured",
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID and WHATSAPP_CLOUD_ACCESS_TOKEN "
                "are required.",
                retryable=False,
            )
            return False

        # Outbound HTTP client. Tighter keepalive matches other platform
        # adapters so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits

        self._http_client = httpx.AsyncClient(
            timeout=30.0, limits=platform_httpx_limits()
        )

        # Inbound webhook server.
        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_get(self._webhook_path, self._handle_verify)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()

        self._mark_connected()
        logger.info(
            "[whatsapp_cloud] Listening on %s:%d%s (Graph %s, phone_id=%s)",
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
            self._api_version,
            self._phone_number_id,
        )
        if not self._verify_token:
            logger.warning(
                "[whatsapp_cloud] WHATSAPP_CLOUD_VERIFY_TOKEN is not set — "
                "the GET subscription handshake will fail until it is."
            )
        if not self._app_secret:
            logger.warning(
                "[whatsapp_cloud] WHATSAPP_CLOUD_APP_SECRET is not set — "
                "incoming webhooks are NOT signature-verified. Do not "
                "expose this endpoint to the public internet until "
                "Phase 3 lands."
            )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.exception("[whatsapp_cloud] webhook server cleanup failed")
            self._runner = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.exception("[whatsapp_cloud] http client close failed")
            self._http_client = None
        self._mark_disconnected()

    # ------------------------------------------------------------------ outbound
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message via Graph API.

        ``chat_id`` is the recipient's WhatsApp ID (``wa_id``) — typically
        their phone number with country code, no plus sign.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self._outgoing_chunk_limit())

        url = self._graph_url("messages")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        last_message_id: Optional[str] = None
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": chat_id,
                "type": "text",
                "text": {"body": chunk, "preview_url": True},
            }
            if reply_to and idx == 0:
                # Quote the user's message on the first chunk only.
                payload["context"] = {"message_id": reply_to}
            try:
                resp = await self._http_client.post(url, headers=headers, json=payload)
            except Exception as exc:
                logger.exception("[whatsapp_cloud] send failed")
                return SendResult(success=False, error=str(exc))

            if resp.status_code != 200:
                # Meta returns structured errors in the body — surface them
                # to the caller so log lines have actionable context.
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text[:500]}
                error_msg = self._format_graph_error(body, resp.status_code)
                logger.warning(
                    "[whatsapp_cloud] send rejected (status=%d): %s",
                    resp.status_code,
                    error_msg,
                )
                return SendResult(success=False, error=error_msg)

            try:
                data = resp.json()
                ids = data.get("messages") or []
                if ids:
                    last_message_id = ids[0].get("id")
            except Exception:
                pass

        return SendResult(success=True, message_id=last_message_id)

    @staticmethod
    def _format_graph_error(body: Dict[str, Any], status_code: int) -> str:
        err = (body or {}).get("error") or {}
        # Graph API error shape:
        # {"error": {"message": "...", "type": "...", "code": ..., "fbtrace_id": "..."}}
        message = err.get("message") or body.get("raw") or "unknown error"
        code = err.get("code")
        if code is not None:
            return f"graph error {code} (HTTP {status_code}): {message}"
        return f"HTTP {status_code}: {message}"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Cloud API doesn't expose a direct "chat info" endpoint the way
        # Slack/Discord do — we just echo the wa_id. Profile name (when
        # known) flows in via webhook ``contacts[].profile.name`` and is
        # cached on the MessageEvent, not here.
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------ inbound
    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": self.platform.value,
                "phone_number_id": self._phone_number_id,
                "webhook_path": self._webhook_path,
                "verify_token_configured": bool(self._verify_token),
                "app_secret_configured": bool(self._app_secret),
            }
        )

    async def _handle_verify(self, request: "web.Request") -> "web.Response":
        """Meta subscription verification handshake.

        Meta calls GET ``<webhook>?hub.mode=subscribe&hub.verify_token=...
        &hub.challenge=...``. We must echo the challenge as plain text iff
        ``hub.mode == "subscribe"`` AND ``hub.verify_token`` matches the
        shared secret. Constant-time comparison.
        """
        if not self._verify_token:
            # Misconfigured server — refuse rather than silently accepting
            # any verify_token, which would let an attacker subscribe.
            return web.Response(status=503, text="verify_token not configured")

        mode = request.query.get("hub.mode", "")
        token = request.query.get("hub.verify_token", "")
        challenge = request.query.get("hub.challenge", "")

        if mode != "subscribe":
            return web.Response(status=400, text="bad mode")

        # Constant-time compare to avoid token-length / token-content leaks
        # via timing. ``hmac.compare_digest`` works on str.
        import hmac as _hmac

        if not _hmac.compare_digest(token, self._verify_token):
            return web.Response(status=403, text="verify_token mismatch")
        if not challenge:
            return web.Response(status=400, text="missing challenge")
        return web.Response(text=challenge, content_type="text/plain")

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """Inbound webhook POST stub.

        Phase 2 only ACKs Meta with 200 so retries don't pile up. Phase 3
        wires in:
          - X-Hub-Signature-256 verification BEFORE JSON parse
          - replay protection via wamid dedup
          - dispatch through ``_should_process_message`` →
            ``handle_message``

        Until Phase 3 lands, do NOT expose this endpoint to the public
        internet — anyone who can reach it can inject fake payloads.
        """
        # Read raw bytes here (not request.json()) so Phase 3's signature
        # verification doesn't have to redo the read. aiohttp will only
        # let us read the body once.
        try:
            raw = await request.read()
        except Exception:
            return web.Response(status=400)
        # Light-touch sanity check — full parsing happens in Phase 3.
        if len(raw) > 3 * 1024 * 1024:
            # Meta's documented max payload is 3MB.
            return web.Response(status=413)
        logger.debug(
            "[whatsapp_cloud] webhook received %d bytes (signature unverified — "
            "Phase 3 will lock this down)",
            len(raw),
        )
        return web.Response(status=200)
