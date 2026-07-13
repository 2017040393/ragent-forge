from __future__ import annotations

from typing import Any, cast

import httpx as _httpx

from ragent_forge.app.ports import HttpResponse


class HttpxClient:
    """HTTPX adapter used by OpenAI-compatible providers."""

    TransportError = _httpx.TransportError

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> HttpResponse:
        return cast(
            HttpResponse,
            _httpx.post(
                url,
                headers=headers,
                json=json,
                timeout=timeout,
            ),
        )

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> Any:
        return _httpx.stream(
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
        )

    def is_transport_error(self, exc: Exception) -> bool:
        return isinstance(exc, _httpx.TransportError)


default_http_client = HttpxClient()
