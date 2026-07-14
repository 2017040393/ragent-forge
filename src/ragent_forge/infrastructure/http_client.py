from __future__ import annotations

import ssl
from threading import RLock
from typing import Any, cast
from urllib.parse import urlsplit

import httpx as _httpx

from ragent_forge.app.ports import HttpResponse


class HttpxClient:
    """HTTPX adapter used by OpenAI-compatible providers."""

    TransportError = _httpx.TransportError

    def __init__(self) -> None:
        self._tls12_hosts: set[str] = set()
        self._transport_lock = RLock()

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> HttpResponse:
        host = _url_host(url)
        if self._requires_tls12(host):
            return self._post_with_tls12(
                url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
        try:
            return self._post(
                url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
        except _httpx.ConnectError as exc:
            if not _is_tls_handshake_eof(exc):
                raise
            response = self._post_with_tls12(
                url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
            with self._transport_lock:
                self._tls12_hosts.add(host)
            return response

    def _post(
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

    def _post_with_tls12(
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
                verify=_tls12_context(),
            ),
        )

    def _requires_tls12(self, host: str) -> bool:
        with self._transport_lock:
            return host in self._tls12_hosts

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> Any:
        verify: ssl.SSLContext | bool = (
            _tls12_context() if self._requires_tls12(_url_host(url)) else True
        )
        return _httpx.stream(
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
            verify=verify,
        )

    def is_transport_error(self, exc: Exception) -> bool:
        return isinstance(exc, _httpx.TransportError)


def _is_tls_handshake_eof(exc: _httpx.ConnectError) -> bool:
    return "UNEXPECTED_EOF_WHILE_READING" in str(exc).upper()


def _tls12_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def _url_host(url: str) -> str:
    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"HTTP URL is missing a host: {url}")
    return host.lower()


default_http_client = HttpxClient()
