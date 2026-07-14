from __future__ import annotations

import ssl

import httpx
import pytest

from ragent_forge.infrastructure import http_client
from ragent_forge.infrastructure.http_client import HttpxClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True}


def test_post_retries_tls_handshake_eof_with_tls12_and_remembers_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ssl.SSLContext | bool | None] = []

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: object,
        timeout: int,
        verify: ssl.SSLContext | bool | None = None,
    ) -> FakeResponse:
        calls.append(verify)
        if verify is None:
            raise httpx.ConnectError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] handshake failed"
            )
        return FakeResponse()

    monkeypatch.setattr(http_client._httpx, "post", fake_post)
    client = HttpxClient()
    request = {
        "url": "https://embeddings.example/v1/embeddings",
        "headers": {"Authorization": "Bearer hidden"},
        "json": {"input": ["query"]},
        "timeout": 30,
    }

    client.post(**request)
    client.post(**request)

    assert calls[0] is None
    assert len(calls) == 3
    assert all(isinstance(value, ssl.SSLContext) for value in calls[1:])
    assert all(
        value.maximum_version == ssl.TLSVersion.TLSv1_2
        for value in calls[1:]
        if isinstance(value, ssl.SSLContext)
    )


def test_post_does_not_retry_unrelated_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: object,
        timeout: int,
        verify: ssl.SSLContext | bool | None = None,
    ) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(http_client._httpx, "post", fake_post)

    with pytest.raises(httpx.ConnectError, match="connection refused"):
        HttpxClient().post(
            "https://embeddings.example/v1/embeddings",
            headers={},
            json={"input": ["query"]},
            timeout=30,
        )

    assert calls == 1


def test_post_only_remembers_the_host_that_required_tls12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ssl.SSLContext | bool | None]] = []

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: object,
        timeout: int,
        verify: ssl.SSLContext | bool | None = None,
    ) -> FakeResponse:
        calls.append((url, verify))
        if "fallback.example" in url and verify is None:
            raise httpx.ConnectError("UNEXPECTED_EOF_WHILE_READING")
        return FakeResponse()

    monkeypatch.setattr(http_client._httpx, "post", fake_post)
    client = HttpxClient()

    client.post(
        "https://fallback.example/embeddings",
        headers={},
        json={},
        timeout=30,
    )
    client.post(
        "https://normal.example/embeddings",
        headers={},
        json={},
        timeout=30,
    )

    assert calls[-1] == ("https://normal.example/embeddings", None)
