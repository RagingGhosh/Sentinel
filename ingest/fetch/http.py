"""The transport boundary, and D44's failure, retry and pacing policy.

Every value below is **engineering policy, not a source requirement** (D44): neither
CFPB nor Socrata documents a numeric rate limit for the endpoints Sentinel reads.

    transient   a connection error or timeout; HTTP 429, 500, 502, 503, 504; a body
                its caller's validator rejects                      -> retried
    permanent   HTTP 403, and every other status outside 2xx         -> never retried
    attempts    at most 6 per request
    backoff     2, 4, 8, 16, 32 seconds, with no jitter
    Retry-After the larger of the backoff and the header, capped at 300 seconds
    pacing      at least 1 second between request starts, per host
    exhaustion  FetchFailed, naming the request and its last status

HTTP 403 stops the run at once. Sentinel never changes its identity to get past a
refusal: the one User-Agent it sends names the project, and nothing here imitates a
browser or any other client.

`HttpClient` has no default transport, so nothing can reach the network by
accident. The real transport, `RequestsTransport`, is built only by
``ingest.cli.main`` for a ``--fetch`` run; tests inject their own.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from ingest.fetch.canonical import format_timestamp

USER_AGENT = "Sentinel-acquisition/1 (+https://github.com/RagingGhosh/Sentinel)"
"""The only identity Sentinel sends. Honest by construction: it names the project."""

MAX_ATTEMPTS = 6
BACKOFF_SECONDS = (2, 4, 8, 16, 32)
"""The wait after each failed attempt but the last. No jitter (D44)."""

RETRY_AFTER_CAP_SECONDS = 300
MIN_REQUEST_INTERVAL_MS = 1000
"""Between two request starts to the same host."""

TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
FORBIDDEN = 403

REQUEST_TIMEOUT_SECONDS = (30, 300)
"""Connect and read timeouts for the real transport. A timeout is transient."""


def policy() -> dict[str, Any]:
    """The values in force, as an acquisition record states them (D46)."""
    return {
        "backoff_seconds": list(BACKOFF_SECONDS),
        "max_attempts": MAX_ATTEMPTS,
        "min_request_interval_ms": MIN_REQUEST_INTERVAL_MS,
        "retry_after_cap_seconds": RETRY_AFTER_CAP_SECONDS,
    }


@dataclass(frozen=True)
class Response:
    """What a transport returns: the status, the headers (lower-cased), the body."""

    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class ClientIdentity:
    """Recorded in every acquisition record as its `client` (D44, D46)."""

    user_agent: str
    library: str
    library_version: str

    def as_record(self) -> dict[str, str]:
        return {
            "library": self.library,
            "library_version": self.library_version,
            "user_agent": self.user_agent,
        }


@dataclass(frozen=True)
class RequestRecord:
    """One request whose response a slice used, as D46's `requests` entries record it."""

    url: str
    params: tuple[tuple[str, str], ...]
    retrieved_at: str
    status: int
    response_sha256: str
    response_bytes: int

    def as_record(self) -> dict[str, Any]:
        return {
            "params": [[name, value] for name, value in self.params],
            "response_bytes": self.response_bytes,
            "response_sha256": self.response_sha256,
            "retrieved_at": self.retrieved_at,
            "status": self.status,
            "url": self.url,
        }


@dataclass(frozen=True)
class Fetched:
    response: Response
    record: RequestRecord


class TransportError(Exception):
    """The transport could not complete a request. Transient unless it says otherwise."""

    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class InvalidResponse(Exception):
    """A caller's validator rejected a 2xx body: unparseable, short, or inconsistent.

    Transient, because a truncated body is the commonest cause; it is retried like
    any other transient failure and becomes `FetchFailed` when attempts run out.
    """


class FetchFailed(Exception):
    """A request failed permanently, or every attempt failed (D44)."""

    def __init__(
        self,
        url: str,
        params: Sequence[tuple[str, str]],
        *,
        status: int | None,
        attempts: int,
        reason: str,
    ) -> None:
        self.url = url
        self.params = tuple(params)
        self.status = status
        self.attempts = attempts
        self.reason = reason
        shown = "none" if status is None else str(status)
        super().__init__(
            f"GET {url} {list(self.params)} failed after {attempts} "
            f"attempt{'' if attempts == 1 else 's'}; last status {shown}: {reason}"
        )


class Transport(Protocol):
    def get(
        self,
        url: str,
        params: Sequence[tuple[str, str]],
        headers: Mapping[str, str],
    ) -> Response: ...


class RequestsTransport:
    """The real transport. Built only by ``ingest.cli.main`` for a ``--fetch`` run."""

    def __init__(self, session: Any = None) -> None:
        self._session = session if session is not None else requests.Session()
        self.identity = ClientIdentity(
            user_agent=USER_AGENT, library="requests", library_version=requests.__version__
        )

    def get(
        self,
        url: str,
        params: Sequence[tuple[str, str]],
        headers: Mapping[str, str],
    ) -> Response:
        try:
            reply = self._session.get(
                url, params=list(params), headers=dict(headers), timeout=REQUEST_TIMEOUT_SECONDS
            )
            body = reply.content
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        except requests.RequestException as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}", transient=False) from exc
        return Response(
            status=reply.status_code,
            headers={name.lower(): value for name, value in reply.headers.items()},
            body=body,
        )


def retry_after_seconds(value: str | None, now: datetime) -> float | None:
    """A `Retry-After` header in seconds, or `None` when absent or unreadable.

    Both forms HTTP allows are read: whole seconds, and an HTTP date. A date in the
    past waits zero. An unreadable value is ignored rather than guessed at.
    """
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(int(text))
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - now).total_seconds())


class HttpClient:
    """GET with D44's policy. The transport is required: there is no default one."""

    def __init__(
        self,
        transport: Transport,
        *,
        user_agent: str = USER_AGENT,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._transport = transport
        self._user_agent = user_agent
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._last_start: dict[str, float] = {}

    def _pace(self, host: str) -> None:
        """Wait until at least MIN_REQUEST_INTERVAL_MS has passed since this host's last start."""
        last = self._last_start.get(host)
        if last is not None:
            wait = last + MIN_REQUEST_INTERVAL_MS / 1000 - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_start[host] = self._clock()

    def get(
        self,
        url: str,
        params: Sequence[tuple[str, str]] = (),
        *,
        validate: Callable[[Response], None] | None = None,
    ) -> Fetched:
        params = tuple((str(name), str(value)) for name, value in params)
        host = urlsplit(url).netloc.lower()
        headers = {"User-Agent": self._user_agent}
        status: int | None = None
        reason = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace(host)
            retry_after: float | None = None
            try:
                response = self._transport.get(url, params, headers)
            except TransportError as exc:
                status, reason = None, str(exc)
                if not exc.transient:
                    raise FetchFailed(
                        url, params, status=None, attempts=attempt, reason=reason
                    ) from exc
            else:
                status = response.status
                if 200 <= status < 300:
                    try:
                        if validate is not None:
                            validate(response)
                    except InvalidResponse as exc:
                        reason = f"invalid body: {exc}"
                    else:
                        record = RequestRecord(
                            url=url,
                            params=params,
                            retrieved_at=format_timestamp(self._now()),
                            status=status,
                            response_sha256=hashlib.sha256(response.body).hexdigest(),
                            response_bytes=len(response.body),
                        )
                        return Fetched(response=response, record=record)
                elif status == FORBIDDEN:
                    raise FetchFailed(
                        url,
                        params,
                        status=status,
                        attempts=attempt,
                        reason="HTTP 403 is never retried; the run stops",
                    )
                elif status in TRANSIENT_STATUSES:
                    reason = f"HTTP {status}"
                    retry_after = retry_after_seconds(
                        response.headers.get("retry-after"), self._now()
                    )
                else:
                    raise FetchFailed(
                        url,
                        params,
                        status=status,
                        attempts=attempt,
                        reason=f"HTTP {status} is not retried",
                    )
            if attempt == MAX_ATTEMPTS:
                break
            wait = float(BACKOFF_SECONDS[attempt - 1])
            if retry_after is not None:
                wait = min(max(wait, retry_after), float(RETRY_AFTER_CAP_SECONDS))
            self._sleep(wait)
        raise FetchFailed(url, params, status=status, attempts=MAX_ATTEMPTS, reason=reason)
