"""D44's transport boundary and its failure, retry and pacing policy.

Every transport here is a scripted fake and every clock is a fake one, so the waits
are asserted exactly and nothing sleeps. The directory's conftest closes the network.
"""

import ast
import hashlib
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests

from ingest.fetch import http
from ingest.fetch.http import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    MIN_REQUEST_INTERVAL_MS,
    RETRY_AFTER_CAP_SECONDS,
    USER_AGENT,
    FetchFailed,
    HttpClient,
    InvalidResponse,
    RequestsTransport,
    Response,
    TransportError,
    policy,
    retry_after_seconds,
)

ROOT = Path(__file__).resolve().parents[3]
URL = "https://example.test/api"
NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """Monotonic seconds that move only when something sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedTransport:
    """Replays a script of responses or exceptions, recording every call."""

    def __init__(self, script, clock=None):
        self.script = list(script)
        self.calls = []
        self.clock = clock

    def get(self, url, params, headers):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": dict(headers),
                "at": self.clock() if self.clock else None,
            }
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def ok(body=b"[]", headers=None):
    return Response(status=200, headers=headers or {}, body=body)


def status(code, headers=None):
    return Response(status=code, headers=headers or {}, body=b"")


def client(script):
    clock = FakeClock()
    transport = ScriptedTransport(script, clock)
    return HttpClient(transport, clock=clock, sleep=clock.sleep, now=lambda: NOW), transport, clock


# --- the policy values ---------------------------------------------------------------


def test_the_policy_values_are_exactly_d44s():
    assert MAX_ATTEMPTS == 6
    assert BACKOFF_SECONDS == (2, 4, 8, 16, 32)
    assert RETRY_AFTER_CAP_SECONDS == 300
    assert MIN_REQUEST_INTERVAL_MS == 1000
    assert policy() == {
        "backoff_seconds": [2, 4, 8, 16, 32],
        "max_attempts": 6,
        "min_request_interval_ms": 1000,
        "retry_after_cap_seconds": 300,
    }


def test_the_network_is_closed():
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.test", 443))


def test_no_client_can_be_built_without_a_transport():
    with pytest.raises(TypeError):
        HttpClient()  # type: ignore[call-arg]


# --- success -------------------------------------------------------------------------


def test_a_success_returns_the_response_and_its_request_record():
    fetch, transport, clock = client([ok(b'[{"a":1}]')])
    fetched = fetch.get(URL, [("$limit", "50000"), ("$order", "unique_key")])

    assert fetched.response.body == b'[{"a":1}]'
    record = fetched.record
    assert record.url == URL
    assert record.params == (("$limit", "50000"), ("$order", "unique_key"))
    assert record.status == 200
    assert record.retrieved_at == "2026-09-28T12:00:00.000000+00:00"
    assert record.response_sha256 == hashlib.sha256(b'[{"a":1}]').hexdigest()
    assert record.response_bytes == len(b'[{"a":1}]')
    assert record.as_record()["params"] == [["$limit", "50000"], ["$order", "unique_key"]]
    assert clock.sleeps == []
    assert len(transport.calls) == 1


def test_the_one_user_agent_names_sentinel_and_imitates_no_other_client():
    fetch, transport, _ = client([ok()])
    fetch.get(URL)
    sent = transport.calls[0]["headers"]
    assert sent == {"User-Agent": USER_AGENT}
    assert "Sentinel" in USER_AGENT
    for imitation in ("Mozilla", "Chrome", "Safari", "curl", "python-requests"):
        assert imitation not in USER_AGENT


# --- transient failures ----------------------------------------------------------------


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_each_transient_status_is_retried_after_two_seconds(code):
    fetch, transport, clock = client([status(code), ok()])
    assert fetch.get(URL).response.status == 200
    assert len(transport.calls) == 2
    assert clock.sleeps == [2]


def test_the_backoff_is_2_4_8_16_32_with_no_jitter():
    fetch, transport, clock = client([status(503)] * 5 + [ok()])
    fetch.get(URL)
    assert clock.sleeps == [2, 4, 8, 16, 32]
    assert len(transport.calls) == 6


def test_six_failed_attempts_raise_fetch_failed_naming_the_request_and_last_status():
    fetch, transport, clock = client([status(500)] * 5 + [status(502)])
    with pytest.raises(FetchFailed) as caught:
        fetch.get(URL, [("page", "7")])
    assert len(transport.calls) == MAX_ATTEMPTS == 6
    assert clock.sleeps == [2, 4, 8, 16, 32], "no wait after the last attempt"
    failure = caught.value
    assert failure.status == 502 and failure.attempts == 6
    assert URL in str(failure) and "page" in str(failure) and "502" in str(failure)


def test_a_connection_error_is_transient():
    fetch, transport, clock = client([TransportError("reset"), ok()])
    assert fetch.get(URL).response.status == 200
    assert clock.sleeps == [2]


def test_a_non_transient_transport_error_is_never_retried():
    fetch, transport, clock = client([TransportError("bad url", transient=False)])
    with pytest.raises(FetchFailed) as caught:
        fetch.get(URL)
    assert len(transport.calls) == 1 and clock.sleeps == []
    assert caught.value.status is None


def test_a_body_the_validator_rejects_is_retried_then_fails():
    def validate(response):
        if response.body != b"good":
            raise InvalidResponse("short body")

    fetch, transport, clock = client([ok(b"bad"), ok(b"good")])
    assert fetch.get(URL, validate=validate).response.body == b"good"
    assert clock.sleeps == [2]

    fetch, transport, _ = client([ok(b"bad")] * 6)
    with pytest.raises(FetchFailed, match="short body"):
        fetch.get(URL, validate=validate)
    assert len(transport.calls) == 6


# --- permanent failures ----------------------------------------------------------------


def test_a_403_stops_at_once_and_is_never_retried():
    fetch, transport, clock = client([status(403), ok()])
    with pytest.raises(FetchFailed, match="403") as caught:
        fetch.get(URL)
    assert len(transport.calls) == 1
    assert clock.sleeps == []
    assert caught.value.status == 403 and caught.value.attempts == 1


def test_a_403_after_retries_still_stops_at_once():
    fetch, transport, clock = client([status(503), status(403), ok()])
    with pytest.raises(FetchFailed, match="403"):
        fetch.get(URL)
    assert len(transport.calls) == 2
    assert clock.sleeps == [2]


@pytest.mark.parametrize("code", [400, 401, 404, 410, 418, 301, 501, 505])
def test_every_other_status_is_permanent(code):
    fetch, transport, clock = client([status(code), ok()])
    with pytest.raises(FetchFailed) as caught:
        fetch.get(URL)
    assert len(transport.calls) == 1 and clock.sleeps == []
    assert caught.value.status == code


# --- Retry-After -------------------------------------------------------------------


def test_a_longer_retry_after_replaces_the_backoff():
    fetch, _, clock = client([status(429, {"retry-after": "10"}), ok()])
    fetch.get(URL)
    assert clock.sleeps == [10]


def test_a_shorter_retry_after_keeps_the_backoff():
    fetch, _, clock = client([status(429, {"retry-after": "1"}), ok()])
    fetch.get(URL)
    assert clock.sleeps == [2]


def test_retry_after_is_capped_at_300_seconds():
    fetch, _, clock = client([status(503, {"retry-after": "86400"}), ok()])
    fetch.get(URL)
    assert clock.sleeps == [300]


def test_retry_after_as_an_http_date():
    later = (NOW + timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    fetch, _, clock = client([status(503, {"retry-after": later}), ok()])
    fetch.get(URL)
    assert clock.sleeps == [45]


def test_an_unreadable_retry_after_is_ignored():
    assert retry_after_seconds("soon", NOW) is None
    assert retry_after_seconds(None, NOW) is None
    past = (NOW - timedelta(minutes=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert retry_after_seconds(past, NOW) == 0


# --- pacing --------------------------------------------------------------------------


def test_two_requests_to_one_host_start_at_least_one_second_apart():
    fetch, transport, clock = client([ok(), ok()])
    fetch.get(URL)
    fetch.get(URL + "/other")
    assert clock.sleeps == [1.0]
    assert transport.calls[1]["at"] - transport.calls[0]["at"] >= 1.0


def test_pacing_is_per_host():
    fetch, transport, clock = client([ok(), ok()])
    fetch.get("https://one.test/a")
    fetch.get("https://two.test/a")
    assert clock.sleeps == []


def test_a_backoff_already_satisfies_the_pacing():
    fetch, transport, clock = client([status(500), ok()])
    fetch.get(URL)
    assert clock.sleeps == [2], "no extra pacing wait after a two-second backoff"


# --- the real transport, without the network ---------------------------------------


class FakeReply:
    def __init__(self, status_code, content, headers):
        self.status_code = status_code
        self.content = content
        self.headers = headers


class FakeSession:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_the_requests_transport_returns_status_lowercased_headers_and_body():
    session = FakeSession(FakeReply(200, b"[1]", {"Retry-After": "3", "Content-Type": "x"}))
    transport = RequestsTransport(session=session)
    response = transport.get(URL, [("a", "1")], {"User-Agent": USER_AGENT})
    assert response == Response(
        status=200, headers={"retry-after": "3", "content-type": "x"}, body=b"[1]"
    )
    assert session.calls[0]["params"] == [("a", "1")]
    assert session.calls[0]["timeout"] == http.REQUEST_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "error, transient",
    [
        (requests.ConnectionError("down"), True),
        (requests.Timeout("slow"), True),
        (requests.exceptions.ChunkedEncodingError("cut"), True),
        (requests.exceptions.InvalidURL("bad"), False),
    ],
)
def test_the_requests_transport_classifies_its_errors(error, transient):
    transport = RequestsTransport(session=FakeSession(error))
    with pytest.raises(TransportError) as caught:
        transport.get(URL, [], {})
    assert caught.value.transient is transient


def test_the_requests_transport_names_itself():
    identity = RequestsTransport(session=FakeSession(None)).identity
    assert identity.user_agent == USER_AGENT
    assert identity.library == "requests"
    assert identity.library_version == requests.__version__


def test_the_real_transport_is_built_only_in_the_command_lines_main():
    """D44: nothing but ``ingest.cli.main`` constructs a RequestsTransport."""
    builders = []
    for path in sorted((ROOT / "ingest").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "RequestsTransport"
                ):
                    builders.append((path.relative_to(ROOT).as_posix(), function.name))
    assert builders == [("ingest/cli.py", "main")]


# --- Retry-After accepts ASCII digits only (D47) ---------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("60", 60),
        ("  60  ", 60),
        ("\t7\n", 7),
        ("0", 0),
        ("²", None),
        ("٦٠", None),
        ("６０", None),
        ("", None),
        ("   ", None),
        ("abc", None),
        ("6 0", None),
        ("+60", None),
        ("-3", None),
        ("1.5", None),
    ],
)
def test_retry_after_reads_ascii_digits_only(value, expected):
    assert retry_after_seconds(value, NOW) == expected


def test_a_non_ascii_retry_after_falls_back_to_the_backoff():
    fetch, transport, clock = client([status(503, {"retry-after": "²"}), ok()])
    fetch.get(URL)
    assert clock.sleeps == [2]
    assert len(transport.calls) == 2
