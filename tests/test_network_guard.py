"""The suite-wide no-network guard in ``tests/conftest.py`` (plan §Q, Task 27).

Every address used here is one that cannot be a real service:

    192.0.2.1            TEST-NET-1, reserved for documentation (RFC 5737)
    2001:db8::1          the IPv6 documentation prefix (RFC 3849)
    *.invalid            a top-level domain reserved never to resolve (RFC 2606)

and no refusal is taken on trust. Each blocking test first replaces the real functions
the guard keeps with tripwires, so a refusal that reached the real resolver or the
real ``connect`` would fail the test with a different error instead of passing.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import conftest as guard

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTATION_V4 = "192.0.2.1"
DOCUMENTATION_V6 = "2001:db8::1"
NEVER_RESOLVES = "sentinel-network-guard-probe.invalid"
REFUSED = guard.GUARD_MESSAGE


class ReachedTheRealNetworkStack(Exception):
    """A refusal did not happen: the call got through to a real socket function."""


@pytest.fixture
def tripwires(monkeypatch):
    """Make every real socket function the guard keeps raise if it is ever reached."""

    def tripwire(name):
        def reached(*args, **kwargs):
            raise ReachedTheRealNetworkStack(name)

        return reached

    for name in list(guard.ORIGINAL):
        monkeypatch.setitem(guard.ORIGINAL, name, tripwire(name))


@pytest.fixture
def recorded(monkeypatch):
    """Replace the real functions with recorders, so what they would be given is seen."""
    calls = []

    def recorder(name):
        def record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return 0 if name == "connect_ex" else []

        return record

    for name in list(guard.ORIGINAL):
        monkeypatch.setitem(guard.ORIGINAL, name, recorder(name))
    return calls


@pytest.fixture
def own_hostname():
    """The machine's host name, unless it is itself a loopback name."""
    name = socket.gethostname()  # asks the operating system; not guarded, no network
    if not name or guard.is_resolvable_locally(name):
        pytest.skip(f"this machine's host name is {name!r}, which is already local")
    return name


def child_environment() -> dict[str, str]:
    """The parent's environment without anything that would steer the child's pytest."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_")) and key != "DJANGO_SETTINGS_MODULE"
    }


def run_child_pytest(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *arguments],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# --- the guard is installed ---------------------------------------------------------


def test_the_guard_is_installed_on_every_entry_point_it_names():
    for name in ("connect", "connect_ex", "sendto"):
        assert getattr(socket.socket, name).__module__ == guard.__name__, name
    for name in ("create_connection", *guard.GUARDED_RESOLVERS):
        assert getattr(socket, name).__module__ == guard.__name__, name


def test_only_the_functions_still_called_through_keep_a_real_original():
    """No real resolver other than getaddrinfo is kept, so none can be reached."""
    assert set(guard.ORIGINAL) - {"sendmsg"} == {
        "connect", "connect_ex", "sendto", "create_connection", "getaddrinfo",
    }  # fmt: skip


@pytest.mark.parametrize(
    "host",
    [None, "localhost", "LOCALHOST.", "127.0.0.1", "127.8.9.10", "::1", "[::1]",
     "::ffff:127.0.0.1", "0.0.0.0", "::", b"127.0.0.1"],
)  # fmt: skip
def test_local_hosts_are_recognised(host):
    assert guard.is_resolvable_locally(host)
    assert guard.is_local_host(host)


def test_the_empty_host_may_be_connected_to_but_never_resolved():
    assert guard.is_local_host("")
    assert not guard.is_resolvable_locally("")


@pytest.mark.parametrize(
    "host",
    [DOCUMENTATION_V4, DOCUMENTATION_V6, NEVER_RESOLVES, "10.0.0.1", "192.168.1.1",
     "169.254.1.1", "::ffff:192.0.2.1", "localhost.example.invalid", "127.0.0.1.invalid",
     "www.consumerfinance.gov", "data.cityofnewyork.us", 127, ("127.0.0.1",)],
)  # fmt: skip
def test_everything_else_is_not_local(host):
    assert not guard.is_local_host(host)
    assert not guard.is_resolvable_locally(host)


# --- 1. socket.create_connection ----------------------------------------------------


@pytest.mark.parametrize(
    "address", [(DOCUMENTATION_V4, 9), (DOCUMENTATION_V6, 9), (NEVER_RESOLVES, 443)]
)
def test_non_loopback_create_connection_is_blocked(tripwires, address):
    with pytest.raises(AssertionError, match=REFUSED):
        socket.create_connection(address, timeout=1)


# --- 2. socket.socket.connect, TCP and UDP ------------------------------------------


@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM], ids=["tcp", "udp"])
@pytest.mark.parametrize("host", [DOCUMENTATION_V4, NEVER_RESOLVES])
def test_non_loopback_socket_connect_is_blocked(tripwires, kind, host):
    with socket.socket(socket.AF_INET, kind) as sock:
        with pytest.raises(AssertionError, match=REFUSED):
            sock.connect((host, 9))
        with pytest.raises(AssertionError, match=REFUSED):
            sock.connect_ex((host, 9))


def test_non_loopback_ipv6_socket_connect_is_blocked(tripwires):
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("this machine has no IPv6 socket support")
    with sock:
        with pytest.raises(AssertionError, match=REFUSED):
            sock.connect((DOCUMENTATION_V6, 9))


def test_a_non_loopback_datagram_is_blocked_without_a_connect(tripwires):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(AssertionError, match=REFUSED):
            sock.sendto(b"probe", (DOCUMENTATION_V4, 9))
        with pytest.raises(AssertionError, match=REFUSED):
            sock.sendto(b"probe", 0, (DOCUMENTATION_V4, 9))


# --- 3. name resolution -------------------------------------------------------------


@pytest.mark.parametrize("host", [NEVER_RESOLVES, DOCUMENTATION_V4, ""])
def test_non_local_getaddrinfo_is_blocked_before_any_lookup(tripwires, host):
    # The tripwires stand in for the real resolver: reaching one would raise
    # ReachedTheRealNetworkStack, so AssertionError proves no lookup was started.
    with pytest.raises(AssertionError, match=REFUSED):
        socket.getaddrinfo(host, 443)


@pytest.mark.parametrize("resolver", ["gethostbyname", "gethostbyname_ex", "gethostbyaddr"])
@pytest.mark.parametrize("host", [NEVER_RESOLVES, DOCUMENTATION_V4, "", None])
def test_the_other_resolvers_are_blocked_before_any_lookup(tripwires, resolver, host):
    with pytest.raises(AssertionError, match=REFUSED):
        getattr(socket, resolver)(host)


def test_reverse_resolution_of_a_non_local_address_is_blocked(tripwires):
    with pytest.raises(AssertionError, match=REFUSED):
        socket.getnameinfo((DOCUMENTATION_V4, 9), 0)
    with pytest.raises(AssertionError, match=REFUSED):
        socket.getfqdn(NEVER_RESOLVES)


# --- this machine's own host name is not local --------------------------------------


def test_the_machines_own_host_name_is_not_treated_as_local(own_hostname):
    assert not guard.is_local_host(own_hostname)
    assert not guard.is_resolvable_locally(own_hostname)
    assert not guard.is_resolvable_locally(own_hostname.upper() + ".")


@pytest.mark.parametrize(
    "resolve",
    [
        lambda name: socket.getaddrinfo(name, 80),
        lambda name: socket.getaddrinfo(name, None),
        lambda name: socket.gethostbyname(name),
        lambda name: socket.gethostbyname_ex(name),
        lambda name: socket.gethostbyaddr(name),
        lambda name: socket.getnameinfo((name, 80), 0),
        lambda name: socket.getfqdn(name),
    ],
    ids=["getaddrinfo", "getaddrinfo-no-port", "gethostbyname", "gethostbyname_ex",
         "gethostbyaddr", "getnameinfo", "getfqdn-with-a-name"],
)  # fmt: skip
def test_resolving_the_machines_own_host_name_is_refused_before_any_real_call(
    tripwires, own_hostname, resolve
):
    # Every real function the guard keeps is a tripwire here. Reaching one raises
    # ReachedTheRealNetworkStack, so AssertionError proves the refusal came first.
    with pytest.raises(AssertionError, match=REFUSED):
        resolve(own_hostname)


@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM], ids=["tcp", "udp"])
def test_connecting_to_the_machines_own_host_name_is_refused(tripwires, own_hostname, kind):
    with pytest.raises(AssertionError, match=REFUSED):
        socket.create_connection((own_hostname, 9), timeout=1)
    with socket.socket(socket.AF_INET, kind) as sock:
        with pytest.raises(AssertionError, match=REFUSED):
            sock.connect((own_hostname, 9))


# --- localhost never reaches the real resolver as a name ----------------------------


def test_getaddrinfo_hands_the_real_resolver_a_literal_never_the_name_localhost(recorded):
    socket.getaddrinfo("localhost", 80)
    socket.getaddrinfo("LocalHost.", 80, socket.AF_INET6)
    socket.getaddrinfo("localhost", 80, family=socket.AF_INET6)
    socket.getaddrinfo(None, 80)
    socket.getaddrinfo("127.0.0.1", 80)
    assert [args[0] for name, args, _ in recorded] == [
        "127.0.0.1", "::1", "::1", None, "127.0.0.1",
    ]  # fmt: skip
    assert {name for name, _, _ in recorded} == {"getaddrinfo"}


def test_connect_hands_the_real_socket_a_literal_never_the_name_localhost(recorded):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(("localhost", 80))
        sock.connect_ex(("LOCALHOST", 81))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(b"x", ("localhost", 82))
        sock.sendto(b"x", 0, ("localhost", 83))
    assert [(name, args[-1]) for name, args, _ in recorded] == [
        ("connect", ("127.0.0.1", 80)),
        ("connect_ex", ("127.0.0.1", 81)),
        ("sendto", ("127.0.0.1", 82)),
        ("sendto", ("127.0.0.1", 83)),
    ]


def test_the_replaced_resolvers_answer_loopback_without_calling_anything_real(tripwires):
    assert socket.gethostbyname("localhost") == "127.0.0.1"
    assert socket.gethostbyname("127.0.0.1") == "127.0.0.1"
    assert socket.gethostbyname_ex("localhost") == ("localhost", [], ["127.0.0.1"])
    assert socket.gethostbyaddr("127.0.0.1") == ("localhost", [], ["127.0.0.1"])
    assert socket.getnameinfo(("127.0.0.1", 80), 0) == ("localhost", "80")
    assert socket.getnameinfo(("127.0.0.1", 80), socket.NI_NUMERICHOST) == ("127.0.0.1", "80")


# --- getfqdn works with no resolution at all ----------------------------------------


@pytest.mark.parametrize("argument", [(), ("",), (None,), ("  ",), ("0.0.0.0",), ("::",)])
def test_getfqdn_for_this_machine_returns_the_host_name_without_resolving(tripwires, argument):
    assert socket.getfqdn(*argument) == socket.gethostname()


def test_getfqdn_returns_a_local_name_unchanged_without_resolving(tripwires):
    assert socket.getfqdn("localhost") == "localhost"
    assert socket.getfqdn("127.0.0.1") == "127.0.0.1"


def test_djangos_cached_dns_name_works_without_resolving(tripwires):
    """The one dependency that calls getfqdn: Django's email Message-ID host name."""
    from django.core.mail.utils import CachedDnsName
    from django.utils.encoding import punycode

    assert str(CachedDnsName()) == punycode(socket.gethostname())


# --- 4. loopback still works ---------------------------------------------------------


def test_a_loopback_tcp_connection_still_works():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            server, _ = listener.accept()
            with server:
                client.sendall(b"ping")
                assert server.recv(4) == b"ping"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as direct:
            direct.settimeout(5)
            direct.connect(("localhost", port))  # rewritten to 127.0.0.1 by the guard
            server, _ = listener.accept()
            server.close()


def test_a_loopback_datagram_still_works():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"ping", receiver.getsockname())
        assert receiver.recv(4) == b"ping"


def test_local_resolution_and_socketpair_still_work():
    assert socket.getaddrinfo("127.0.0.1", 80)
    assert socket.getaddrinfo("localhost", 80)  # answered as 127.0.0.1
    assert socket.getaddrinfo(None, 80)
    left, right = socket.socketpair()  # on Windows this connects over loopback
    with left, right:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"


# --- 5 and 7. --fetch without closed_network ----------------------------------------


@pytest.mark.parametrize(
    "source, host", [("cfpb", "www.consumerfinance.gov"), ("nyc311", "data.cityofnewyork.us")]
)
def test_fetch_without_closed_network_cannot_make_a_request(
    tmp_path, monkeypatch, tripwires, source, host
):
    """``main --fetch`` builds the real transport, and this test asks for no network
    fixture at all: neither `closed_network` nor the fetch directory's guard applies
    here. The suite-wide guard alone refuses the run's first request.

    The tripwires prove the refusal came before any real lookup or connect. A fixed
    commit and NO_PROXY make the run reach its first source request deterministically.

    What the refused run leaves behind is asserted exactly (plan Task 27's acceptance):
    `Acquisition.__init__` creates the acquisition directory and takes its writer lock
    before the first request, and releases the lock when the run fails, so the *empty*
    acquisition directory remains. Nothing else may: no file of any kind, and so no
    corpus, page, partition, journal or acquisition record.

    `ingest.cli` imports scipy and, through the corpus layer, pyarrow, so this one test
    needs the training tier, exactly as `tests/ingest/test_cli.py` does. It is imported
    here rather than at module level so that every other test in this module still
    runs where only the application tier is installed.
    """
    pytest.importorskip("scipy.stats", reason="scipy arrives with requirements/ml.txt")
    pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")
    import ingest.cli as cli

    monkeypatch.chdir(tmp_path)  # the acquisition directory is relative to the cwd
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(cli, "current_commit", lambda: "c" * 40)
    corpus_root = tmp_path / "corpus"

    with pytest.raises(AssertionError, match=REFUSED) as caught:
        cli.main(
            ["--source", source, "--start", "2024-01-01", "--end", "2024-01-01",
             "--corpus-root", str(corpus_root), "--fetch"]
        )  # fmt: skip

    assert host in str(caught.value), "the refused request was the source's own"

    left_behind = sorted(tmp_path.rglob("*"))
    files = [path for path in left_behind if not path.is_dir()]
    directories = {path.relative_to(tmp_path).as_posix() for path in left_behind if path.is_dir()}
    acquisition = f"data/acquisitions/{source}/2024-01-01_2024-01-01"
    assert files == [], "a refused first request leaves no file at all"
    assert directories == {
        "data",
        "data/acquisitions",
        f"data/acquisitions/{source}",
        acquisition,
    }, "only the empty acquisition directory, and its parents, may remain"
    assert list((tmp_path / acquisition).iterdir()) == [], "the acquisition directory is empty"
    assert not corpus_root.exists(), "no corpus was written"


# --- 6. a child pytest process -------------------------------------------------------


def test_the_guard_is_in_force_in_a_child_run_of_the_real_tree():
    """A fresh interpreter collecting from ``tests/`` loads ``tests/conftest.py`` by
    itself, with no flag asking for it, and the guard's own blocking tests pass there."""
    result = run_child_pytest(
        "tests/test_network_guard.py",
        "-k",
        "create_connection_is_blocked or guard_is_installed or loopback_tcp",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert " passed" in output and "failed" not in output, output[-2000:]


def test_a_planted_outbound_attempt_fails_a_child_run(tmp_path):
    """A test that tries to leave the machine fails, with the guard's own message."""
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        textwrap.dedent(
            f"""
            import socket

            import pytest


            def test_tries_to_leave_the_machine():
                socket.create_connection(("{DOCUMENTATION_V4}", 9), timeout=1)


            def test_resolves_a_foreign_name():
                socket.getaddrinfo("{NEVER_RESOLVES}", 443)


            def test_stays_on_the_machine():
                assert socket.getaddrinfo("127.0.0.1", 80)
                with pytest.raises(AssertionError):
                    socket.create_connection(("{DOCUMENTATION_V4}", 9), timeout=1)
            """
        ),
        encoding="utf-8",
    )
    # The planted file is outside tests/, so the guard is loaded the way any pytest
    # run under tests/ loads it -- by importing tests.conftest -- named explicitly.
    result = run_child_pytest("-p", "tests.conftest", str(planted))
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-4000:]
    assert "2 failed, 1 passed" in output, output[-2000:]
    assert output.count(REFUSED) >= 2, output[-4000:]


def test_without_the_guard_the_planted_file_is_not_refused_by_it(tmp_path):
    """The control for the test above: the same child run without ``tests.conftest``
    never loads the guard, so its message above is evidence of the guard itself."""
    planted = tmp_path / "test_control.py"
    planted.write_text(
        "import socket\n\n\ndef test_guard_absent():\n"
        "    assert socket.getaddrinfo.__module__ == 'socket'\n"
        "    assert socket.create_connection.__module__ == 'socket'\n",
        encoding="utf-8",
    )
    result = run_child_pytest(str(planted))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert REFUSED not in output
