"""The suite-wide no-network guard (plan §Q, Task 27).

Plan §Q: "Network is never touched in tests ... and a test asserts no test performs
an outbound request." Tasks 24 and 25 registered real fetchers for both sources, and
``ingest.cli.main`` builds a real ``RequestsTransport`` for ``--fetch``, so one test
that forgets a per-test guard would contact a real source. This module closes that
door for every test under ``tests/``, in every process that loads it -- the main run
and any child pytest run alike -- because the guard is installed when the module is
imported, before collection, and is never removed.

What it does. It replaces the Python-level entry points of the ``socket`` module that
start an outbound connection or a name lookup. Each replacement refuses any target
that is not local by raising ``AssertionError`` *before* any real socket function is
called, so no packet and no name-service query is started:

    socket.create_connection        TCP connect by (host, port)
    socket.socket.connect           TCP and UDP connect
    socket.socket.connect_ex        the same, error-code form
    socket.socket.sendto            connectionless (UDP) send to an address
    socket.socket.sendmsg           the same, where the platform has it
    socket.getaddrinfo              name resolution
    socket.gethostbyname            name resolution
    socket.gethostbyname_ex         name resolution
    socket.gethostbyaddr            reverse resolution
    socket.getnameinfo              reverse resolution
    socket.getfqdn                  replaced outright; see below

**The real resolver is only ever given ``None`` or a numeric loopback literal**, for
which no lookup of any kind exists to perform. Three consequences:

  * "Local" means no host (``None``), ``localhost``, a loopback address
    (``127.0.0.0/8``, ``::1``, or either as an IPv4-mapped IPv6 address) and the
    unspecified address (``0.0.0.0``, ``::``). A connect may also name the empty
    string, which the socket layer reads as the unspecified address without a lookup,
    and may use a Unix-domain socket.
  * ``localhost`` never reaches the real resolver as a name. It is rewritten to
    ``127.0.0.1``, or to ``::1`` for an IPv6 request, first. ``gethostbyname``,
    ``gethostbyname_ex``, ``gethostbyaddr`` and ``getnameinfo`` never call their real
    counterparts at all: for a local target they return a fixed loopback answer.
  * **This machine's own host name is not local.** Resolving it can query DNS, LLMNR
    or mDNS, so every resolver above refuses it like any other name.

``socket.getfqdn`` is the one function that legitimately needs the host name: Django
calls it with no argument to build an email Message-ID. The standard-library version
resolves the name through ``gethostbyaddr``. The replacement here does not: with no
argument it returns ``socket.gethostname()``, which asks the operating system and not
the network; a local name comes back unchanged; anything else is refused.

Binding, listening and accepting are untouched: the guard is about outbound traffic.

What it does not do. **This is not an operating-system network sandbox.** It cannot
see, and therefore does not stop:

  * a process that is not a pytest run loading this file -- ``git``, ``curl``, or a
    plain ``python script.py`` started by a test;
  * a C extension or library that opens sockets without going through the ``socket``
    module's Python attributes, and any code that calls ``_socket`` directly;
  * asyncio on Windows connecting to an IP *literal* through the proactor event loop,
    which uses overlapped I/O (``ConnectEx``) rather than ``socket.connect``; a
    connection by host name is still stopped at ``getaddrinfo``;
  * a socket inherited already connected, or one connected before this file loaded;
  * code that saved the original functions before this module was imported, or that
    deliberately restores them.

The directory-level guard in ``tests/ingest/fetch/conftest.py`` and the opt-in
``closed_network`` fixture in ``tests/ingest/test_cli.py`` are kept as they are. They
are stricter inside their own scope (they refuse loopback too) and, being applied per
test on top of this one, they take precedence there.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any, NoReturn

GUARD_MESSAGE = "the test suite's network guard"
"""The start of every refusal. It contains the word "network", which the older,
narrower guards' tests match on."""

LOOPBACK_V4 = "127.0.0.1"
LOOPBACK_V6 = "::1"

ORIGINAL: dict[str, Callable[..., Any]] = {}
"""The real functions this guard still calls, by name: the socket methods, plus
``create_connection`` and ``getaddrinfo``. No other resolver has an entry, because no
other real resolver is ever called. The guard's own tests replace these entries with
tripwires to prove that a refusal never reaches them."""


def _text(host: object) -> str | None:
    """`host` as a lower-cased name without a trailing dot, or `None` if it is not text."""
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    if not isinstance(host, str):
        return None
    return host.strip().lower().rstrip(".")


def _loopback_literal(name: str) -> bool:
    """Whether `name` is a numeric loopback or unspecified address."""
    literal = name.strip("[]").partition("%")[0]  # "[::1]" and a "%scope" suffix
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_loopback or address.is_unspecified


def is_resolvable_locally(host: object) -> bool:
    """Whether a resolver may be asked about `host`: nothing, ``localhost`` or loopback.

    The empty string and this machine's own host name are not: both can start a
    name-service query.
    """
    if host is None:
        return True
    name = _text(host)
    return name is not None and (name == "localhost" or _loopback_literal(name))


def is_local_host(host: object) -> bool:
    """Whether a connect or send to `host` stays on this machine.

    `is_resolvable_locally`, plus the empty string, which the socket layer reads as the
    unspecified address without resolving anything.
    """
    return is_resolvable_locally(host) or _text(host) == ""


def _numeric(host: Any, family: int) -> Any:
    """`host` with ``localhost`` replaced by a loopback literal, so no name is looked up."""
    if _text(host) == "localhost":
        return LOOPBACK_V6 if family == socket.AF_INET6 else LOOPBACK_V4
    return host


def _refuse(action: str, target: object) -> NoReturn:
    raise AssertionError(
        f"{GUARD_MESSAGE} refused {action} {target!r}: a test must not open an outbound "
        "network connection or resolve a non-local name (plan §Q, Task 27). Inject a fake "
        "transport, or use a loopback address."
    )


def _local_address(sock: socket.socket, address: Any) -> Any:
    """`address` made numeric if a connect or send to it stays local; otherwise refuse."""
    unix = getattr(socket, "AF_UNIX", None)
    if unix is not None and sock.family == unix:
        return address
    if (
        sock.family in (socket.AF_INET, socket.AF_INET6)
        and isinstance(address, tuple)
        and address
        and is_local_host(address[0])
    ):
        return (_numeric(address[0], sock.family), *address[1:])
    # An address family this guard does not understand is never assumed to be local.
    _refuse("a connection or datagram to", address)


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    return ORIGINAL["connect"](self, _local_address(self, address))


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    return ORIGINAL["connect_ex"](self, _local_address(self, address))


def _guarded_sendto(self: socket.socket, data: Any, *rest: Any) -> Any:
    # sendto(data, address) or sendto(data, flags, address): the address is last.
    if not rest:
        _refuse("a datagram to", None)
    return ORIGINAL["sendto"](self, data, *rest[:-1], _local_address(self, rest[-1]))


def _guarded_sendmsg(self: socket.socket, buffers: Any, *rest: Any) -> Any:
    # sendmsg(buffers[, ancdata[, flags[, address]]]). No address means a connected
    # socket, whose connect was already checked.
    if len(rest) >= 3 and rest[2] is not None:
        rest = (*rest[:2], _local_address(self, rest[2]), *rest[3:])
    return ORIGINAL["sendmsg"](self, buffers, *rest)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    host = address[0] if isinstance(address, tuple) and address else address
    if not is_local_host(host):
        _refuse("a connection to", address)
    # The real function resolves through socket.getaddrinfo and connects through
    # socket.socket.connect, both of which are the guarded versions above and below.
    return ORIGINAL["create_connection"](address, *args, **kwargs)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if not is_resolvable_locally(host):
        _refuse("resolving", host)
    family = kwargs.get("family", args[1] if len(args) > 1 else 0)
    return ORIGINAL["getaddrinfo"](_numeric(host, family), *args, **kwargs)


def _loopback_answer(name: Any) -> str:
    """The address a local `name` stands for, without asking any resolver."""
    text = _text(name)
    if text is None or text == "localhost":
        return LOOPBACK_V4
    return text.strip("[]")


def _guarded_gethostbyname(name: Any) -> str:
    if name is None or not is_resolvable_locally(name):
        _refuse("resolving", name)
    return _loopback_answer(name)


def _guarded_gethostbyname_ex(name: Any) -> tuple[str, list[str], list[str]]:
    if name is None or not is_resolvable_locally(name):
        _refuse("resolving", name)
    return ("localhost", [], [_loopback_answer(name)])


def _guarded_gethostbyaddr(name: Any) -> tuple[str, list[str], list[str]]:
    if name is None or not is_resolvable_locally(name):
        _refuse("resolving", name)
    return ("localhost", [], [_loopback_answer(name)])


def _guarded_getnameinfo(sockaddr: Any, flags: int) -> tuple[str, str]:
    host = sockaddr[0] if isinstance(sockaddr, tuple) and len(sockaddr) >= 2 else None
    if host is None or not is_resolvable_locally(host):
        _refuse("resolving", sockaddr)
    numeric = bool(flags & socket.NI_NUMERICHOST)
    return (_loopback_answer(host) if numeric else "localhost", str(sockaddr[1]))


def _guarded_getfqdn(name: str | None = "") -> str:
    """A network-free ``socket.getfqdn``.

    The standard-library version resolves the host name through ``gethostbyaddr``.
    This one never resolves anything: no argument, ``None`` or the empty string (or
    the unspecified address, which the standard library treats the same way) gives
    ``gethostname()``; a local name is returned unchanged; and any other name -- this
    machine's own included -- is refused.
    """
    name = (name or "").strip()
    if not name or name in ("0.0.0.0", "::"):
        return socket.gethostname()  # asks the operating system, not the network
    if not is_resolvable_locally(name):
        _refuse("resolving", name)
    return name


_SOCKET_METHODS: dict[str, Callable[..., Any]] = {
    "connect": _guarded_connect,
    "connect_ex": _guarded_connect_ex,
    "sendto": _guarded_sendto,
    "sendmsg": _guarded_sendmsg,
}
_CALLED_THROUGH: dict[str, Callable[..., Any]] = {
    "create_connection": _guarded_create_connection,
    "getaddrinfo": _guarded_getaddrinfo,
}
_REPLACED_OUTRIGHT: dict[str, Callable[..., Any]] = {
    "gethostbyname": _guarded_gethostbyname,
    "gethostbyname_ex": _guarded_gethostbyname_ex,
    "gethostbyaddr": _guarded_gethostbyaddr,
    "getnameinfo": _guarded_getnameinfo,
    "getfqdn": _guarded_getfqdn,
}
GUARDED_RESOLVERS = ("getaddrinfo", *_REPLACED_OUTRIGHT)
"""Every name-resolution entry point this guard covers."""


def install() -> None:
    """Install the guard. Idempotent; never undone, so no later test can run without it."""
    if ORIGINAL:
        return
    for name, guarded in _SOCKET_METHODS.items():
        if hasattr(socket.socket, name):  # sendmsg does not exist on every platform
            ORIGINAL[name] = getattr(socket.socket, name)
            setattr(socket.socket, name, guarded)
    for name, guarded in _CALLED_THROUGH.items():
        ORIGINAL[name] = getattr(socket, name)
        setattr(socket, name, guarded)
    for name, guarded in _REPLACED_OUTRIGHT.items():
        setattr(socket, name, guarded)  # the real function is deliberately not kept


install()
