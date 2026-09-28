"""Every test under tests/ingest/fetch/ runs with the network closed (D44).

Transports here are always injected fakes. The guard is autouse so that no test in
this directory can reach a socket even by mistake; `test_the_network_is_closed`
proves the guard is in force rather than assuming it.
"""

import socket

import pytest


def _refuse(*args, **kwargs):
    raise AssertionError("a fetch test attempted a network connection")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
