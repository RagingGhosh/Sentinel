"""Acquisition: the layer every concrete source fetcher stands on (addendum D44, D46).

What lives here, and nothing beyond it:

    canonical    the one byte form of an acquisition record, a journal line and a page
    http         the transport boundary, and D44's retry, backoff and pacing policy
    acquisition  the acquisition directory, its journal, its immutable record, and
                 the verification ``ingest()`` runs before it normalizes one
    nyc311       NYC 311's day-sliced, count-verified fetcher (Task 24, D48)
    registry     where a source registers its fetcher; NYC 311 is registered, and
                 each other source's fetcher is added by that source's own task

Source-specific fetch behaviour lives only in a source's own fetcher module. A
source's fetcher is a plain ``Fetcher`` built from a ``FetchContext``, and
``python -m ingest.cli --fetch`` refuses every source not yet registered.

Imports only the standard library and ``requests`` (D44): no pyarrow, no scipy, no
``ingest.cli`` and no Django. Tests never touch the network: every request goes
through an injected transport, and the real one is built only in ``ingest.cli.main``.
"""
