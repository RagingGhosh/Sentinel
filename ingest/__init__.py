"""Corpus ingestion: public source APIs to a versioned on-disk corpus.

Django-independent by design. Invoked as ``python -m ingest.cli``, never as a
management command, so the application's import graph never reaches training
dependencies and no migration risk is introduced.

Serving code must not import this package. See tests/test_import_boundaries.py.
"""
