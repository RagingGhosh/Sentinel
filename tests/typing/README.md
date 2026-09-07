Static-typing fixtures for `tests/ingest/test_source_page_protocol.py`.

These are **not** collected by pytest and are **not** in CI's mypy target list
(`mypy complaints domains ml accounts ingest`). They are type-checked only by
that test, which is deliberate: `rejects_narrowed_page.py` is *supposed* to fail
type-checking, and would break the build if mypy walked it as ordinary source.
