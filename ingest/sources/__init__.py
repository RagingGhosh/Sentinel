"""Source adapters: one module per external corpus source.

An adapter is the only place that knows a source's field names and quirks.
Everything downstream — storage, roster derivation, splits, training — works in
terms of `CorpusRecord` and the per-source outcome types from `ingest.schema`,
so adding a source means adding a module here and nothing else.

Django-independent, like the rest of `ingest/`.
"""
