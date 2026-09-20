"""Offline experiments: the runnable end of the training stack.

Each module here loads a corpus through `ingest.manifest.load_corpus`, routes it
through the Task 9-13 leakage machinery, evaluates with Task 14's metrics and
writes a Task 15 artifact. Django-independent and never imported by serving.
"""
