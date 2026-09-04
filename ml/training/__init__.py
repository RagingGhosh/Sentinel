"""Offline training and evaluation.

Consumes the corpus produced by ``ingest`` and produces versioned artifacts.
Django-independent, and never imported by serving code: the modules Django
loads at startup stay free of scikit-learn and onnxruntime, which is what keeps
the web process inside its memory budget.
"""
