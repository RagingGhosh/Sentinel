import importlib

import pytest
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.urls import resolve


def test_secret_key_has_no_default(monkeypatch):
    """A missing SECRET_KEY must fail loudly, not fall back to a shipped value."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ImproperlyConfigured):
        importlib.reload(importlib.import_module("config.settings.base"))


def test_debug_defaults_off(monkeypatch):
    """A misconfigured deploy must fail closed."""
    monkeypatch.setenv("SECRET_KEY", "test-key")
    monkeypatch.delenv("DEBUG", raising=False)
    module = importlib.reload(importlib.import_module("config.settings.base"))
    assert module.DEBUG is False


def test_oauth_credentials_come_from_environment(monkeypatch):
    """Credentials must never be sourced from the database."""
    monkeypatch.setenv("SECRET_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id-from-env")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret-from-env")
    module = importlib.reload(importlib.import_module("config.settings.base"))
    app = module.SOCIALACCOUNT_PROVIDERS["google"]["APP"]
    assert app["client_id"] == "id-from-env"
    assert app["secret"] == "secret-from-env"


def test_login_redirect_url_actually_resolves():
    """A future URL rename must break loudly here, not send every login to a 404."""
    match = resolve(str(settings.LOGIN_REDIRECT_URL))
    assert match.url_name == "complaint-list"
