import os

# These two defaults must run before django.setup() below (and before any
# Django settings import). DJANGO_SETTINGS_MODULE is intentionally set here,
# via the environment, rather than as a pytest.ini_options key in
# pyproject.toml: pytest-django resolves that ini key eagerly, during
# pytest_load_initial_conftests, before this file has even been imported --
# which would try to read SECRET_KEY from the environment before the default
# below is in place and abort collection with ImproperlyConfigured. See the
# matching comment in pyproject.toml's [tool.pytest.ini_options].
os.environ.setdefault("SECRET_KEY", "test-only-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

import django  # noqa: E402

django.setup()
