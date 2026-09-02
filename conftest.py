import os

os.environ.setdefault("SECRET_KEY", "test-only-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

import django  # noqa: E402

django.setup()
