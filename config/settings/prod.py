from .base import *  # noqa: F403
from .base import MIDDLEWARE, env

DATABASES = {"default": env.db_url("DATABASE_URL")}

# Whitenoise serves static files in production only. STATIC_ROOT is populated
# by `collectstatic`, a deploy step, so under dev/test its absence would just
# make the middleware warn on every request; django.contrib.staticfiles
# covers those environments instead. Must sit directly after
# SecurityMiddleware, per whitenoise's own requirement.
MIDDLEWARE = [
    *MIDDLEWARE[:1],
    "whitenoise.middleware.WhiteNoiseMiddleware",
    *MIDDLEWARE[1:],
]

SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}
