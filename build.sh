#!/usr/bin/env bash
set -o errexit
# Production installs the runtime tier only. Test and training packages
# (pytest, ruff, mypy, pandas, pyarrow, scikit-learn, onnxruntime) are
# deliberately absent — see requirements/base.txt.
pip install -r requirements/base.txt
python manage.py collectstatic --no-input
python manage.py migrate
python manage.py bootstrap_groups
python manage.py seed_demo
