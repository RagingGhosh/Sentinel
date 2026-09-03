# Phase 1 — carried-forward items

Everything here was found during Phase 1's review process, judged non-blocking,
and deliberately deferred. None of it prevents Phase 1 from working; all of it
is worth revisiting.

## Fast-follows (worth doing before Phase 2 gets underway)

**API `assign` has no terminal-state guard.** The HTML template hides the assign
control for `CLOSED` and `DUPLICATE` complaints, but `services.assign` never
raises `InvalidTransition` and the API action performs no equivalent check — so
an agent can reassign a closed complaint over the API but not through the UI.
Not a permission or service-layer bypass; the two layers simply disagree about
one edge case. The real question is whether reassigning a closed complaint is
meaningful at all; answer that, then make both layers agree.

**Seed data reads as filler.** The demo complaint titles and bodies were
generalised to be pack-agnostic during the final fix wave — further than the
defect required. They are now templated ("A newly submitted complaint awaiting
triage in Consumer Financial") rather than the concrete narratives they
replaced. Functionally correct, but the deployed demo is what a portfolio
reviewer sees. Phase 2 replaces seeded data with real ingested CFPB/311 records,
which resolves this — the only question is whether the gap matters in the
meantime.

## Test coverage gaps

- Only one illegal-transition edge is tested (`SUBMITTED → CLOSED`). A
  parametrized sweep over every `(from, to)` pair is a few lines and is more
  valuable now that the fix wave added transitions.
- `test_results_are_immutable` covers only `TriagePrediction`; `RiskScore` and
  `Match` are untested (same decorator, near-zero risk).
- `test_bootstrap_is_idempotent` asserts only the group count, not that
  permissions remain correct after a second call.
- No "leaves no partial write behind" test for `triage()` or `mark_duplicate()`,
  though `transition()` has one. Both validate before their first write, so the
  behaviour is correct by construction.
- The `InvalidTransition → messages.error` path in the views has no test. It
  became genuinely reachable only once the fix wave added the new actions.
- `complaints.change_complaint` is granted to Agent and Admin but no permission
  matrix row exercises it — and nothing in the codebase consumes it. Add the row
  or drop the grant.

## Configuration and deployment

- **`requirements.txt` is an unfiltered `pip freeze` and `build.sh` installs it
  in production** — pytest, ruff, mypy, factory_boy and friends all ship to a
  512MB instance. Worth splitting into `requirements-dev.txt` **before** Phase 3
  measures RSS, so the measurement reflects the real deployment.
- `SECURE_SSL_REDIRECT = True` will 301 a plain-HTTP health-check probe.
  `SECURE_PROXY_SSL_HEADER` should handle proxied traffic, but this is the first
  thing to check if the deployed health check reports unhealthy.
- mypy's `ignore_missing_imports = true` is global rather than scoped to the
  specific untyped packages that need it.
- CI type-checks `complaints domains ml accounts` but not `config`.

## Dead ends and cosmetics

- `ml/apps.py` is unreachable — `ml` is deliberately not in `INSTALLED_APPS`
  (it holds no models). Delete it or comment why it exists unregistered.
- `PRIORITY_RANK` in `complaints/models.py` is defined and unused. It is the
  forward-looking `priority_rank` risk feature for Phase 3.
- `Complaint.is_overdue` uses a function-local `timezone` import guarding a
  circular import that does not exist here.
- `ComplaintListView.select_related("category")` over-fetches; `list.html` never
  renders category.
- `perform_create` and serializer parameters are untyped.
- The M1 queue-ordering test asserts on the compiled SQL string rather than an
  ordering outcome, because SQLite sorts NULLs first regardless and cannot
  distinguish the two code paths. Correct, but the assertion is coupled to
  SQLite emitting a literal `NULLS FIRST` token — latent fragility if the CI
  runner's sqlite3 build changes.

## Deferred from the spec

**Request IDs on log lines** (spec §8). Phase 1 ships structured JSON logging
but no request-ID middleware, because the thing request IDs exist to correlate —
ML failures carrying complaint ID and model version — does not exist until
Phase 3. The middleware lands there.
