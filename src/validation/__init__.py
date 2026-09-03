"""Data validation: the assertions that used to live in a test file.

Milestone 2 checked the ingested table from ``tests/integration``. That was the
right place to start and the wrong place to stay: a test run is something a
developer does, and these questions — is the table chronological, does the
result agree with the score, did shot coverage collapse — are things a
*pipeline* needs to answer every time it runs, on data no test ever sees.

So the checks moved here and became a report: an object with per-check
outcomes and severities that the CLI can render, the ingest pipeline can gate
on, and the integration suite can still assert against without owning the
logic.
"""
