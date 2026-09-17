"""Data validation, run by the pipeline on every ingest.

Questions such as "is the table chronological", "does the result agree with
the score" and "did shot coverage collapse" are answered as a report: per-check
outcomes and severities that the CLI renders, the ingest pipeline gates on, and
the integration suite asserts against.
"""
