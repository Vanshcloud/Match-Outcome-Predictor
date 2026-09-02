"""Pipelines: the orchestration each stage of the project exposes.

A pipeline coordinates; it does not compute. Every piece of real logic lives in
a domain package (``ingestion``, ``feature_engineering``, ``models``) and is
unit-testable without one, so a pipeline stays a readable sequence of steps
rather than the place where behaviour hides.
"""
