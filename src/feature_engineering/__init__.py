"""Feature engineering: windows over the past, and the registry that says so.

Every feature here is a summary of what happened before a match. What makes
them trustworthy is not the code but two mechanisms: the registry, which records
the canonical columns each feature reads and therefore which ones have anything
to prove, and the probes in :mod:`src.validation.temporal`, which prove it by
recomputing each builder over truncated and rewritten inputs.
"""
