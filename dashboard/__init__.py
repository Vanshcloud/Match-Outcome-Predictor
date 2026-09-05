"""The dashboard: a reader for measurements this project already made.

Milestone 12. Streamlit over the reporting layer, with one panel that asks the
API for a live probability.

**It computes nothing.** Every table here is produced by
:mod:`src.pipelines.report` or :mod:`src.pipelines.backtest` and every
probability by the service. A dashboard that recomputed a metric would be a
second number to reconcile with the model card, and the card is the one that is
generated from the runs that measured the model.

**Two data paths, deliberately.** Reports are read from disk, because they are
a measurement that already exists and a file is the honest way to read one.
Predictions come over HTTP, because the alternative is loading the artefact a
second time in a second process — and then "what does the model say" would have
two answers that could drift.

``dashboard`` imports ``src``. It does not import ``api``: it is a *client* of
that service, over the network, and CI enforces both halves.
"""
