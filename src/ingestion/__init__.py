"""Data ingestion: provider adapters that all produce one canonical schema.

The rule this package exists to enforce is that **no downstream module ever
learns which provider a match came from**. Adapters differ; the table they
produce does not. Adding API-Football, Understat or StatsBomb later means
writing one class against :class:`~src.ingestion.base.MatchProvider` and adding
a registry entry — no change to features, models, API or dashboard.
"""
