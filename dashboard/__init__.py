"""A football dashboard over a prediction engine it is only ever a client of.

``streamlit run dashboard/app.py``

Milestone 12. What this package is *for* is stated in one line: it presents.
Every probability is answered by the service over HTTP, every measurement of
the model is read from a report a pipeline wrote, and every result is read from
the canonical match table. Nothing here fits, scores, or recomputes — a
dashboard that recomputed a metric would be a second number to reconcile with
the model card, and the card is the one generated from the runs that measured
the model.

**Four layers, in one direction.**

``domain``
    What a match, a forecast and a competition are. Value types, no I/O.

``providers``
    Where football comes from. One protocol per kind of source, and an
    implementation per source: the canonical table for results, the service for
    forecasts, and — until Milestone 13 — a null feed for fixtures that returns
    nothing and says why.

``services``
    Orchestration and caching. What the home page needs, assembled from three
    providers and a favourites list.

``views``
    Streamlit. Thin, and the only layer that would be rewritten if this became
    a React client reading the same API.

The arrows point one way: ``views → services → providers → domain``, and
``domain`` imports none of them. That is what makes Milestone 13 one provider
class, Milestone 14 one favourites store, and Milestone 17 one more protocol.

``dashboard`` imports ``src``. It does not import ``api``: it is a *client* of
that service, over the network, and CI enforces both halves.
"""
