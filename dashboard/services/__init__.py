"""Orchestration and caching, between the providers and the pages.

A view asks one question — *what goes on the home page*, *what is this
fixture's history* — and gets one answer. Assembling that from three providers,
a favourites list and a cache is this layer's job, and keeping it out of the
views is what lets a view be read in one screen and a service be tested without
one.

**Caching lives here and only here.** Providers do raw reads and hold no state,
because a provider that cached would be one a test has to reset between cases
and one a second front end would have to disable. Streamlit reruns the whole
script on every interaction, so the caching is real work: without it a reader
dragging a filter re-reads a ten-megabyte Parquet each time.
"""
