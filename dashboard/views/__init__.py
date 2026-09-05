"""The pages.

One module per screen, each exposing zero-argument ``render`` functions that
:mod:`dashboard.app` binds to :class:`streamlit.Page`. Zero-argument on
purpose: ``st.Page`` takes a callable, and a page whose dependencies arrived as
arguments would have to be wrapped in a partial to be registered and could not
be run directly by ``AppTest.from_function``. Each view calls
:func:`dashboard.context.resolve` for what it needs instead.

**A view composes; it does not fetch.** No module here constructs a client,
opens a file or names a provider class. They lay :mod:`dashboard.ui` over what
:mod:`dashboard.services` assembled from :mod:`dashboard.providers`, which is
what lets Milestone 13 connect a fixture feed without any file in this package
changing.
"""
