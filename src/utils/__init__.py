"""Cross-cutting utilities: paths, configuration, logging, HTTP.

Nothing in this package imports from any other ``src`` package. It is the
bottom of the dependency graph, which is what lets every other module depend on
it without creating a cycle.
"""
