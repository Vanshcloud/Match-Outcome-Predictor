"""Test suite.

The whole suite runs offline. Nothing here reaches the network, reads a
developer's ``.env``, or needs a prior download — a suite that only passes on a
machine already carrying 250k downloaded matches is a suite nobody can trust,
and CI runs on a clean checkout every time.
"""
