"""Ratings: the first derived quantities in this project, and the first that
could leak.

Each model turns the canonical table into one row of pre-match numbers per
match. What makes them trustworthy is not the algorithm but the pair of probes
in :mod:`src.validation.temporal`, which prove a row could not have seen its
own match.
"""
