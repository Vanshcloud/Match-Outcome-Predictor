"""Splits, baselines, and the model zoo.

Nothing here reaches for a store or a file. A split is a function of a frame
and a baseline is a function of two frames, which is what lets the leakage
suite hand them truncated inputs and get a truthful answer.
"""
