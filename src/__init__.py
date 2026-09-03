"""Match Outcome Predictor — calibrated home/draw/away probabilities for football.

The version here is the single source of truth. pyproject.toml reads it via
``[tool.setuptools.dynamic]`` rather than repeating it, and
``tests/unit/test_version.py`` fails if any other copy drifts from it.
"""

__version__ = "0.4.0"
