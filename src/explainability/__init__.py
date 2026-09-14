"""Where a model's forecast comes from, by feature block.

Two methods, deliberately, and they are asked the same question so that their
answers can be put in one table:

- :mod:`~src.explainability.shapley` decomposes the model's own arithmetic.
  Exact, fast, and available only for the three families whose estimator is a
  forest of trees rather than a pipeline around one.
- :mod:`~src.explainability.permutation` breaks a block and measures what the
  score loses. Model-agnostic, so it covers all six families and is the only
  one of the two that can be read against the ablation.

The third answer already exists: the ablation, which *retrains* without a
block. Three methods that agree is a check on all three, and where they
disagree the disagreement is the finding — a block a model leans on heavily and
can do without is a block whose information lives somewhere else too.

Nothing here is part of a forecast. An explanation that could change a
prediction is not an explanation.
"""
