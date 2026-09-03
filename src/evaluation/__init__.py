"""How good a forecast is, measured in the units the question is asked in.

Accuracy is the wrong target for a three-class problem where one class is never
the most likely outcome. A well-calibrated model that rarely *predicts* "draw"
is behaving correctly, not failing — roughly a quarter of football matches are
drawn and almost none of them are the modal outcome beforehand. So the metrics
here are proper scoring rules first and accuracy last, and the reported number
is always paired with the count of matches it was computed over.

Scoped to metrics for now. Milestone 10 adds the backtests and model cards that
sit on top of them.
"""
