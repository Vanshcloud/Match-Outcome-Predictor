"""How good a forecast is, measured in the units the question is asked in.

Accuracy is the wrong target for a three-class problem where one class is never
the most likely outcome. A well-calibrated model that rarely *predicts* "draw"
is behaving correctly, not failing — roughly a quarter of football matches are
drawn and almost none of them are the modal outcome beforehand. So the metrics
here are proper scoring rules first and accuracy last, and the reported number
is always paired with the count of matches it was computed over.

Two modules. `metrics` scores a forecast — log loss, RPS, accuracy — and
`reliability` asks the other question a single score cannot answer: whether a
stated probability happens as often as it says. A model can be sharp and
overconfident, or timid and honest, and land on the same loss; Milestone 9's
calibration layer is measured on the difference.
"""
