"""The figures. Three properties, and none of them is about appearance.

A chart in this project computes nothing — the frames arrive built by
:mod:`dashboard.services.reports`, which itself computes nothing. So what is worth pinning
is that each figure *plots the column it claims to*, survives the empty frame a
filter can produce, and marks the benchmark rather than hiding it in a legend.
"""

from __future__ import annotations

import pandas as pd

from dashboard.charts import BOOKMAKER, competition_bars, reliability_diagram, score_bars


def reliability_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin": [1, 2, 3],
            "lower": [0.1, 0.2, 0.3],
            "upper": [0.2, 0.3, 0.4],
            "n": [10, 5000, 200],
            "predicted": [0.16, 0.25, 0.34],
            "observed": [0.17, 0.25, 0.33],
            "gap": [0.01, 0.0, -0.01],
        }
    )


def test_the_diagram_draws_the_line_it_is_testing_against() -> None:
    """The diagonal is the hypothesis, so it is always present — including on
    an empty table, where it is the only thing there is to show."""
    figure = reliability_diagram(reliability_table())
    diagonal = figure.data[0]
    assert list(diagonal.x) == [0, 1] and list(diagonal.y) == [0, 1]
    assert "perfect" in str(diagonal.name)


def test_the_diagram_plots_stated_against_observed() -> None:
    table = reliability_table()
    observed = reliability_diagram(table).data[1]
    assert list(observed.x) == list(table["predicted"])
    assert list(observed.y) == list(table["observed"])


def test_marker_area_follows_the_matches_in_a_bin() -> None:
    """The bins are wildly uneven, and a bin holding ten matches must not read
    as loudly as one holding five thousand — the same weighting the calibration
    error applies, made visible."""
    sizes = list(reliability_diagram(reliability_table()).data[1].marker.size)
    assert sizes[1] > sizes[2] > sizes[0]
    assert min(sizes) > 0


def test_an_empty_table_still_draws_the_diagonal_and_nothing_else() -> None:
    figure = reliability_diagram(pd.DataFrame(columns=["predicted", "observed", "n", "gap"]))
    assert len(figure.data) == 1


def test_a_single_bin_of_one_match_still_gets_a_visible_marker() -> None:
    """The floor on marker size. A lone bin far from the diagonal is often the
    interesting one, and a marker scaled to zero is one nobody finds."""
    lone = reliability_table().head(1).assign(n=[1])
    assert min(reliability_diagram(lone).data[1].marker.size) > 0


def test_the_bookmaker_bar_is_coloured_as_the_benchmark() -> None:
    table = pd.DataFrame(
        {
            "forecaster": ["bookmaker", "ensemble-calibrated"],
            "n": [100, 100],
            "log_loss": [0.99, 1.02],
        }
    )
    colours = list(score_bars(table).data[0].marker.color)
    assert colours[0] == BOOKMAKER and colours[1] != BOOKMAKER


def test_the_score_axis_frames_the_scores_rather_than_starting_at_zero() -> None:
    """Every score lies near 1.0. From zero, 0.9993 and 1.0751 are the same
    length and the chart says nothing; the axis is framed around them and the
    marks are dots, which claim a position rather than a length."""
    table = pd.DataFrame(
        {
            "forecaster": ["bookmaker", "class_prior"],
            "n": [100, 100],
            "log_loss": [0.9993, 1.0751],
        }
    )
    figure = score_bars(table, labels={"bookmaker": "Closing line"})
    low, high = figure.layout.xaxis.range
    assert 0.9 < low < 0.9993 and 1.0751 < high < 1.2
    assert list(figure.data[0].y) == ["Closing line", "class_prior"]


def test_an_infinite_score_is_dropped_rather_than_breaking_the_axis() -> None:
    """`home_always` scores infinite log loss and is not clipped. A bar chart
    cannot draw infinity, and clipping it here would state a number the
    evaluation layer deliberately refuses to."""
    table = pd.DataFrame(
        {
            "forecaster": ["bookmaker", "home_always"],
            "n": [100, 100],
            "log_loss": [0.99, float("inf")],
        }
    )
    assert list(score_bars(table).data[0].y) == ["bookmaker"]


def test_competition_bars_show_the_worst_first_and_are_capped() -> None:
    table = pd.DataFrame(
        {
            "competition_id": [f"C{index}" for index in range(20)],
            "matches": [500] * 20,
            "calibration_error": [0.05 - index * 0.001 for index in range(20)],
        }
    )
    figure = competition_bars(table, top=5)
    assert list(figure.data[0].y) == ["C0", "C1", "C2", "C3", "C4"]


def test_a_table_whose_bins_are_all_empty_still_produces_markers() -> None:
    """Degenerate, and reachable: a reliability table can carry a bin with no
    matches in it, and dividing by the largest count would be a division by
    zero that renders as an empty chart rather than an error."""
    table = reliability_table().assign(n=[0, 0, 0])
    sizes = list(reliability_diagram(table).data[1].marker.size)
    assert len(sizes) == 3
    assert all(size > 0 for size in sizes)
