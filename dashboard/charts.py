"""The one figure a table cannot replace, and two that earn their place.

Most results in this project are five-row tables, which is why no plotting
library was installed for the model reports. Plotly is here for the few
figures a table genuinely cannot replace, and no more.

**The reliability diagram.** Its claim is a diagonal: a forecast stated at 0.61
should happen 61% of the time, and the y = x line *is* the hypothesis. A reader
checks a model against it by eye in a way that a column of signed gaps does not
support, because the question is not "how big is the largest gap" but "does the
curve bend, and where". Nothing else here has that property.

The other two are a dot plot of scores and a bar chart of per-competition
error, and both are honest conveniences: they order things a table already
contains. The scores are dots rather than bars because they all lie near 1.0,
where a bar from zero says nothing and a bar on a cut axis misstates a length.

Every figure is built from a frame a panel hands it and computes nothing. A
chart that aggregated would be a third place a metric lives, after the pipeline
and the card.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

DIAGONAL = "#94a3b8"
MODEL = "#2563eb"
BOOKMAKER = "#f59e0b"
BARS = "#0f766e"

HEIGHT = 420


def _style(figure: go.Figure, *, title: str, x_title: str, y_title: str) -> go.Figure:
    """The look every figure here shares, applied in one place.

    A function rather than a dict of keyword arguments splatted into each
    call: `update_layout` is typed per-argument, and a `dict[str, object]`
    unpacked into it type-checks as nothing in particular — which is how a
    misspelt layout key survives to be ignored at run time.
    """
    figure.update_layout(
        template="simple_white",
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        height=HEIGHT,
        hovermode="closest",
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return figure


def reliability_diagram(table: pd.DataFrame, *, title: str = "Reliability") -> go.Figure:
    """Stated probability against observed rate, with the line being claimed.

    Marker area is proportional to the matches in a bin, because the bins are
    wildly uneven — most of a three-class football forecast sits between 0.15
    and 0.55 — and a bin holding one match should not read as loudly as one
    holding twenty thousand. That is the same weighting
    :func:`~src.evaluation.reliability.expected_calibration_error` applies, made
    visible rather than restated.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect calibration",
            line={"color": DIAGONAL, "dash": "dash", "width": 1},
            hoverinfo="skip",
        )
    )
    if not table.empty:
        figure.add_trace(
            go.Scatter(
                x=table["predicted"],
                y=table["observed"],
                mode="markers+lines",
                name="observed",
                line={"color": MODEL, "width": 2},
                marker={
                    "color": MODEL,
                    "size": _marker_sizes(table["n"]),
                    "sizemode": "area",
                    "line": {"color": "white", "width": 1},
                },
                customdata=table[["n", "gap"]].to_numpy(),
                hovertemplate=(
                    "stated %{x:.3f}<br>happened %{y:.3f}"
                    "<br>%{customdata[0]:,} matches<br>gap %{customdata[1]:+.4f}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1])
    figure.update_layout(showlegend=True)
    return _style(figure, title=title, x_title="stated probability", y_title="observed rate")


def _marker_sizes(counts: pd.Series, *, largest: int = 38, smallest: int = 6) -> pd.Series:
    """Bin counts as marker areas, with a floor so a small bin stays visible.

    A bin of one match must still be *findable* — it is often the interesting
    one, sitting far from the diagonal — so the scale has a floor rather than
    running to zero.
    """
    peak = float(counts.max()) if len(counts) else 0.0
    if peak <= 0:
        return pd.Series([smallest] * len(counts), index=counts.index)
    return smallest + (counts / peak) * (largest - smallest)


def score_bars(
    table: pd.DataFrame, *, metric: str = "log_loss", labels: Mapping[str, str] | None = None
) -> go.Figure:
    """Every forecaster's score as a dot, ordered, with the bookmaker marked.

    Dots on an axis framed around the scores, not bars from zero. Every score
    here lies between about 0.99 and 1.08, so bars from zero come out the same
    length and hide the differences the table is about. A bar's length claims a
    quantity from zero, so a bar on a cut axis would misstate it; a dot's
    position does not.

    The benchmark is coloured rather than annotated: it is the number this
    project measures itself against, and a reader should not have to find it in
    a legend.
    """
    rows = table[table[metric].notna() & (table[metric] != float("inf"))]
    colours = [BOOKMAKER if name == "bookmaker" else BARS for name in rows["forecaster"]]
    names = [(labels or {}).get(name, name) for name in rows["forecaster"]]
    figure = go.Figure(
        go.Scatter(
            x=rows[metric],
            y=names,
            mode="markers",
            marker={"color": colours, "size": 14, "line": {"color": "white", "width": 1}},
            customdata=rows[["n"]].to_numpy(),
            hovertemplate="%{y}<br>"
            + metric
            + " %{x:.4f}<br>%{customdata[0]:,} matches<extra></extra>",
        )
    )
    if not rows.empty:
        low, high = float(rows[metric].min()), float(rows[metric].max())
        pad = max((high - low) * 0.08, 0.001)
        figure.update_xaxes(range=[low - pad, high + pad])
    figure.update_yaxes(autorange="reversed")
    readable = metric.replace("_", " ")
    return _style(
        figure,
        title=f"{readable} over the matches every forecaster could price — lower is better",
        x_title=readable,
        y_title="",
    )


def competition_bars(table: pd.DataFrame, *, top: int = 15) -> go.Figure:
    """Where the model's probabilities are least honest, worst first."""
    rows = table.head(top)
    figure = go.Figure(
        go.Bar(
            x=rows["calibration_error"],
            y=rows["competition_id"],
            orientation="h",
            marker_color=BARS,
            customdata=rows[["matches"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>calibration error %{x:.4f}" "<br>%{customdata[0]:,} matches<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(autorange="reversed")
    return _style(
        figure,
        title=f"Least reliable competitions (top {top})",
        x_title="calibration error",
        y_title="",
    )
