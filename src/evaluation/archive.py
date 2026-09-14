"""Served forecasts scored against what happened, and when that means anything.

Every prediction this service answers with is written to
:mod:`src.storage.predictions`, with the model, the version
and the moment it was served. This module is the other end of that: it joins
the archive to the canonical table, scores the forecasts whose matches have
since been played, and compares the result to what the walk-forward backtest
said the same model does.

**That comparison is the only definition of drift this project will use.** Not
a distance between feature distributions, not a population-stability index over
the design matrix: those measure that the input moved, which is a hypothesis
about performance rather than a measurement of it. Here the outcome is known,
so the question "is the served model worse than the backtest said" can be asked
directly. A proxy is what you use when you cannot score the thing itself.

**The number is worthless until the archive is big enough, so this module says
how big.** Per-match log loss is a heavy-tailed quantity — over the 62,036
walk-forward forecasts its standard deviation is 0.3976, against a mean of
1.0165 — and the mean of a handful of draws from that says nothing about the
model. :func:`detectable` turns the archive's size into the smallest shift that
could be told apart from noise, and :func:`needed` runs it backwards. Both
appear beside every drift figure this project reports, because a drift number
without them is a reading of noise presented as a finding.

**Three exclusions, all of them load-bearing.**

*In-sample rows are dropped.* The shipped artefact is fitted on the whole
history, so a request for a match inside that history is answered by a model
that trained on it. Those forecasts are real and were really served — they stay
in the archive — but scoring them against the backtest would compare a
memorised answer to an out-of-sample one and report the difference as drift.

*Unresolved rows are not failures.* A forecast for a fixture that has not been
played, or that `make data` has not caught up with, has no outcome to be scored
against yet. It is counted and reported rather than dropped silently: the
difference between "we served nothing" and "we served plenty and none of it can
be scored yet" is the difference between an outage and a Tuesday.

*Repeats are collapsed to the last one.* A cached fixture priced three times is
three rows in the log — deliberately, because ``predicted_at`` is what the log
is for — but it is one forecast about one match, and counting it three times
would divide the standard error by an extra √3 the evidence does not support.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation.metrics import CLASSES, terms
from src.ingestion.base import CANONICAL_COLUMNS, TARGET_COLUMN
from src.utils.logging import get_logger

logger = get_logger(__name__)

KEY_COLUMN = "match_id"
"""What a served forecast and a played match are joined on.

Named here rather than imported from :mod:`src.models.ensemble`, which also
names it: this package is arithmetic over frames and CI holds it to importing
nothing but the canonical schema, the metrics and utils. The assertion below is
what keeps the local name honest — it is a *canonical* column, not one this
module invented.
"""

assert KEY_COLUMN in CANONICAL_COLUMNS, "the join key is not a canonical column"

FORECAST_COLUMNS: tuple[str, str, str] = ("prob_home", "prob_draw", "prob_away")
"""The three probabilities, in :data:`~src.evaluation.metrics.CLASSES` order.

The same three the prediction log stores and the same three
:data:`src.models.ensemble.FORECAST_COLUMNS` names, and the order is
load-bearing rather than incidental — it is the class order every metric in
this package accumulates over. It cannot be imported from either place without
putting the evaluation package downstream of the model or the store, so it is
declared here and pinned by a test against both.
"""

VERSION_COLUMNS: tuple[str, str] = ("model", "model_version")
"""What a drift figure is *about*.

One row of the report per fitted artefact, not one for the archive as a whole.
A log that spans a redeploy holds two models' forecasts, their means pooled
would describe neither, and "the served model got worse" is a claim about a
version.
"""

SERVED_AT = "predicted_at"

SUMMARY_COLUMNS: tuple[str, ...] = (
    *VERSION_COLUMNS,
    "logged",
    "in_sample",
    "unresolved",
    "n",
    "log_loss",
    "rps",
    "accuracy",
    "baseline",
    "spread",
    "drift",
    "detectable",
    "distinguishable",
    "first_served",
    "last_served",
)
"""The report's columns, in the order a reader reads them.

``logged`` is every row for this version, and ``n`` is the subset that could be
scored: the two are far apart in a young archive and the gap is the point.
``drift`` is meaningless without ``detectable`` beside it, which is why they are
adjacent and why ``distinguishable`` is a column rather than a sentence
somebody has to work out. ``spread`` is the σ that produced ``detectable`` —
carried rather than assumed, so a reader can see which population the error bar
came from and a page can price the archive it still needs.
"""

CONFIDENCE = 0.95
"""The level :func:`detectable` and :func:`needed` work at, two-sided.

A convention rather than a discovery, stated so that a reader comparing this
project's threshold to another one knows what they are comparing.
"""


@dataclass(frozen=True, slots=True)
class Reference:
    """What the walk-forward folds say the shipped model does, per match.

    The two numbers a drift figure is read against, taken from **one**
    population so they cannot describe different things: ``log_loss`` is what
    the served mean is compared to, and ``spread`` is what says whether the
    comparison could mean anything at the archive's size.

    Over the shipped model this is 62,036 forecasts, a mean of 1.0165 and a
    standard deviation of 0.3976 — the mean's own standard error is 0.0016, so
    treating it as known in :func:`detectable` costs nothing.
    """

    n: int
    log_loss: float
    spread: float


def reference(forecasts: pd.DataFrame, *, name: str | None = None) -> Reference:
    """The baseline and the per-match spread, measured on the fold forecasts.

    Both from the backtest rather than from the archive, on purpose: an archive
    small enough to need the sample-size question answered is an archive whose
    own standard deviation is as noisy as its mean. The model is the same one,
    so its per-match spread is the same quantity.

    Args:
        forecasts: The per-match diagnostic pass `make card` writes.
        name: Which forecaster in it to measure. ``None`` measures every row,
            which is right for a frame holding one forecaster and wrong for the
            diagnostic pass — the caller names the model, because *which model
            is shipped* is a fact about the model package rather than about
            arithmetic over a frame.

    Returns:
        A :class:`Reference`, with ``nan`` figures when the frame holds fewer
        than two rows for that forecaster — one match has no spread, and
        reporting zero would make every difference look detectable.
    """
    rows = forecasts if name is None else forecasts[forecasts["forecaster"] == name]
    if len(rows) < 2:
        return Reference(n=len(rows), log_loss=float("nan"), spread=float("nan"))
    scored = terms(rows[list(FORECAST_COLUMNS)].to_numpy(dtype=float), rows[TARGET_COLUMN])
    return Reference(
        n=len(rows),
        log_loss=float(scored["log_loss"].mean()),
        spread=float(scored["log_loss"].std(ddof=1)),
    )


def detectable(spread: float, n: int, *, confidence: float = CONFIDENCE) -> float:
    """The smallest shift in mean log loss that ``n`` forecasts could tell from noise.

    ``z · σ / √n``, the half-width of a confidence interval for a mean. The
    baseline it is compared against is treated as known rather than estimated,
    which is close enough to true here — it comes from 62,036 forecasts, whose
    own standard error is 0.0016 — and stating it is the difference between an
    approximation and a fudge.

    **The normal approximation needs a few hundred rows to be worth anything.**
    Per-match log loss is bounded below and unbounded above, so at ``n`` in the
    single digits this figure is optimistic about a distribution it is the
    wrong shape for. That is not a reason to hide it: at those sizes it already
    returns something far larger than any drift worth acting on, which is the
    correct answer arrived at by a route that is only roughly right.

    Returns:
        ``inf`` for an empty archive — nothing is distinguishable from nothing
        — and ``nan`` when the spread is unknown.
    """
    if n <= 0:
        return float("inf")
    if not np.isfinite(spread):
        return float("nan")
    return float(norm.isf((1 - confidence) / 2) * spread / np.sqrt(n))


def needed(spread: float, difference: float, *, confidence: float = CONFIDENCE) -> int:
    """How many scored forecasts it takes to see a shift of ``difference``.

    :func:`detectable` solved for ``n``, and the honest way to read a young
    archive: the question is not "has it drifted" but "could this archive tell
    me if it had".

    Measured on the shipped model's spread of 0.3976, this is 243 forecasts for
    a 0.05 shift, 2,286 for the 0.0163 that separates this model from the
    closing line, and 6,073 for 0.01.
    """
    if difference <= 0 or not np.isfinite(spread):
        raise ValueError("a detectable difference is positive and needs a finite spread")
    return int(np.ceil((norm.isf((1 - confidence) / 2) * spread / difference) ** 2))


def latest(archive: pd.DataFrame) -> pd.DataFrame:
    """One row per match and version: the last forecast served for it.

    The log is append-only and the service caches, so one fixture asked for
    three times is three identical rows with three timestamps. That is what the
    log is for and it is not deduplicated *there* — but three copies of one
    forecast are one piece of evidence about the model, and scoring them as
    three would understate the error bar on every figure downstream.

    Ordered by ``predicted_at`` and kept last, so the surviving row is the
    forecast that was current when the match kicked off.
    """
    if archive.empty:
        return archive
    ordered = archive.sort_values(SERVED_AT, kind="stable")
    return ordered.drop_duplicates([KEY_COLUMN, *VERSION_COLUMNS], keep="last")


def resolved(archive: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The archive with each match's outcome attached where there is one.

    A left join, deliberately: a forecast whose match has not been played is
    part of the archive and part of the count, and an inner join would make it
    vanish from a report whose subject is partly how much of the archive can be
    scored at all.
    """
    if archive.empty:
        return archive.assign(**{TARGET_COLUMN: pd.Series(dtype="string")})
    outcomes = matches[[KEY_COLUMN, TARGET_COLUMN]].drop_duplicates(KEY_COLUMN)
    return archive.merge(outcomes, on=KEY_COLUMN, how="left")


def scorable(resolved_archive: pd.DataFrame) -> pd.DataFrame:
    """The rows a drift figure may be computed from: played, and out of sample.

    Both filters are exclusions of forecasts that were genuinely served, and
    the counts of what each one removed are reported beside the result — see
    this module's own docstring for why each is not optional.
    """
    if resolved_archive.empty:
        return resolved_archive
    known = resolved_archive[TARGET_COLUMN].isin(CLASSES)
    return resolved_archive[known & ~resolved_archive["in_sample"].astype(bool)]


def summarise(
    archive: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    spread: float,
    baseline: float | None = None,
) -> pd.DataFrame:
    """The drift report: one row per served model version.

    Every step is visible in the output rather than folded into it — how many
    forecasts were logged, how many were dropped for being in-sample, how many
    have no outcome yet, and how many are left to score. A report that showed
    only the last of those would put a mean over eleven matches beside a mean
    over sixty-two thousand and invite them to be read as comparable.

    Args:
        archive: Rows of the prediction log, as
            :meth:`~src.storage.predictions.PostgresPredictionLog.recent`
            returns them.
        matches: The canonical table, or any frame with ``match_id`` and
            ``result``.
        spread: Per-match log-loss standard deviation, from
            :func:`reference`.
        baseline: What the backtest says this model scores. ``None`` when the
            scores table is absent, which leaves ``drift`` null rather than
            comparing against a number that is not there.

    Returns:
        One row per ``(model, model_version)``, newest first by last served.
        Empty — with :data:`SUMMARY_COLUMNS` — for an empty archive, which is
        the ordinary state of a service nobody has called yet.
    """
    joined = resolved(latest(archive), matches)
    if joined.empty:
        logger.info("prediction archive is empty; nothing to score")
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    keep = scorable(joined)
    rows = [
        _one(version, group, keep, spread=spread, baseline=baseline)
        for version, group in joined.groupby(list(VERSION_COLUMNS), sort=False)
    ]
    table = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    return table.sort_values("last_served", ascending=False, kind="stable").reset_index(drop=True)


def _one(
    version: tuple[Hashable, ...] | Hashable,
    group: pd.DataFrame,
    keep: pd.DataFrame,
    *,
    spread: float,
    baseline: float | None,
) -> dict[str, object]:
    """One version's row of the report."""
    model, model_version = version if isinstance(version, tuple) else (version, "")
    mine = keep[(keep["model"] == model) & (keep["model_version"] == model_version)]
    in_sample = int(group["in_sample"].astype(bool).sum())
    unresolved = int((~group[TARGET_COLUMN].isin(CLASSES)).sum())
    loss, rps, accuracy = _metrics(mine)
    reach = detectable(spread, len(mine))
    drift = float("nan") if baseline is None or not len(mine) else loss - float(baseline)
    return {
        "model": str(model),
        "model_version": str(model_version),
        "logged": len(group),
        "in_sample": in_sample,
        "unresolved": unresolved,
        "n": len(mine),
        "log_loss": loss,
        "rps": rps,
        "accuracy": accuracy,
        "baseline": float("nan") if baseline is None else float(baseline),
        "spread": float(spread),
        "drift": drift,
        "detectable": reach,
        # `nan > x` is False, which is the answer that belongs here: with no
        # baseline and nothing scored there is no difference to distinguish.
        "distinguishable": bool(abs(drift) > reach),
        "first_served": _when(group[SERVED_AT].min()),
        "last_served": _when(group[SERVED_AT].max()),
    }


def _metrics(rows: pd.DataFrame) -> tuple[float, float, float]:
    """Log loss, RPS and accuracy over the scorable rows, or three nulls.

    Through :func:`~src.evaluation.metrics.terms`, so a served forecast and a
    walk-forward one are scored by the same function — the two numbers are put
    side by side and subtracted, and two implementations of a log loss is the
    cheapest way to publish a drift figure that is an arithmetic difference.
    """
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    scored = terms(rows[list(FORECAST_COLUMNS)].to_numpy(dtype=float), rows[TARGET_COLUMN])
    return (
        float(scored["log_loss"].mean()),
        float(scored["rps"].mean()),
        float(scored["hit"].mean()),
    )


def _when(value: Any) -> dt.datetime:
    """A logged timestamp as a plain datetime.

    Unguarded, because ``predicted_at`` is ``NOT NULL`` in the log's schema and
    a group here always has at least one row: there is no null to handle, and a
    branch for one would be untestable code standing in for a constraint the
    database already enforces.
    """
    return pd.Timestamp(value).to_pydatetime()


def horizon(spread: float, differences: Sequence[float]) -> pd.DataFrame:
    """How many scored forecasts each of ``differences`` would take to see.

    The table that turns "the archive holds eleven rows" into something a
    reader can act on. Printed by `make archive` and rendered on the model
    page, because "not yet" is only useful with "how long" beside it.
    """
    return pd.DataFrame(
        {
            "difference": list(differences),
            "needed": [needed(spread, one) for one in differences],
        }
    )


__all__ = [
    "CONFIDENCE",
    "SUMMARY_COLUMNS",
    "VERSION_COLUMNS",
    "Reference",
    "detectable",
    "horizon",
    "latest",
    "needed",
    "reference",
    "resolved",
    "scorable",
    "summarise",
]
