"""Where the shipped model is honest, where it is not, and the card that says so.

Milestone 9 measured calibration as one pooled number, deliberately, and left
the two breakdowns that number hides. Both are here, and both come off the same
diagnostic pass over the folds:

**By class.** Pooling all three statements answers "are these probabilities
honest". Splitting them answers "which of the three is this model dishonest
about", and in a sport where a quarter of matches are drawn and a draw is
almost never the modal outcome, those have different answers.

**By competition.** Aggregate honesty hides the competitions that pay for it. A
model calibrated to 0.002 overall can be out by four times that in a league it
has fewer matches of, and that is the number a person deciding whether to trust
a forecast for a given fixture needs.

The card itself is assembled here and rendered in
:mod:`src.evaluation.model_card`, which does not know what a Parquet file is.
That split is the same one the dataset card uses, and it is what lets the
rendering be tested against four hand-written frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src import __version__
from src.evaluation.metrics import CLASSES
from src.evaluation.model_card import CARD_FILENAME, ModelCard
from src.evaluation.reliability import (
    DEFAULT_BINS,
    expected_calibration_error,
    reliability,
)
from src.ingestion.base import TARGET_COLUMN
from src.models.calibration import Calibrated
from src.models.dataset import DESIGN_COLUMNS
from src.models.ensemble import FORECAST_COLUMNS, MEMBERS, SHIPPED, ensemble
from src.models.splits import DEFAULT_FOLDS
from src.pipelines.backtest import COMMON, pooled_table
from src.pipelines.derived import persist
from src.pipelines.tables import FORECASTS_FILENAME
from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_COMPETITION_MATCHES = 200
"""Below this a competition's calibration error is a sampling artefact.

Two hundred matches is roughly six hundred statements spread over ten bins. The
smallest competitions in this table field fewer, and a card whose "least
reliable" section is a list of the smallest samples is a card that has said
nothing.
"""

WORST_COMPETITIONS = 5


def reliability_by_class(
    forecasts: pd.DataFrame, *, bins: int = DEFAULT_BINS
) -> dict[str, pd.DataFrame]:
    """One reliability table per outcome class, over the same matches."""
    priced = forecasts.dropna(subset=list(FORECAST_COLUMNS))
    stated = priced[list(FORECAST_COLUMNS)].to_numpy(dtype=float)
    return {
        label: reliability(stated, priced[TARGET_COLUMN], bins=bins, classes=(label,))
        for label in CLASSES
    }


def reliability_by_competition(
    forecasts: pd.DataFrame,
    *,
    bins: int = DEFAULT_BINS,
    minimum: int = MIN_COMPETITION_MATCHES,
) -> pd.DataFrame:
    """One row per competition: matches, and how honest the model was there.

    Sorted worst first, which is the order the question is asked in. Small
    competitions are dropped rather than reported with a wide error bar the
    table has no column for.
    """
    priced = forecasts.dropna(subset=list(FORECAST_COLUMNS))
    rows = [
        {
            "competition_id": str(competition),
            "matches": len(group),
            "calibration_error": expected_calibration_error(
                reliability(
                    group[list(FORECAST_COLUMNS)].to_numpy(dtype=float),
                    group[TARGET_COLUMN],
                    bins=bins,
                )
            ),
        }
        for competition, group in priced.groupby("competition_id")
        if len(group) >= minimum
    ]
    if not rows:
        return pd.DataFrame(columns=["competition_id", "matches", "calibration_error"])
    return (
        pd.DataFrame(rows).sort_values("calibration_error", ascending=False).reset_index(drop=True)
    )


def card_settings(columns: Sequence[str], folds: int, members: Sequence[str] = MEMBERS) -> dict:
    """How the shipped model was built, as the card's last table.

    Assembled rather than read off the object: the shipped model is a
    calibration wrapper around a mean of three estimators, so "its
    hyperparameters" is four sets of them and a scalar fitted per fold. What a
    reader needs is which three, over how many columns, with what fitted where.
    """
    return {
        "members": ", ".join(members),
        "blend": "mean of member probabilities, unweighted",
        "calibration": "one temperature per fold, fitted on the last 365 days of the training half",
        "columns": len(columns),
        "folds": folds,
        "selection": "members admitted below 0.99 error correlation, measured on the tuning slice",
    }


def build_model_card(
    scores: pd.DataFrame,
    forecasts: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    name: str = SHIPPED,
    columns: Sequence[str] = (),
    folds: int = DEFAULT_FOLDS,
    bins: int = DEFAULT_BINS,
    settings: Mapping[str, object] | None = None,
) -> ModelCard:
    """Everything the card states, gathered from the tables that measured it.

    Args:
        scores: The persisted backtest, at its finest grain.
        forecasts: The diagnostic pass for ``name``, per match.
        matches: The joined table, for the history it spans.
        name: The forecaster the card is about.
        columns: The design columns the model reads.
        folds: How many walk-forward steps produced ``scores``.
        bins: Reliability bins.
        settings: Overrides the assembled build description.
    """
    priced = forecasts[forecasts["forecaster"] == name].dropna(subset=list(FORECAST_COLUMNS))
    pooled = reliability(
        priced[list(FORECAST_COLUMNS)].to_numpy(dtype=float), priced[TARGET_COLUMN], bins=bins
    )
    logger.info("card for %s over %d priced match(es)", name, len(priced))
    return ModelCard(
        name=name,
        version=__version__,
        settings=settings if settings is not None else card_settings(columns, folds),
        scores=pooled_table(scores, COMMON),
        reliability=pooled,
        per_class=reliability_by_class(priced, bins=bins),
        worst=reliability_by_competition(priced, bins=bins).head(WORST_COMPETITIONS),
        matches=len(priced),
        competitions=int(matches["competition_id"].nunique()),
        folds=folds,
        trained_from=f"{pd.Timestamp(matches['date'].min()):%Y-%m-%d}",
        trained_to=f"{pd.Timestamp(matches['date'].max()):%Y-%m-%d}",
        columns=tuple(columns),
    )


def write_forecasts(forecasts: pd.DataFrame, destination_dir: Path) -> Path:
    """Persist the per-match diagnostic pass, with a manifest beside it.

    The rows the card computes and then throws away. They cost about five
    minutes to produce and answer the one question the scored table cannot —
    whether a stated probability happens at the rate it states — so Milestone
    12's dashboard reads them instead of recomputing them on every page load,
    and a second `make card` is the only thing that has to.

    Written through the same :func:`~src.pipelines.derived.persist` every
    derived table uses, so the provenance of a reliability diagram is checkable
    the same way the provenance of a rating is.

    Takes the destination directory rather than the reports root and the
    subdirectory name. Reaching into :mod:`src.pipelines.train` for that one
    string would import the model zoo to learn a filename, which is the import
    Milestone 11 found by watching a container fail to start.

    Args:
        forecasts: The per-match pass, as :func:`~src.models.ensemble.fold_forecasts`
            returns it.
        destination_dir: Where to write it. The caller already knows, because
            it read the scores from the same place.
    """
    return persist(
        forecasts,
        destination_dir / FORECASTS_FILENAME,
        extra={
            "kind": "forecasts",
            "forecasters": sorted(forecasts["forecaster"].unique()),
            "rows": len(forecasts),
        },
    )


def write_model_card(card: ModelCard, docs_dir: Path) -> Path:
    """Render the card and write it, creating the directory if it is absent.

    No generation timestamp in the file. The dataset card carries one because
    its subject is a download that changes under it; this one's subject is a
    version-controlled model, and a header that changes on every run turns "the
    numbers moved" into a diff nobody reads.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    destination = docs_dir / CARD_FILENAME
    destination.write_text(card.render(), encoding="utf-8")
    logger.info("wrote %s", destination)
    return destination


def shipped_forecaster(
    columns: Sequence[str] = DESIGN_COLUMNS, members: Sequence[str] = MEMBERS
) -> Calibrated:
    """The model the card is about, built the way Milestone 9 scored it."""
    return Calibrated(ensemble(members, columns))
