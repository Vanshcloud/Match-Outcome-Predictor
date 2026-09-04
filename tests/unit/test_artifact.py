"""The fitted counterpart of the blend, and the guarantees it has to keep.

Fitted once here rather than per test. Every family in the zoo is fitted on
several thousand rows to build one of these, and a module that refitted for
each assertion would be a module people stop running — which is the only way a
test suite makes a codebase worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASSES
from src.models.artifact import (
    ArtifactError,
    FittedMember,
    ServableModel,
    fit_servable,
)
from src.models.calibration import NEUTRAL
from src.models.dataset import DESIGN_COLUMNS, DatasetError
from src.models.ensemble import SHIPPED
from src.models.zoo import ModelError
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

# Twenty clubs, not twelve. `MIN_HOLDOUT_MATCHES` is 380 — one English season
# — and the calibration layer cuts its holdout by *date*, at 365 days. A
# twelve-club league fields 132 fixtures a year, so its holdout would always be
# too thin and the temperature would always come back neutral: the test for the
# calibrated path would silently be a second test of the uncalibrated one.
LEAGUE = modelled_frame(seasons=season_labels(2010, 14), teams=20)

# One family standing in for the blend of three. What is under test is the
# composition — fit, blend, calibrate, predict, pickle — and fitting an MLP
# three times to assert that a probability sums to one buys nothing the
# logistic regression does not.
MEMBERS = ("logistic_regression",)

MODEL = fit_servable(LEAGUE, members=MEMBERS)


# ---- what it was fitted on ---------------------------------------------------


def test_it_records_the_window_it_could_see() -> None:
    """``trained_through`` is the whole basis of the in-sample flag. If it
    drifted from the data, every served prediction would carry a claim about
    itself that was wrong."""
    assert MODEL.trained_matches == len(LEAGUE)
    assert MODEL.trained_from == LEAGUE["date"].min()
    assert MODEL.trained_through == LEAGUE["date"].max()
    assert MODEL.name == SHIPPED
    assert MODEL.member_names == MEMBERS


def test_the_columns_come_from_the_members_rather_than_a_second_list() -> None:
    assert MODEL.columns == DESIGN_COLUMNS
    assert all(member.columns == DESIGN_COLUMNS for member in MODEL.members)


def test_a_fixture_inside_the_window_is_in_sample_and_one_after_it_is_not() -> None:
    assert MODEL.is_in_sample(MODEL.trained_through)
    assert MODEL.is_in_sample(MODEL.trained_from)
    assert not MODEL.is_in_sample(MODEL.trained_through + pd.Timedelta(1, "D"))


# ---- predicting --------------------------------------------------------------


def test_it_returns_one_row_per_fixture_summing_to_one() -> None:
    stated = MODEL.predict(LEAGUE.head(25))
    assert stated.shape == (25, len(CLASSES))
    assert np.allclose(stated.sum(axis=1), 1.0)
    assert ((stated >= 0) & (stated <= 1)).all()


def test_the_forecast_is_in_classes_order() -> None:
    """The one error that is invisible: a transposed forecast still sums to one
    and still scores plausibly. The synthetic league decides its results from
    `elo_expected_home`, so the home column has to move with it."""
    lopsided = LEAGUE.nlargest(40, "elo_expected_home")
    balanced = LEAGUE.nsmallest(40, "elo_expected_home")
    assert MODEL.predict(lopsided)[:, 0].mean() > MODEL.predict(balanced)[:, 0].mean()


def test_pricing_nothing_is_refused_rather_than_answered_with_an_empty_array() -> None:
    with pytest.raises(ArtifactError, match="no fixtures to price"):
        MODEL.predict(LEAGUE.head(0))


def test_a_missing_design_column_names_the_column() -> None:
    """Propagated from the dataset layer unchanged. A model quietly predicting
    from twenty-nine columns because the ratings were not joined is a model
    whose output means nothing."""
    with pytest.raises(DatasetError, match="elo_home"):
        MODEL.predict(LEAGUE.head(5).drop(columns=["elo_home"]))


# ---- calibration -------------------------------------------------------------


def test_a_real_holdout_produces_a_fitted_temperature() -> None:
    """Not asserted to be above one — that is a measurement, not a property.
    What has to hold is that a scalar was actually fitted rather than left at
    the neutral value because the split was too thin."""
    assert MODEL.temperature != NEUTRAL
    assert 0.2 <= MODEL.temperature <= 5.0


def test_a_thin_history_leaves_the_temperature_neutral_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The forecast is then the blend's own, uncalibrated. Reported at warning
    level, because a served model that silently stopped calibrating would score
    slightly worse and look identical."""
    small = modelled_frame(seasons=season_labels(2018, 2), teams=8)
    with caplog.at_level("WARNING"):
        thin = fit_servable(small, members=MEMBERS)
    assert thin.temperature == NEUTRAL
    assert "holdout was too thin" in caplog.text


# ---- refusals ----------------------------------------------------------------


def test_an_empty_table_cannot_be_fitted_on() -> None:
    with pytest.raises(ArtifactError, match="empty table"):
        fit_servable(LEAGUE.head(0), members=MEMBERS)


def test_a_frame_with_no_date_column_is_refused() -> None:
    """The training window is what the in-sample flag is computed from, so a
    model that could be fitted without one would be a model that could not say
    what it had seen."""
    with pytest.raises(ArtifactError, match="no 'date' column"):
        fit_servable(LEAGUE.drop(columns=["date"]), members=MEMBERS)


def test_a_frame_whose_dates_are_all_null_is_refused() -> None:
    undated = LEAGUE.head(50).assign(date=pd.Series([pd.NaT] * 50, dtype="datetime64[ns]"))
    with pytest.raises(ArtifactError, match="training window is undefined"):
        fit_servable(undated, members=MEMBERS)


# ---- the members -------------------------------------------------------------


def test_a_member_rebuilds_its_forecaster_from_the_zoo() -> None:
    """The forecaster is not stored — it holds an unpicklable lambda — so it is
    rebuilt by name. This is what keeps the CLASSES scatter as one
    implementation rather than a copy living in the artefact."""
    member = MODEL.members[0]
    assert member.forecaster.name == member.name
    assert member.forecaster.columns == member.columns


def test_a_member_naming_a_family_the_zoo_no_longer_has_fails_on_use() -> None:
    """An artefact that outlived its family should say so rather than serve
    from an estimator nothing in the codebase describes."""
    orphan = FittedMember(
        name="a_family_that_was_removed",
        columns=DESIGN_COLUMNS,
        estimator=MODEL.members[0].estimator,
    )
    stranded = ServableModel(
        name=SHIPPED,
        members=(orphan,),
        temperature=NEUTRAL,
        trained_matches=1,
        trained_from=LEAGUE["date"].min(),
        trained_through=LEAGUE["date"].max(),
    )
    with pytest.raises(ModelError, match="a_family_that_was_removed"):
        stranded.predict(LEAGUE.head(1))
