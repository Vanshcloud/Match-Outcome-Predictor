"""The prediction log against a real PostgreSQL.

Skips without ``PREDICTION_LOG_DSN``. `docker compose up postgres` and

    PREDICTION_LOG_DSN=postgresql://predictor:predictor@127.0.0.1:5432/predictions \\
        make test-int

is what runs it. The unit suite drives every branch of this class against a
fake connection, which proves the control flow and cannot prove the SQL is
valid — that is the one thing here, and it is worth exactly one module.

**It writes to the database it is pointed at.** Rows are inserted under a match
id unique to the run and removed afterwards, so pointing this at a real log
costs a handful of rows rather than the table.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from src.storage.predictions import (
    COLUMNS,
    TABLE,
    PostgresPredictionLog,
    open_prediction_log,
    psycopg_connect,
)

pytestmark = pytest.mark.integration

DSN = os.environ.get("PREDICTION_LOG_DSN")

RUN = f"integration:{uuid.uuid4()}"
"""A match id no other run will use, so a failure leaves nothing another run
has to reason about and two runs cannot see each other's rows."""


@pytest.fixture(scope="module")
def log() -> Iterator[PostgresPredictionLog]:
    if not DSN:
        pytest.skip("no PREDICTION_LOG_DSN; `docker compose up postgres` and set it")
    opened = open_prediction_log(DSN)
    assert isinstance(opened, PostgresPredictionLog)
    try:
        yield opened
    finally:
        connection = opened.connection()
        cursor = connection.cursor()
        cursor.execute(f"DELETE FROM {TABLE} WHERE match_id = %s", [RUN])  # noqa: S608
        connection.commit()
        cursor.close()
        opened.close()


def _row(**overrides: object) -> dict[str, object]:
    return {
        "predicted_at": datetime.now(tz=UTC),
        "match_id": RUN,
        "competition_id": "ENG_1",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": "2025-03-01",
        "model": "ensemble-calibrated",
        "model_version": "0.11.0",
        "prob_home": 0.4821,
        "prob_draw": 0.2604,
        "prob_away": 0.2575,
        "in_sample": False,
        **overrides,
    }


def test_the_schema_is_created_and_creating_it_again_is_harmless(
    log: PostgresPredictionLog,
) -> None:
    """`ensure_schema` runs on every startup, so it has to be idempotent
    against a database that already has the table."""
    log.ensure_schema()
    log.ensure_schema()


def test_a_row_survives_the_round_trip_with_its_types_intact(
    log: PostgresPredictionLog,
) -> None:
    """The one thing a fake connection cannot check: that PostgreSQL accepts
    these statements and gives back what was put in."""
    assert log.record([_row()]) == 1

    written = log.recent(limit=50)
    assert list(written.columns) == list(COLUMNS)
    mine = written[written["match_id"] == RUN]
    assert len(mine) == 1

    row = mine.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["prob_home"] == pytest.approx(0.4821)
    assert bool(row["in_sample"]) is False
    assert str(row["match_date"]) == "2025-03-01"


def test_a_batch_is_written_in_one_round_trip(log: PostgresPredictionLog) -> None:
    assert log.record([_row(), _row(), _row()]) == 3
    assert len(log.recent(limit=100)[lambda frame: frame["match_id"] == RUN]) >= 4


def test_predictions_come_back_newest_first(log: PostgresPredictionLog) -> None:
    """The ordering the one consumer depends on: what did we most recently say
    about this match?"""
    recent = log.recent(limit=100)
    assert recent["predicted_at"].is_monotonic_decreasing


def test_the_log_reconnects_after_being_closed(log: PostgresPredictionLog) -> None:
    """A pooled or restarted database drops connections. The service holds one
    for the life of the process, so it has to survive that."""
    log.close()
    assert log.record([_row()]) == 1


def test_a_dsn_that_points_nowhere_fails_at_startup_rather_than_silently() -> None:
    """A configured log that cannot connect is a misconfiguration, and the
    service should refuse to come up rather than answer with no audit trail."""
    from src.storage.base import StorageError

    nowhere = PostgresPredictionLog(psycopg_connect("postgresql://x@127.0.0.1:1/none"))
    with pytest.raises(StorageError, match="could not connect"):
        nowhere.ensure_schema()
