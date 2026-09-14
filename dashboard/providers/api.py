"""The model, asked over HTTP.

The dashboard does not load ``models/servable.joblib``. If it did, "what does
the model say" would have two implementations — the served one and this one —
and the day they disagreed the disagreement would be invisible, because nothing
compares them. So this provider speaks to the running service and has no
opinion about models. CI enforces the other half: ``dashboard`` may not import
``api``, and the dashboard image does not contain it.

**Every failure is an empty answer, not an exception.** The prediction section
is one of six on a page, and a reader with no service running should still get
the other five. :class:`~dashboard.client.ServiceError` is caught here and
surfaced as :attr:`ApiPredictions.error`, which the views render as a sentence
naming the command that starts the service.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dashboard.client import PredictionClient, ServiceError
from dashboard.domain.match import Fixture, MatchStatus, Prediction
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ApiPredictions:
    """The prediction service, as a provider.

    Mutable — unusually for this codebase — because :attr:`error` records what
    the last call failed with, and a view renders that instead of the section.
    A frozen version would mean either raising through the page or a second
    return value on every method.
    """

    client: PredictionClient
    name: str = "prediction-service"
    error: str | None = field(default=None)

    @property
    def available(self) -> bool:
        """Whether **this project's** service is up and has a model and a table.

        Three questions, because ``/health`` answers 200 while degraded on
        purpose: "the container is wedged" and "the model has not been built
        yet" are different problems and a person does different things about
        them. To this provider they are the same — it cannot price a fixture —
        and the detail is kept in :attr:`error` for the caption.

        The first question used to be missing, and the way that surfaced is
        worth recording. ``DASHBOARD_API_URL`` defaults to port 8000, an
        unrelated service was listening there, and its ``/health`` answered
        ``{"status": "ok"}`` — so the sidebar reported a healthy prediction
        service while every ``/predict`` would have come back 404. A liveness
        probe that accepts any 200 is a probe for "something is listening",
        which is not the question anyone was asking.
        """
        try:
            health = self.client.health()
        except ServiceError as failure:
            self.error = f"{failure}"
            return False
        if not _is_this_service(health):
            self.error = (
                f"something is answering at {self.client.base_url}, but it is not this "
                "project's API — its `/health` carries no component list. Point "
                "`DASHBOARD_API_URL` at the service `make api` starts."
            )
            return False
        if health.get("status") != "ok":
            self.error = _degraded(health)
            return False
        self.error = None
        return True

    def priceable(
        self, *, competitions: Sequence[str] | None = None, limit: int = 20
    ) -> list[Fixture]:
        """Fixtures the model has the inputs to price, most recent first.

        The service takes one competition, not a list, so a reader following
        several leagues is asked for once per league and the answers are
        merged. That is the endpoint's shape rather than this provider's, and
        the alternative — one unfiltered call, filtered here — would page
        through fixtures from thirty-nine competitions to find the reader's
        three.
        """
        wanted: list[str | None] = list(competitions) if competitions else [None]
        found: list[Fixture] = []
        for competition_id in wanted:
            try:
                rows = self.client.fixtures(competition_id=competition_id, limit=limit)
            except ServiceError as failure:
                self.error = f"{failure}"
                return []
            found.extend(to_fixtures(rows))
        self.error = None
        return sorted(found, key=lambda one: one.date, reverse=True)[:limit]

    def predict(self, match_id: str) -> Prediction | None:
        """One fixture, priced, or ``None`` with the reason in :attr:`error`."""
        try:
            answer = self.client.predict({"match_id": match_id})
        except ServiceError as failure:
            self.error = f"{failure}"
            return None
        self.error = None
        return as_prediction(match_id, answer)


def _is_this_service(health: Mapping[str, Any]) -> bool:
    """Whether that ``/health`` body is the one ``api/schemas.py`` describes.

    ``components`` is a required field of ``HealthResponse`` and is present in
    every state the service has, ready or degraded, so its absence means
    something else is on the port. Checked by shape rather than by matching a
    component's name: the set of components is a thing this project will add to
    — an odds provider, a cache — and a check that enumerated them would fail
    on the change that adds one, which is the wrong thing to be brittle
    about.
    """
    return isinstance(health.get("components"), list)


def _degraded(health: Mapping[str, Any]) -> str:
    """Which components are not ready, named the way ``/health`` names them."""
    components = health.get("components", [])
    parts = [
        f"{part.get('name')} ({part.get('detail')})"
        for part in components
        if isinstance(part, Mapping) and not part.get("ready")
    ]
    return "the service is up but not ready: " + (", ".join(parts) or "no detail given")


def as_prediction(match_id: str, answer: Mapping[str, Any]) -> Prediction:
    """A ``/predict`` body as the domain type. The one place that knows the wire."""
    fixture = answer.get("fixture")
    fixture = fixture if isinstance(fixture, Mapping) else {}
    return Prediction(
        match_id=match_id,
        probabilities=dict(answer["probabilities"]),
        model=str(answer.get("model", "")),
        model_version=str(answer.get("model_version", "")),
        in_sample=bool(answer.get("in_sample", False)),
        home_team=str(fixture.get("home_team", "")),
        away_team=str(fixture.get("away_team", "")),
        competition_id=str(fixture.get("competition_id", "")),
        date=_as_date(fixture["date"]) if fixture.get("date") else None,
    )


def to_fixtures(rows: Sequence[Mapping[str, Any]]) -> list[Fixture]:
    """``GET /fixtures`` rows as fixtures.

    Status is :attr:`~dashboard.domain.match.MatchStatus.UNKNOWN` for all of
    them: the endpoint answers "which matches can this model price" and says
    nothing about whether one has been played, so claiming either would be
    inventing the half the service withheld.
    """
    return [
        Fixture(
            match_id=str(row["match_id"]),
            competition_id=str(row["competition_id"]),
            date=_as_date(row["date"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            status=MatchStatus.UNKNOWN,
        )
        for row in rows
    ]


def _as_date(value: object) -> dt.date:
    """A date however the wire spelled it. Pydantic serialises one as text."""
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
