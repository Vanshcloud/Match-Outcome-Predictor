"""What a request may say and what a response will say.

Pydantic v2 models, which are also the OpenAPI document: the schema at
``/openapi.json`` is generated from these classes rather than written beside
them, so a field that changes shape cannot leave the documentation describing
the old one.

**Every response names its model and its version.** A probability with no
provenance is a number someone will still be quoting after the model that
produced it has been replaced twice.

**`in_sample` is on the response, not hidden.** The served model is fitted on
the whole history, so a fixture inside that history was trained on and its
probability is not the out-of-sample thing the walk-forward folds measured.
Reporting it costs one boolean; omitting it would let the two be quoted
interchangeably, which is the single most likely way a number from this service
ends up overstated.
"""

from __future__ import annotations

# `import datetime as dt` rather than `from datetime import date`. Two models
# here have a field genuinely called ``date``, and pydantic resolves a string
# annotation against the *module* namespace — where the field name would shadow
# the type and ``date | None`` becomes an unsupported operation on a FieldInfo.
# The failure is at class construction, which is at least loud; the qualified
# name removes the collision rather than working around it.
import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.dataset import DESIGN_COLUMNS


class _Schema(BaseModel):
    """Base for every model here.

    ``extra="forbid"`` on a *request* is the point: a caller who sends
    ``home`` where the field is ``home_team`` gets a 422 naming the field
    rather than a prediction about whatever the default resolved to.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    """``protected_namespaces=()`` because several fields here are genuinely
    called ``model_something`` — the served model's name and version — and
    pydantic otherwise warns that they shadow its own ``model_`` methods. The
    alternative is renaming a field in the public API to avoid a warning about
    a collision that does not exist."""


# ---- requests ---------------------------------------------------------------


class FixtureRequest(_Schema):
    """One match, named either by its id or by what a fixture list shows.

    Both are accepted because both are what a caller actually holds: an id if
    they got it from ``/fixtures``, and the competition, the two clubs and the
    date if they are reading a league table. Exactly one of the two forms must
    be complete, which is checked here rather than in the handler — a request
    that is half of each is a request nobody can answer, and it should fail
    with a 422 that says so rather than a 404 that implies the match is
    missing.
    """

    match_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Canonical match id, as returned by /fixtures. Sixteen hex "
            "characters, derived from the natural key — not a readable "
            "composite, and not something to construct by hand."
        ),
        examples=["e005dcc0757e9c04"],
    )
    competition_id: str | None = Field(
        default=None, min_length=1, max_length=50, examples=["ENG_1"]
    )
    home_team: str | None = Field(default=None, min_length=1, max_length=120, examples=["Arsenal"])
    away_team: str | None = Field(default=None, min_length=1, max_length=120, examples=["Chelsea"])
    date: dt.date | None = Field(default=None, description="Kick-off date, YYYY-MM-DD.")

    @model_validator(mode="after")
    def _one_complete_form(self) -> FixtureRequest:
        by_id = self.match_id is not None
        natural = (self.competition_id, self.home_team, self.away_team, self.date)
        by_key = all(part is not None for part in natural)
        if by_id == by_key:
            raise ValueError(
                "name the fixture exactly one way: `match_id`, or all four of "
                "`competition_id`, `home_team`, `away_team` and `date`"
            )
        return self

    def as_lookup(self) -> dict[str, str | None]:
        """The arguments :meth:`src.pipelines.serving.FixtureIndex.resolve` takes."""
        return {
            "match_id": self.match_id,
            "competition_id": self.competition_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "date": self.date.isoformat() if self.date is not None else None,
        }


class BatchRequest(_Schema):
    """Several fixtures at once, priced in one pass through the model.

    The upper bound is configuration (``API_MAX_BATCH``) rather than a literal
    here, so it is enforced where the request is handled and reported in the
    error rather than baked into the schema at import time.
    """

    fixtures: list[FixtureRequest] = Field(min_length=1)


# ---- responses --------------------------------------------------------------


class Probabilities(_Schema):
    """The forecast. Three named fields, summing to one.

    Named rather than a three-element array. An array in CLASSES order is the
    single easiest thing in this project to read wrongly, and it goes on
    summing to one while it is wrong.
    """

    home: float = Field(ge=0.0, le=1.0, examples=[0.4821])
    draw: float = Field(ge=0.0, le=1.0, examples=[0.2604])
    away: float = Field(ge=0.0, le=1.0, examples=[0.2575])


class Fixture(_Schema):
    """The match a prediction is about, as a caller would recognise it.

    No scoreline and no odds: the first would answer a different question, and
    the second is the benchmark this project measures itself against — handing
    it back beside a forecast invites exactly the comparison the model card
    says not to make.
    """

    match_id: str
    competition_id: str
    competition: str | None = None
    country: str | None = None
    season: str | None = None
    date: dt.date
    home_team: str
    away_team: str


class PredictionResponse(_Schema):
    """One fixture, priced."""

    fixture: Fixture
    probabilities: Probabilities
    model: str = Field(description="The forecaster, by the name it is scored under.")
    model_version: str
    in_sample: bool = Field(
        description=(
            "True when the fixture is inside the served model's training window. "
            "Such a probability is in-sample and is not the out-of-sample number "
            "docs/MODEL_CARD.md reports."
        )
    )
    predicted_at: dt.datetime
    limitations_url: str = Field(
        default="/model-card/limitations",
        description="Where this model must not be used, in full.",
    )


class BatchResponse(_Schema):
    """Several fixtures, priced, plus the ones that could not be found.

    Partial success rather than all-or-nothing. A batch of fifty in which one
    club is misspelled should return forty-nine forecasts and name the
    fiftieth, not fail entirely — and the caller needs to be told which one,
    which is what ``unresolved`` is for.
    """

    predictions: list[PredictionResponse]
    unresolved: list[FixtureRequest] = Field(
        default_factory=list,
        description="Fixtures the table holds no row for. Never silently dropped.",
    )
    recorded: int = Field(
        default=0, description="How many predictions the prediction log accepted."
    )


class FixtureSummary(_Schema):
    """A row of ``/fixtures``: enough to name a match in a prediction request."""

    match_id: str
    competition_id: str
    date: dt.date
    home_team: str
    away_team: str


class FixtureListResponse(_Schema):
    fixtures: list[FixtureSummary]
    count: int


class ComponentHealth(_Schema):
    """One dependency, and whether the service can use it."""

    name: str
    ready: bool
    detail: str | None = None


class HealthResponse(_Schema):
    """Liveness and readiness in one document.

    Two different questions with one answer, because they have one answer here:
    the process is up, and it either has a model and a fixture table or it does
    not. ``status`` is ``"ok"`` only when every component is ready, so a
    load balancer can read one field and a human can read the rest.
    """

    status: str = Field(examples=["ok", "degraded"])
    version: str
    components: list[ComponentHealth]


class VersionResponse(_Schema):
    """What is actually running, in enough detail to reproduce a number.

    The provenance a served probability needs and a log line cannot carry: the
    project version, the model, its members, how many matches it was fitted on
    and through what date, the temperature, and whether the libraries that
    unpickled it are the ones that pickled it.
    """

    version: str
    model: str | None = None
    members: list[str] = Field(default_factory=list)
    design_columns: int = len(DESIGN_COLUMNS)
    temperature: float | None = None
    trained_matches: int | None = None
    trained_from: dt.date | None = None
    trained_through: dt.date | None = None
    fixtures_indexed: int = 0
    prediction_log: str = Field(examples=["postgres", "disabled"])
    library_mismatches: list[str] = Field(
        default_factory=list,
        description=(
            "Libraries whose installed version differs from the one that fitted "
            "the artefact. Reported rather than enforced."
        ),
    )


class LimitationsResponse(_Schema):
    """The model card's own 'what it must not be used for', served.

    The plan asked for the limitations to be reachable from a response rather
    than buried in a repository. The text is
    :data:`src.evaluation.model_card.LIMITATIONS` — the same object the card is
    rendered from, so this endpoint cannot drift from the document.
    """

    model: str | None = None
    limitations: list[str]
    model_card_url: str = (
        "https://github.com/Vanshcloud/Match-Outcome-Predictor/blob/main/docs/MODEL_CARD.md"
    )


class ErrorResponse(_Schema):
    """The shape every non-2xx body has.

    One shape, so a client writes one branch. FastAPI's default is
    ``{"detail": ...}`` and that is kept rather than replaced — a bespoke
    envelope would differ from the 422 the framework raises before any handler
    runs, and a client would need both.
    """

    detail: str
