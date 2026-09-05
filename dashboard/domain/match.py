"""What a match and a forecast *are* to this application.

Every card, every list and every detail page renders a :class:`Fixture`.
Nothing renders a dictionary from an HTTP response or a row of a DataFrame, and
that is the point: this project has two sources of matches today and will have
four by Milestone 17, and a page written against one of them is a page that has
to be rewritten when the second arrives.

Pure value types. No provider, no network, no file, no Streamlit — everything
that fetches lives in :mod:`dashboard.providers` and everything that renders in
:mod:`dashboard.views`, and both import this rather than the other way round.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class MatchStatus(StrEnum):
    """Where a match is in its life.

    Three states rather than a free-text label, because the page branches on
    it: a scheduled match shows a kick-off time, a live one shows a minute, and
    a finished one shows a score.
    """

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"

    UNKNOWN = "unknown"
    """A match the source can name but not place in time.

    This is what ``/fixtures`` returns and it is not a defect to be tidied
    away: the service answers "which matches can I price", the answer is drawn
    from a table of played matches, and it withholds the scoreline on purpose.
    Rendering that as ``FINISHED`` would put a blank score on a card and invite
    a reader to conclude the match ended 0-0.
    """


@dataclass(frozen=True, slots=True)
class Fixture:
    """One match, as a card renders it.

    Optional everywhere it can be missing, because a source that has the
    kick-off minute is a different source from the one that has the final
    score, and a page that required both would be a page neither can feed.

    ``crest_url`` is unset by every source in this repository and is read by
    :func:`dashboard.ui.crest_html`, which draws initials instead. It is the
    single field a badge provider has to start filling.
    """

    match_id: str
    competition_id: str
    date: dt.date
    home_team: str
    away_team: str
    status: MatchStatus = MatchStatus.UNKNOWN
    competition: str | None = None
    country: str | None = None
    kickoff: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    minute: int | None = None
    home_crest_url: str | None = None
    away_crest_url: str | None = None

    @property
    def has_score(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def teams(self) -> tuple[str, str]:
        return self.home_team, self.away_team

    def involves(self, team: str) -> bool:
        """Whether ``team`` played, matched the way a person types a club name."""
        wanted = " ".join(team.split()).casefold()
        return any(" ".join(name.split()).casefold() == wanted for name in self.teams)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A fixture priced, as the service answered.

    A named type rather than the response dictionary, because four views read
    it and the alternative is four places that know the wire format. Built by
    :func:`dashboard.providers.api.as_prediction`, which is the one place that
    does.

    ``in_sample`` is carried rather than dropped for the reason the API returns
    it: the served model is fitted on the whole history, so a fixture inside
    that history was trained on, and quoting its probability as the
    walk-forward number is the single most likely way a figure from this
    project ends up overstated.
    """

    match_id: str
    probabilities: Mapping[str, float]
    model: str
    model_version: str
    in_sample: bool

    @property
    def outcome(self) -> str:
        """The most likely of the three, by the name the model returns it under."""
        return max(self.probabilities, key=lambda key: self.probabilities[key])

    @property
    def stated(self) -> float:
        """How firmly the model committed: the largest of the three probabilities.

        **Not a calibrated confidence.** What this probability is *worth* is a
        measurement, and it comes from the reliability tables the model card is
        generated from — see :mod:`dashboard.services.reports`.
        """
        return float(self.probabilities[self.outcome])
