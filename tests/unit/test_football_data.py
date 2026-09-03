"""The football-data.co.uk adapter.

No test here touches the network. Parsing tests pre-populate the cache with a
settled season, which the freshness rule then serves without a download;
caching and error-path tests inject a recording stub in place of the HTTP
client.

Every fixture is a reduced copy of a real file shape — the 1993/94 seven-column
layout, the modern hundred-and-twenty-column one, the secondary feed's
country layout — because a fixture invented from the schema tests the schema,
not the provider.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.ingestion.base import CANONICAL_SCHEMA, Capability, SeasonUnavailableError
from src.ingestion.cache import FetchCache
from src.ingestion.football_data import (
    FootballDataProvider,
    derive_result,
    is_season_settled,
    parse_date,
    parse_float,
    parse_int,
    pick_odds,
)
from src.ingestion.registry import Competition, Feed, Registry
from src.utils.http import DownloadResult

TODAY = date(2026, 9, 3)

ENG = Competition(
    id="ENG_1", country="England", name="Premier League", tier=1, feed=Feed.MAIN, code="E0"
)
BRA = Competition(id="BRA_1", country="Brazil", name="Serie A", tier=1, feed=Feed.EXTRA, code="BRA")
ARG_CUP = Competition(
    id="ARG_CUP",
    country="Argentina",
    name="Copa",
    tier=None,
    feed=Feed.EXTRA,
    code="ARG",
    league_filter="Copa De La Liga Profesional",
)

REGISTRY = Registry(earliest_season=1993, competitions=(ENG, BRA, ARG_CUP))

# The modern primary-feed shape, reduced to the columns the adapter reads.
MODERN_MAIN = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,AvgCH,AvgCD,AvgCA\n"
    "E0,16/08/2024,20:00,Man United,Fulham,1,0,H,0,0,D,R Jones,"
    "14,10,5,2,12,10,7,8,2,3,0,0,1.66,4.02,5.25\n"
    "E0,17/08/2024,12:30,Ipswich,Liverpool,0,2,A,0,1,A,T Robinson,"
    "7,15,2,7,10,9,3,6,3,1,0,0,7.87,4.75,1.43\n"
)

# The 1993/94 shape: seven named columns, twenty-one unnamed, and blank rows.
ARCHAIC_MAIN = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,,,\n"
    "E0,14/08/93,Arsenal,Coventry,0,3,A,,,,\n"
    "E0,14/08/93,Chelsea,Blackburn,1,2,A,,,,\n"
    ",,,,,,,,,,\n"
    ",,,,,,,,,,\n"
)

EXTRA_FEED = (
    "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,AvgCH,AvgCD,AvgCA\n"
    "Brazil,Serie A,2024,14/04/2024,20:00,Palmeiras,Santos,2,1,H,1.80,3.50,4.20\n"
    "Brazil,Serie A,2024,15/04/2024,21:00,Flamengo,Corinthians,1,1,D,2.10,3.30,3.60\n"
    "Brazil,Serie A,2025,12/04/2025,20:00,Palmeiras,Flamengo,0,0,D,2.50,3.10,3.00\n"
    # An unplayed fixture: scheduled, no score. Every country file carries the
    # rest of the current season this way.
    "Brazil,Serie A,2025,30/11/2025,20:00,Santos,Corinthians,,,,,,\n"
)

MULTI_LEAGUE_FEED = (
    "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
    "Argentina,Liga Profesional,2024,01/03/2024,20:00,Boca Juniors,River Plate,1,0,H\n"
    "Argentina,Copa De La Liga Profesional,2024,05/03/2024,20:00,Racing Club,Independiente,2,2,D\n"
    # Whitespace variant of the same competition. The provider really does this.
    "Argentina , Copa De La Liga Profesional ,2024,09/03/2024,20:00,Velez,Huracan,0,1,A\n"
)


class RecordingClient:
    """Stands in for HttpClient, recording downloads instead of making them."""

    def __init__(
        self,
        payloads: dict[str, str] | None = None,
        error: Exception | None = None,
        *,
        not_modified: bool = True,
        etag: str | None = '"v1"',
        last_modified: str | None = "Thu, 28 Jan 2021 22:47:08 GMT",
    ):
        self.payloads = payloads or {}
        self.error = error
        self.not_modified = not_modified
        self.etag = etag
        self.last_modified = last_modified
        self.requested: list[str] = []
        self.conditional: list[tuple[str | None, str | None]] = []

    def download(
        self,
        url: str,
        destination: Path,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        **_kwargs: object,
    ) -> DownloadResult:
        self.requested.append(url)
        self.conditional.append((etag, last_modified))
        if self.error is not None:
            raise self.error
        # Mimic a real 304: when the caller sends a validator matching what
        # this stub is serving, answer "not modified" and touch nothing.
        if self.not_modified and (etag or last_modified):
            return DownloadResult(
                path=destination, modified=False, etag=etag, last_modified=last_modified
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = next((v for k, v in self.payloads.items() if url.endswith(k)), None)
        if body is None:
            raise _http_error(404)
        destination.write_text(body, encoding="utf-8")
        return DownloadResult(
            path=destination,
            modified=True,
            etag=self.etag,
            last_modified=self.last_modified,
            bytes_written=len(body.encode()),
        )

    def close(self) -> None:
        return None


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status}", response=response)


def make_provider(
    tmp_path: Path,
    client: RecordingClient | None = None,
    *,
    cache: FetchCache | None = None,
    **kwargs: object,
) -> FootballDataProvider:
    return FootballDataProvider(
        REGISTRY,
        tmp_path / "raw",
        client=client or RecordingClient(),
        today=TODAY,
        cache=cache if cache is not None else FetchCache.load(tmp_path / "fetch_cache.json"),
        **kwargs,  # type: ignore[arg-type]
    )


SEEDED_ETAG = '"seeded"'
SEEDED_LAST_MODIFIED = "Thu, 28 Jan 2021 22:47:08 GMT"


def seed(tmp_path: Path, competition: Competition, season: str, body: str) -> None:
    """Write a file into the cache as though a previous run had fetched it.

    Both halves matter. The file goes on disk, *and* a fetch-cache entry
    records the validators that came with it — that is what a real prior run
    leaves behind, and it is what lets the next run ask "changed?" instead of
    downloading again. Seeding only the file would model a machine that
    downloaded data and forgot it had.
    """
    provider = make_provider(tmp_path)
    path = provider.local_path(competition, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")

    cache = FetchCache.load(tmp_path / "fetch_cache.json")
    cache.record_fetch(
        provider.url_for(competition, season),
        etag=SEEDED_ETAG,
        last_modified=SEEDED_LAST_MODIFIED,
        sha256=None,
        size=path.stat().st_size,
    )
    cache.save()


# ---- field parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("16/08/2024", "2024-08-16"), ("14/08/93", "1993-08-14"), ("01/01/05", "2005-01-01")],
)
def test_both_date_formats_parse(raw: str, expected: str) -> None:
    """dd/mm/yy until 2014/15, dd/mm/yyyy after. Both appear in one ingest."""
    parsed = parse_date(raw)
    assert parsed is not None
    assert parsed.strftime("%Y-%m-%d") == expected


def test_a_blank_date_is_missing_not_an_error() -> None:
    """It marks an unplayed fixture, which the caller drops as a row."""
    assert parse_date("") is None


def test_an_unparseable_date_raises() -> None:
    from src.ingestion.csv_reader import ProviderFileError

    with pytest.raises(ProviderFileError, match="date format"):
        parse_date("2024-08-16")


@pytest.mark.parametrize(
    ("raw", "expected"), [("3", 3), ("", None), ("2.0", 2), ("x", None), ("  5 ", 5)]
)
def test_integer_parsing(raw: str, expected: int | None) -> None:
    assert parse_int(raw) == expected


def test_float_parsing() -> None:
    assert parse_float("1.66") == 1.66
    assert parse_float("") is None
    assert parse_float("n/a") is None


@pytest.mark.parametrize(("home", "away", "expected"), [(2, 1, "H"), (0, 3, "A"), (1, 1, "D")])
def test_result_derivation(home: int, away: int, expected: str) -> None:
    assert derive_result(home, away) == expected


# ---- odds -------------------------------------------------------------------


def test_market_average_closing_odds_win() -> None:
    """Averages outrank a single book, and closing outranks opening."""
    row = {
        "AvgCH": "1.6",
        "AvgCD": "4.2",
        "AvgCA": "5.2",
        "AvgH": "1.5",
        "AvgD": "4.0",
        "AvgA": "5.0",
        "B365H": "9",
    }
    assert pick_odds(row) == (1.6, 4.2, 5.2)


def test_odds_fall_back_through_the_eras() -> None:
    """No single triple spans the history: nothing before 2003/04, Bet365 only
    until Betbrain averages appear, market averages only from about 2019."""
    assert pick_odds({"BbAvH": "2.0", "BbAvD": "3.3", "BbAvA": "3.9"}) == (2.0, 3.3, 3.9)
    assert pick_odds({"B365H": "2.1", "B365D": "3.2", "B365A": "3.4"}) == (2.1, 3.2, 3.4)


def test_a_partial_triple_is_refused_entirely() -> None:
    """Mixing a home price from one book with a draw price from another
    produces an overround no real market had, quietly corrupting the very
    benchmark this project measures itself against."""
    assert pick_odds({"AvgCH": "1.6", "AvgCD": "4.2", "B365A": "5.2"}) == (None, None, None)


def test_no_odds_at_all_is_not_an_error() -> None:
    """Nothing before 2003/04 carries any."""
    assert pick_odds({}) == (None, None, None)


# ---- the canonical contract -------------------------------------------------


def test_every_feed_produces_the_identical_schema(tmp_path: Path) -> None:
    """The single most important property in this package. Downstream code is
    written once against this schema and must never learn which feed a match
    came from."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    seed(tmp_path, ENG, "1993-94", ARCHAIC_MAIN)
    seed(tmp_path, BRA, "", EXTRA_FEED)
    provider = make_provider(tmp_path)

    frames = [
        provider.fetch(ENG, "2024-25"),
        provider.fetch(ENG, "1993-94"),
        provider.fetch(BRA, "2024"),
    ]
    for frame in frames:
        assert list(frame.columns) == list(CANONICAL_SCHEMA)
        assert {c: str(d) for c, d in frame.dtypes.items()} == CANONICAL_SCHEMA


def test_a_modern_season_populates_everything(tmp_path: Path) -> None:
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")

    assert len(frame) == 2
    first = frame.iloc[0]
    assert first["home_team"] == "Man United"
    assert first["home_team_id"] == "eng:man-united"
    assert first["home_goals"] == 1
    assert first["result"] == "H"
    assert first["ht_result"] == "D"
    assert first["home_shots"] == 14
    assert first["referee"] == "R Jones"
    assert first["odds_home"] == 1.66
    assert first["kickoff"] == "20:00"
    assert first["competition_id"] == "ENG_1"
    assert first["season"] == "2024-25"


def test_an_archaic_season_leaves_optional_columns_null(tmp_path: Path) -> None:
    """1993/94 has seven columns. Absent is not zero: "this feed reports no
    shots" and "no shots were taken" are different facts, and conflating them
    would teach a model that football in 1993 had no shooting."""
    seed(tmp_path, ENG, "1993-94", ARCHAIC_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "1993-94")

    assert len(frame) == 2, "blank rows should have been dropped"
    assert frame["home_shots"].isna().all()
    assert frame["referee"].isna().all()
    assert frame["odds_home"].isna().all()
    assert frame["home_goals"].notna().all()


def test_integer_columns_stay_integral(tmp_path: Path) -> None:
    """NumPy's int64 has no null, so one missing value silently promotes the
    column to float and 13 shots becomes 13.0. The nullable Int16 dtype is what
    keeps counts countable."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert str(frame["home_shots"].dtype) == "Int16"
    assert frame["home_shots"].iloc[0] == 14


def test_match_ids_are_unique_and_deterministic(tmp_path: Path) -> None:
    """Determinism is what makes re-ingesting idempotent: a storage upsert
    replaces rows rather than duplicating them."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    first = make_provider(tmp_path).fetch(ENG, "2024-25")
    second = make_provider(tmp_path).fetch(ENG, "2024-25")

    assert first["match_id"].nunique() == len(first)
    assert list(first["match_id"]) == list(second["match_id"])


# ---- the secondary feed -----------------------------------------------------


def test_a_country_file_is_split_by_season(tmp_path: Path) -> None:
    seed(tmp_path, BRA, "", EXTRA_FEED)
    provider = make_provider(tmp_path)
    assert len(provider.fetch(BRA, "2024")) == 2
    assert len(provider.fetch(BRA, "2025")) == 1


def test_calendar_seasons_are_reported_as_candidates(tmp_path: Path) -> None:
    """Brazil and the USA run February to November inside one calendar year,
    so their labels are years rather than year pairs."""
    seed(tmp_path, BRA, "", EXTRA_FEED)
    assert make_provider(tmp_path).candidate_seasons(BRA) == ("2024", "2025")


def test_unplayed_fixtures_are_dropped(tmp_path: Path) -> None:
    """Failing on them would make the current season unfetchable; keeping them
    would put a match with no result into the training target."""
    seed(tmp_path, BRA, "", EXTRA_FEED)
    frame = make_provider(tmp_path).fetch(BRA, "2025")
    assert len(frame) == 1
    assert frame["result"].notna().all()


def test_a_league_filter_selects_one_competition(tmp_path: Path) -> None:
    """Argentina's file mixes the league with a cup. Unfiltered they merge into
    one impossible competition."""
    seed(tmp_path, ARG_CUP, "", MULTI_LEAGUE_FEED)
    frame = make_provider(tmp_path).fetch(ARG_CUP, "2024")
    assert len(frame) == 2
    assert set(frame["home_team"]) == {"Racing Club", "Velez"}


def test_the_league_filter_tolerates_the_providers_own_whitespace(tmp_path: Path) -> None:
    """' Copa De La Liga Profesional ' and the unpadded spelling are the same
    competition. An exact match drops whichever the filter did not use."""
    seed(tmp_path, ARG_CUP, "", MULTI_LEAGUE_FEED)
    assert "Velez" in set(make_provider(tmp_path).fetch(ARG_CUP, "2024")["home_team"])


# ---- validation on the way in -----------------------------------------------


def test_a_result_contradicting_the_score_is_dropped(tmp_path: Path) -> None:
    """Measured across all 303,517 ingested matches, the stated result never
    disagreed with
    the score. That is precisely why a disagreement now means something
    structural is wrong rather than one typo."""
    body = MODERN_MAIN.replace(",1,0,H,", ",1,0,A,")
    seed(tmp_path, ENG, "2024-25", body)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert len(frame) == 1
    assert "Man United" not in set(frame["home_team"])


def test_a_team_playing_itself_is_dropped(tmp_path: Path) -> None:
    """Impossible, and a signal that columns have shifted."""
    body = MODERN_MAIN.replace("Man United,Fulham", "Fulham,Fulham")
    seed(tmp_path, ENG, "2024-25", body)
    assert len(make_provider(tmp_path).fetch(ENG, "2024-25")) == 1


def test_a_season_with_no_playable_rows_is_unavailable(tmp_path: Path) -> None:
    seed(tmp_path, ENG, "2024-25", "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n")
    with pytest.raises(SeasonUnavailableError):
        make_provider(tmp_path).fetch(ENG, "2024-25")


# ---- capabilities -----------------------------------------------------------


def test_observed_capabilities_reflect_the_season_not_the_feed(tmp_path: Path) -> None:
    """The registry declares what a feed *can* supply; this reports what a
    season really did. Conflating them is how a column of nulls is mistaken for
    a competition that simply does not report the statistic."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    seed(tmp_path, ENG, "1993-94", ARCHAIC_MAIN)
    provider = make_provider(tmp_path)

    modern = provider.observed_capabilities(provider.fetch(ENG, "2024-25"))
    archaic = provider.observed_capabilities(provider.fetch(ENG, "1993-94"))

    assert Capability.MATCH_STATS in modern
    assert Capability.MATCH_STATS not in archaic
    assert Capability.RESULTS in archaic
    # Both are the same competition, whose feed declares the full set.
    assert ENG.supports(Capability.MATCH_STATS)


# ---- URLs, caching and missing files ----------------------------------------


def test_urls_match_the_two_layouts(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    assert provider.url_for(ENG, "2024-25").endswith("/mmz4281/2425/E0.csv")
    assert provider.url_for(ENG, "1993-94").endswith("/mmz4281/9394/E0.csv")
    assert provider.url_for(BRA, "2024").endswith("/new/BRA.csv")


def test_a_settled_season_is_never_refetched(tmp_path: Path) -> None:
    """Without this, refreshing the current season re-downloads three decades
    of history that cannot have changed."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    client = RecordingClient()
    make_provider(tmp_path, client).fetch(ENG, "2024-25")
    assert client.requested == []


def test_an_unsettled_season_is_always_revalidated(tmp_path: Path) -> None:
    """The current season gains rows every match day, so age is the wrong
    question — it is asked of the server every run. A conditional request costs
    one round trip and no body when nothing has changed."""
    seed(tmp_path, ENG, "2026-27", MODERN_MAIN)
    client = RecordingClient({"2627/E0.csv": MODERN_MAIN})
    make_provider(tmp_path, client).fetch(ENG, "2026-27")
    assert len(client.requested) == 1


@pytest.mark.parametrize(
    ("season", "settled"),
    [("1993-94", True), ("2023-24", True), ("2024-25", True), ("2025-26", False)],
)
def test_season_settlement(season: str, settled: bool) -> None:
    assert is_season_settled(season, TODAY) is settled


def test_a_404_becomes_season_unavailable(tmp_path: Path) -> None:
    provider = make_provider(tmp_path, RecordingClient(error=_http_error(404)))
    with pytest.raises(SeasonUnavailableError, match="not published"):
        provider.fetch(ENG, "2024-25")


def test_html_served_as_csv_becomes_season_unavailable(tmp_path: Path) -> None:
    """The provider answers a missing division-season with HTTP 300 and a
    "Multiple Choices" page. raise_for_status does not treat 3xx as an error,
    so the download succeeds and only a content check catches it."""
    client = RecordingClient({"2425/E0.csv": "<!DOCTYPE HTML><html>300 Multiple Choices</html>"})
    provider = make_provider(tmp_path, client)
    with pytest.raises(SeasonUnavailableError, match="HTML page"):
        provider.fetch(ENG, "2024-25")


def test_an_html_page_is_not_left_in_the_cache(tmp_path: Path) -> None:
    """Otherwise the next run finds a fresh-looking cached file and parses it."""
    client = RecordingClient({"2425/E0.csv": "<!DOCTYPE HTML><html>oops</html>"})
    provider = make_provider(tmp_path, client)
    with pytest.raises(SeasonUnavailableError):
        provider.fetch(ENG, "2024-25")
    assert not provider.local_path(ENG, "2024-25").exists()


def test_a_server_error_is_not_swallowed(tmp_path: Path) -> None:
    """A 500 means the provider is broken, not that the season does not exist.
    Reporting it as "unavailable" would silently produce a model trained on
    however much data happened to download that day."""
    provider = make_provider(tmp_path, RecordingClient(error=_http_error(500)))
    with pytest.raises(requests.HTTPError):
        provider.fetch(ENG, "2024-25")


def test_candidate_seasons_span_the_registry_range(tmp_path: Path) -> None:
    seasons = make_provider(tmp_path).candidate_seasons(ENG)
    assert seasons[0] == "1993-94"
    assert seasons[-1] == "2026-27"
    assert len(seasons) == len(set(seasons))


def test_an_injected_client_is_not_closed_by_the_provider(tmp_path: Path) -> None:
    """The caller that opened it owns it. Closing a shared client here would
    break the next competition in the same run."""
    client = RecordingClient()
    provider = make_provider(tmp_path, client)
    provider.close()
    assert provider._client is client


def test_the_provider_satisfies_the_protocol(tmp_path: Path) -> None:
    from src.ingestion.base import MatchProvider

    assert isinstance(make_provider(tmp_path), MatchProvider)


def test_dates_are_real_timestamps(tmp_path: Path) -> None:
    """Sorting and temporal splits both depend on this being a datetime rather
    than a string that happens to sort correctly."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert frame["date"].dtype == "datetime64[ns]"
    assert frame["date"].iloc[0] == pd.Timestamp("2024-08-16")


# ---- remaining drop paths and lifecycle -------------------------------------


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        pytest.param(("16/08/2024", ""), "no date", id="missing-date"),
        pytest.param(("Man United,Fulham", ",Fulham"), "no home team", id="missing-home-team"),
        pytest.param(("Man United,Fulham", "Man United,"), "no away team", id="missing-away-team"),
    ],
)
def test_rows_missing_a_date_or_a_team_are_dropped(
    tmp_path: Path, mutation: tuple[str, str], reason: str
) -> None:
    """A postponed fixture appears with its teams and no date; an abandoned one
    can lose a team name. Neither is a match, and neither should abort a
    season."""
    body = MODERN_MAIN.replace(*mutation, 1)
    seed(tmp_path, ENG, "2024-25", body)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert len(frame) == 1, reason


def test_a_season_whose_every_row_is_unplayable_is_unavailable(tmp_path: Path) -> None:
    """Distinct from an empty file: the rows parsed fine and none was a played
    match. A country file in February looks exactly like this."""
    body = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Brazil,Serie A,2027,01/03/2027,20:00,Palmeiras,Santos,,,\n"
    )
    seed(tmp_path, BRA, "", body)
    with pytest.raises(SeasonUnavailableError, match="no playable matches"):
        make_provider(tmp_path).fetch(BRA, "2027")


def test_provider_identity_and_competition_listing(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    assert provider.name == "football-data"
    assert provider.competitions() == REGISTRY.competitions


def test_a_client_is_created_on_demand_and_closed(tmp_path: Path) -> None:
    """A provider constructed without one owns what it makes, and must close
    it — the opposite of the injected case."""
    provider = FootballDataProvider(REGISTRY, tmp_path / "raw", today=TODAY)
    assert provider._client is None
    client = provider.client
    assert provider.client is client, "the client should be created once, not per access"
    provider.close()
    assert provider._client is None


# ---- implausible values are nulled, not dropped -----------------------------
#
# Both cases below are real upstream typos found by running over all 305,501
# ingested matches. Both are vanishingly rare, and in both the *match* is fine
# — only one field is impossible. Nulling the field keeps a valid training row
# and removes a number that would silently poison any feature built from it.


def test_an_impossible_book_is_nulled_and_the_match_kept(tmp_path: Path) -> None:
    """A real bookmaker's implied probabilities always sum above 1.0; the
    excess is the margin. 2.39/5.72/10.84 sums to 0.685, which is free money
    and therefore a typo. Measured: 33 such rows in 246,052 priced matches."""
    body = MODERN_MAIN.replace(",1.66,4.02,5.25", ",2.39,5.72,10.84")
    seed(tmp_path, ENG, "2024-25", body)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")

    assert len(frame) == 2, "the match itself is valid and must survive"
    assert pd.isna(frame.iloc[0]["odds_home"])
    assert pd.isna(frame.iloc[0]["odds_draw"])
    assert pd.isna(frame.iloc[0]["odds_away"])
    assert frame.iloc[0]["result"] == "H"
    assert frame.iloc[1]["odds_home"] == 7.87, "the other row is untouched"


def test_a_normal_book_survives(tmp_path: Path) -> None:
    """The median real overround is 1.074. Nothing near that may be discarded."""
    seed(tmp_path, ENG, "2024-25", MODERN_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert frame["odds_home"].notna().all()


def test_shots_on_target_exceeding_shots_nulls_both(tmp_path: Path) -> None:
    """A shot on target is a shot. Nine such rows exist in 128,498, seven of
    them in 2000/01 — the first season shot data was published at all. There is
    no way to tell which of the pair is wrong, so both go."""
    body = MODERN_MAIN.replace(",14,10,5,2,", ",4,10,9,2,")
    seed(tmp_path, ENG, "2024-25", body)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")

    assert len(frame) == 2
    assert pd.isna(frame.iloc[0]["home_shots"])
    assert pd.isna(frame.iloc[0]["home_shots_on_target"])
    assert frame.iloc[0]["away_shots"] == 10, "the away side was consistent"
    assert frame.iloc[0]["home_corners"] == 7, "unrelated stats are untouched"


def test_equal_shots_and_shots_on_target_is_allowed(tmp_path: Path) -> None:
    """Every shot being on target is unusual, not impossible."""
    body = MODERN_MAIN.replace(",14,10,5,2,", ",5,10,5,2,")
    seed(tmp_path, ENG, "2024-25", body)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert frame.iloc[0]["home_shots"] == 5
    assert frame.iloc[0]["home_shots_on_target"] == 5


# ---- the file's own Div is authoritative ------------------------------------

# `1993-94/SP2.csv` as the provider actually serves it: a copy of SP1, with the
# Div column inside still saying so. Four of its siblings are the same file
# under other names, including Portugal's and Scotland's first seasons.
MISLABELLED_MAIN = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,,,\n"
    "SP1,05/09/93,Ath Bilbao,Albacete,4,1,H,,,,\n"
    "SP1,05/09/93,Barcelona,Osasuna,4,0,H,,,,\n"
)

MIXED_MAIN = MODERN_MAIN + (
    "SP1,18/08/2024,15:00,Barcelona,Osasuna,4,0,H,2,0,H,J Ruiz,"
    "20,4,9,1,8,12,9,2,1,2,0,0,1.20,7.00,12.0\n"
)


def test_rows_naming_another_division_are_dropped(tmp_path: Path) -> None:
    """The URL says which division a file is; the file says it too, in every
    row. When they disagree the file is right — it is the data, and the URL is
    only where it was published."""
    seed(tmp_path, ENG, "2024-25", MIXED_MAIN)
    frame = make_provider(tmp_path).fetch(ENG, "2024-25")
    assert len(frame) == 2
    assert "Barcelona" not in set(frame["home_team"])


def test_a_whole_file_of_another_division_yields_nothing(tmp_path: Path) -> None:
    """The real case: five cached files are copies of SP1.csv served under
    another name. Trusting the URL put 380 Spanish matches into Portugal's
    first season wearing `por:` team ids, every one internally consistent, so
    no per-row check could see it."""
    seed(tmp_path, ENG, "1993-94", MISLABELLED_MAIN)
    with pytest.raises(SeasonUnavailableError):
        make_provider(tmp_path).fetch(ENG, "1993-94")


def test_a_blank_div_is_kept(tmp_path: Path) -> None:
    """Some early files leave the column empty on the odd row. Dropping those
    would trade a rare provider error for a common one."""
    body = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,,,\n"
        "E0,14/08/93,Arsenal,Coventry,0,3,A,,,,\n"
        ",14/08/93,Chelsea,Blackburn,1,2,A,,,,\n"
    )
    seed(tmp_path, ENG, "1993-94", body)
    assert len(make_provider(tmp_path).fetch(ENG, "1993-94")) == 2


def test_the_drop_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Silently discarding rows is how a provider error becomes a mystery."""
    seed(tmp_path, ENG, "2024-25", MIXED_MAIN)
    with caplog.at_level("WARNING"):
        make_provider(tmp_path).fetch(ENG, "2024-25")
    assert "dropped 1 row(s) labelled ['SP1']" in caplog.text


def test_the_secondary_feed_is_unaffected(tmp_path: Path) -> None:
    """Country files carry no Div column; they are filtered by League instead."""
    seed(tmp_path, BRA, "", EXTRA_FEED)
    assert not make_provider(tmp_path).fetch(BRA, "2024").empty
