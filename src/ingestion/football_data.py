"""The football-data.co.uk adapter.

One class covering both of the provider's file layouts, because they are one
provider with one set of quirks and splitting them into two adapters would
duplicate the caching, decoding and normalisation for no gain. The layouts
differ in shape, not in kind:

===========  ==========================================  =========================
             primary (``main``)                          secondary (``extra``)
===========  ==========================================  =========================
URL          ``/mmz4281/{season_code}/{div}.csv``        ``/new/{country}.csv``
Granularity  one competition-season per file             one country, all seasons
Detail       shots, corners, fouls, cards, referee       results and odds only
Season       encoded in the URL, always split            a column, split or calendar
History      1993/94 onward                              roughly 2012 onward
===========  ==========================================  =========================

Everything either layout produces lands in
:data:`~src.ingestion.base.CANONICAL_SCHEMA`, so no downstream module ever asks
which feed a match came from.

Caching is age-based with one exception that matters: a season that has
finished is immutable, so its file is never re-fetched however old the copy is.
Without that rule a full refresh would re-download three decades of settled
history to pick up the current season's weekend results.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.ingestion.base import (
    CANONICAL_SCHEMA,
    Capability,
    Result,
    SeasonUnavailableError,
    make_match_id,
)
from src.ingestion.csv_reader import ProviderFileError, looks_like_csv, read_provider_csv
from src.ingestion.registry import (
    Competition,
    Feed,
    Registry,
    normalise_season,
    season_label_to_code,
    season_start_year,
)
from src.ingestion.teams import team_id
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROVIDER_NAME = "football-data"
BASE_URL = "https://www.football-data.co.uk"

# --- Column maps -------------------------------------------------------------
#
# Provider column -> canonical column. Absent keys become nulls, which is how a
# 1993 file with seven columns and a 2024 file with a hundred and twenty go
# through the same code path.

MAIN_COLUMNS: dict[str, str] = {
    "Date": "date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "Referee": "referee",
}

EXTRA_COLUMNS: dict[str, str] = {
    "Date": "date",
    "Time": "kickoff",
    "Home": "home_team",
    "Away": "away_team",
    "HG": "home_goals",
    "AG": "away_goals",
    "Res": "result",
}

# Closing decimal odds, best available triple, most preferred first.
#
# The provider's odds columns changed twice in thirty years and no single
# triple spans the history: nothing before 2003/04, Bet365 only until Betbrain
# averages appear around 2009, and the current market-average columns
# (``AvgC*``) only from about 2019. Market *averages* outrank any single
# bookmaker because they are the better-calibrated benchmark, and *closing*
# prices outrank opening ones because closing is the market's final word.
ODDS_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("AvgCH", "AvgCD", "AvgCA"),  # market average, closing — current era
    ("AvgH", "AvgD", "AvgA"),  # market average, opening
    ("BbAvH", "BbAvD", "BbAvA"),  # Betbrain average, ~2009-2019
    ("B365CH", "B365CD", "B365CA"),  # Bet365 closing
    ("B365H", "B365D", "B365A"),  # Bet365, the longest single-book history
)

INTEGER_COLUMNS: frozenset[str] = frozenset(
    name for name, dtype in CANONICAL_SCHEMA.items() if dtype.startswith("Int")
)

# A season file stops changing once the season is over. This is how many years
# after a season's start year it is treated as settled and never re-fetched.
# Two, not one: a split season starting in 2023 finishes in mid-2024, so only
# from 2025 is it certainly complete regardless of today's month.
SEASON_SETTLED_AFTER_YEARS = 2

# A real bookmaker's implied probabilities always sum above 1.0 — the excess is
# the margin. A triple summing below it is free money, so it is a provider
# typo rather than a price. Measured over 246,052 priced matches the median
# overround is 1.074 and only 33 rows (0.013%) fall below 1.0.
#
# 0.99 rather than 1.00 to leave room for rounding in the published prices.
MIN_PLAUSIBLE_OVERROUND = 0.99


# --- Field parsing -----------------------------------------------------------


def parse_date(value: str) -> pd.Timestamp | None:
    """Parse the provider's two date formats.

    ``dd/mm/yy`` until 2014/15 and ``dd/mm/yyyy`` after it, with both appearing
    across a full ingest. The two-digit form is unambiguous here because
    ``%y`` pivots at 69, and this provider's history runs 1993-2068.

    Returns ``None`` rather than raising: a blank date belongs to an unplayed
    fixture, which the caller drops as a row rather than as an error.
    """
    text = value.strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise ProviderFileError(f"unrecognised date format: {value!r}")


def parse_int(value: str) -> int | None:
    """Parse an integer field, treating blank and non-numeric as missing.

    The provider leaves a field empty when a statistic was not recorded, and
    occasionally writes a stray non-numeric marker. Both mean "no value" and
    neither is worth failing an entire season for.
    """
    text = value.strip()
    if not text:
        return None
    try:
        # float() first: some integer-valued fields are written as "2.0".
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pick_odds(row: dict[str, str]) -> tuple[float | None, float | None, float | None]:
    """Return the best available closing odds triple for one row.

    All three prices must come from the *same* triple. Mixing a home price from
    one bookmaker with a draw price from another produces an overround that is
    not a real market's, which would quietly corrupt the benchmark this
    project measures itself against.
    """
    for home_col, draw_col, away_col in ODDS_CANDIDATES:
        home = parse_float(row.get(home_col, ""))
        draw = parse_float(row.get(draw_col, ""))
        away = parse_float(row.get(away_col, ""))
        if home is not None and draw is not None and away is not None:
            return home, draw, away
    return None, None, None


def derive_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return Result.HOME.value
    if home_goals < away_goals:
        return Result.AWAY.value
    return Result.DRAW.value


def is_season_settled(season: str, today: date | None = None) -> bool:
    """Whether ``season`` is finished and its file can be cached indefinitely."""
    current = (today or datetime.now(tz=UTC).date()).year
    return season_start_year(season) + SEASON_SETTLED_AFTER_YEARS <= current


class FootballDataProvider:
    """Reads football-data.co.uk into the canonical match schema.

    Satisfies :class:`~src.ingestion.base.MatchProvider` structurally; it does
    not inherit from it, so the protocol stays a description rather than a base
    class that adapters must import.
    """

    def __init__(
        self,
        registry: Registry,
        raw_dir: Path,
        *,
        client: HttpClient | None = None,
        base_url: str = BASE_URL,
        max_age_days: float = 1.0,
        today: date | None = None,
    ) -> None:
        """
        Args:
            registry: Which competitions to expose.
            raw_dir: Where downloaded files are cached, untouched and unedited.
            client: Injected so tests never reach the network. A client is
                created on demand when not supplied.
            base_url: Overridden in tests to point at a local fixture server.
            max_age_days: How stale an unsettled file may be before refetching.
            today: Injected so cache tests are not time-dependent.
        """
        self.registry = registry
        self.raw_dir = raw_dir
        self.base_url = base_url.rstrip("/")
        self.max_age_days = max_age_days
        self._today = today
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def client(self) -> HttpClient:
        if self._client is None:
            self._client = HttpClient()
        return self._client

    def close(self) -> None:
        """Close the HTTP client, but only one this provider created itself."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # -- discovery ------------------------------------------------------------

    def competitions(self) -> tuple[Competition, ...]:
        return self.registry.competitions

    def candidate_seasons(self, competition: Competition) -> tuple[str, ...]:
        """Season labels worth attempting, oldest first.

        For the secondary feed this is exact — the file lists its own seasons.
        For the primary feed it is a range, because the provider publishes no
        index and establishing the truth would cost one request per
        division-season. :meth:`fetch` raises
        :class:`~src.ingestion.base.SeasonUnavailableError` for the misses.
        """
        if competition.feed is Feed.EXTRA:
            rows = self._read_country_file(competition)
            return tuple(
                sorted({normalise_season(row["Season"]) for row in rows if row.get("Season")})
            )

        today = self._today or datetime.now(tz=UTC).date()
        # A season starting in year Y is published from roughly August of Y, so
        # the newest worth attempting is Y itself once August has passed.
        newest = today.year if today.month >= 8 else today.year - 1
        return tuple(
            f"{year}-{(year + 1) % 100:02d}"
            for year in range(self.registry.earliest_season, newest + 1)
        )

    # -- fetching -------------------------------------------------------------

    def url_for(self, competition: Competition, season: str) -> str:
        if competition.feed is Feed.MAIN:
            return f"{self.base_url}/mmz4281/{season_label_to_code(season)}/{competition.code}.csv"
        return f"{self.base_url}/new/{competition.code}.csv"

    def local_path(self, competition: Competition, season: str) -> Path:
        if competition.feed is Feed.MAIN:
            return self.raw_dir / PROVIDER_NAME / "main" / season / f"{competition.code}.csv"
        return self.raw_dir / PROVIDER_NAME / "extra" / f"{competition.code}.csv"

    def _is_fresh(self, path: Path, competition: Competition, season: str) -> bool:
        """Whether the cached file can be used without re-downloading."""
        if not path.is_file() or path.stat().st_size == 0:
            return False
        # A finished season's file is immutable. Without this, refreshing the
        # current season would re-download three decades of settled history.
        if competition.feed is Feed.MAIN and is_season_settled(season, self._today):
            return True
        age_days = (datetime.now(tz=UTC).timestamp() - path.stat().st_mtime) / 86_400
        return age_days <= self.max_age_days

    def ensure_file(self, competition: Competition, season: str = "") -> Path:
        """Return a local path to the provider file, downloading if needed.

        ``season`` is meaningful only for the primary feed, where it selects
        the file. A secondary-feed country file holds every season at once, so
        callers there omit it.

        Raises:
            SeasonUnavailableError: If the provider does not hold this file. Both of
                its "missing" responses are covered — a 404, and the HTTP 300
                with an HTML body that ``raise_for_status`` does not treat as
                an error.
        """
        path = self.local_path(competition, season)
        if self._is_fresh(path, competition, season):
            logger.debug("cache hit: %s", path.name)
            return path

        url = self.url_for(competition, season)
        try:
            self.client.download(url, path)
        except Exception as error:  # noqa: BLE001 - re-raised below, narrowed
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 404:
                raise SeasonUnavailableError(f"{competition.id} {season}: not published") from error
            raise

        # The HTML-as-CSV guard. A missing division-season answers HTTP 300
        # with a "Multiple Choices" page, which downloads perfectly happily.
        # Checked here rather than in the parser so the bad file is removed
        # before it can be mistaken for a cached copy on the next run.
        if not looks_like_csv(path.read_bytes()):
            path.unlink(missing_ok=True)
            raise SeasonUnavailableError(
                f"{competition.id} {season}: provider returned an HTML page, not CSV"
            )
        return path

    def _read_country_file(self, competition: Competition) -> list[dict[str, str]]:
        """Read a secondary-feed country file, filtered to one competition."""
        rows = read_provider_csv(self.ensure_file(competition))
        if competition.league_filter is None:
            return rows
        # Compared stripped: the provider's own League column holds
        # ' J1 League' alongside 'J1 League', and an exact match would drop
        # whichever spelling the filter did not happen to use.
        wanted = competition.league_filter.strip()
        return [row for row in rows if row.get("League", "").strip() == wanted]

    def fetch(self, competition: Competition, season: str) -> pd.DataFrame:
        """Return one competition-season in the canonical schema.

        Raises:
            SeasonUnavailableError: If the provider holds no such season.
        """
        if competition.feed is Feed.MAIN:
            rows = read_provider_csv(self.ensure_file(competition, season))
            column_map = MAIN_COLUMNS
        else:
            rows = [
                row
                for row in self._read_country_file(competition)
                if row.get("Season") and normalise_season(row["Season"]) == season
            ]
            column_map = EXTRA_COLUMNS

        if not rows:
            raise SeasonUnavailableError(f"{competition.id} {season}: no rows")

        return self._to_canonical(rows, competition, season, column_map)

    # -- normalisation --------------------------------------------------------

    def _to_canonical(
        self,
        rows: list[dict[str, str]],
        competition: Competition,
        season: str,
        column_map: dict[str, str],
    ) -> pd.DataFrame:
        """Map provider rows onto the canonical schema.

        Unplayed and unusable rows are dropped here, counted, and logged. They
        are a normal part of every feed — a country file always carries the
        current season's scheduled fixtures alongside its played ones — so
        dropping them silently would hide a real problem behind an expected
        one, and failing on them would make the current season unfetchable.
        """
        records: list[dict[str, object]] = []
        dropped_unplayed = 0
        dropped_invalid = 0
        nulled_odds = 0
        nulled_shots = 0

        for row in rows:
            mapped = {canonical: row.get(source, "") for source, canonical in column_map.items()}

            match_date = parse_date(mapped["date"])
            home_team = mapped["home_team"].strip()
            away_team = mapped["away_team"].strip()
            home_goals = parse_int(mapped["home_goals"])
            away_goals = parse_int(mapped["away_goals"])

            # A fixture with no date, no teams or no score has not been played.
            # Every country file carries the rest of the current season this
            # way, so this is the common case, not an anomaly.
            if match_date is None or not home_team or not away_team:
                dropped_unplayed += 1
                continue
            if home_goals is None or away_goals is None:
                dropped_unplayed += 1
                continue
            if home_team == away_team:
                # Never observed, and impossible. Worth a distinct counter: it
                # would mean the columns had shifted, which padding a ragged
                # row could in principle cause.
                dropped_invalid += 1
                continue

            # The provider's own result column is cross-checked rather than
            # trusted. Measured across all 305,499 ingested matches it never
            # disagreed with
            # the score, which is exactly why a disagreement now would mean
            # something structural is wrong rather than one typo.
            derived = derive_result(home_goals, away_goals)
            stated = mapped["result"].strip().upper()
            if stated and stated != derived:
                dropped_invalid += 1
                logger.warning(
                    "%s %s: %s v %s stated %s but scored %d-%d; dropped",
                    competition.id,
                    season,
                    home_team,
                    away_team,
                    stated,
                    home_goals,
                    away_goals,
                )
                continue

            odds_home, odds_draw, odds_away = pick_odds(row)
            if (odds_home, odds_draw, odds_away) != (None, None, None):
                overround = 1 / odds_home + 1 / odds_draw + 1 / odds_away  # type: ignore[operator]
                if overround < MIN_PLAUSIBLE_OVERROUND:
                    # Null the price, keep the match. The result is still a
                    # perfectly good training row; only the odds are wrong, and
                    # a benchmark computed from an impossible book would be
                    # quietly and confidently misleading.
                    odds_home = odds_draw = odds_away = None
                    nulled_odds += 1

            iso_date = match_date.strftime("%Y-%m-%d")

            record: dict[str, object] = {
                "match_id": make_match_id(
                    PROVIDER_NAME, competition.id, season, iso_date, home_team, away_team
                ),
                "provider": PROVIDER_NAME,
                "competition_id": competition.id,
                "country": competition.country,
                "competition": competition.name,
                "tier": competition.tier,
                "season": season,
                "date": match_date,
                "kickoff": mapped.get("kickoff", "").strip() or None,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": team_id(competition.id, home_team),
                "away_team_id": team_id(competition.id, away_team),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": derived,
                "referee": mapped.get("referee", "").strip() or None,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }

            # Half-time and match statistics: present or null, never absent.
            for canonical in ("ht_home_goals", "ht_away_goals"):
                record[canonical] = parse_int(mapped.get(canonical, ""))
            ht_home, ht_away = record["ht_home_goals"], record["ht_away_goals"]
            record["ht_result"] = (
                derive_result(ht_home, ht_away)
                if isinstance(ht_home, int) and isinstance(ht_away, int)
                else None
            )
            for canonical in (
                "home_shots",
                "away_shots",
                "home_shots_on_target",
                "away_shots_on_target",
                "home_corners",
                "away_corners",
                "home_fouls",
                "away_fouls",
                "home_yellows",
                "away_yellows",
                "home_reds",
                "away_reds",
            ):
                record[canonical] = parse_int(mapped.get(canonical, ""))

            # A shot on target is a shot, so more of the latter than the former
            # is impossible. Rare — 9 rows in 128,498, seven of them in 2000/01,
            # the first season the provider published shot data at all — and
            # there is no way to tell which of the pair is wrong, so both go.
            # Nulling the field beats dropping the match: the result is valid,
            # and a rolling shot-accuracy feature computed over "0 shots, 5 on
            # target" is not.
            for side in ("home", "away"):
                shots = record[f"{side}_shots"]
                on_target = record[f"{side}_shots_on_target"]
                if isinstance(shots, int) and isinstance(on_target, int) and on_target > shots:
                    record[f"{side}_shots"] = None
                    record[f"{side}_shots_on_target"] = None
                    nulled_shots += 1

            records.append(record)

        if dropped_unplayed or dropped_invalid:
            logger.info(
                "%s %s: kept %d, dropped %d unplayed and %d invalid",
                competition.id,
                season,
                len(records),
                dropped_unplayed,
                dropped_invalid,
            )
        if nulled_odds or nulled_shots:
            logger.info(
                "%s %s: nulled %d implausible odds triple(s) and %d shot pair(s)",
                competition.id,
                season,
                nulled_odds,
                nulled_shots,
            )
        if not records:
            raise SeasonUnavailableError(f"{competition.id} {season}: no playable matches")

        return self._as_canonical_frame(records)

    @staticmethod
    def _as_canonical_frame(records: list[dict[str, object]]) -> pd.DataFrame:
        """Build the frame with every canonical column, in order, typed.

        Reindexing rather than trusting the records to be complete: a record
        that omitted a key would otherwise produce a frame missing a column,
        and the schema contract is that the shape never varies.
        """
        frame = pd.DataFrame.from_records(records).reindex(columns=list(CANONICAL_SCHEMA))
        return frame.astype(CANONICAL_SCHEMA)

    def observed_capabilities(self, frame: pd.DataFrame) -> frozenset[Capability]:
        """Which capabilities this frame actually populated.

        The registry declares what a *feed* can supply; this reports what a
        season really did. The two differ constantly — the primary feed carries
        no shot data before 2000/01 — and conflating them is how a column of
        nulls gets mistaken for a competition that simply does not report it.
        """
        observed = {Capability.RESULTS}
        checks: tuple[tuple[Capability, str], ...] = (
            (Capability.HALF_TIME, "ht_home_goals"),
            (Capability.MATCH_STATS, "home_shots"),
            (Capability.REFEREE, "referee"),
            (Capability.ODDS, "odds_home"),
        )
        for capability, column in checks:
            if frame[column].notna().any():
                observed.add(capability)
        return frozenset(observed)
