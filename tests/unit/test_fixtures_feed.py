"""The fixture list: what it keeps, what it drops, and the id it builds.

The load-bearing assertion in this file is :func:`test_the_id_is_the_one_the_played_match_will_carry`.
Everything else here is bookkeeping; that one is the join the whole archive
rests on, and if it breaks, a forecast is served, logged, and never scored —
silently, forever, because an id that matches nothing looks exactly like a
match that has not been played yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import fixtures as feed
from src.ingestion.base import make_match_id
from src.ingestion.csv_reader import ProviderFileError
from src.ingestion.football_data import PROVIDER_NAME
from tests.factories import league_registry

TODAY = pd.Timestamp("2026-09-09")


def row(
    *,
    div: str = "E0",
    date: str = "12/09/2026",
    home: str = "Arsenal",
    away: str = "Chelsea",
    home_goals: str = "",
    away_goals: str = "",
    kickoff: str = "15:00",
) -> dict[str, str]:
    """One line of the published file, in its own column vocabulary."""
    return {
        "Div": div,
        "Date": date,
        "Time": kickoff,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": "",
    }


class TestSeasonFor:
    def test_a_mid_season_fixture_takes_the_season_being_played(self) -> None:
        played = feed.Played(date=pd.Timestamp("2026-09-06"), season="2026-27")
        assert feed.season_for(pd.Timestamp("2026-09-12"), played) == "2026-27"

    def test_a_season_that_ran_long_keeps_its_label(self) -> None:
        # 2019-20 finished in August 2020 across fifteen competitions. Counting
        # months would file these fixtures under 2020-21 and every id would be
        # wrong; asking what is being played gets it right.
        played = feed.Played(date=pd.Timestamp("2020-07-22"), season="2019-20")
        assert feed.season_for(pd.Timestamp("2020-07-26"), played) == "2019-20"

    def test_after_a_summer_the_fixture_starts_a_new_season(self) -> None:
        played = feed.Played(date=pd.Timestamp("2026-05-24"), season="2025-26")
        assert feed.season_for(pd.Timestamp("2026-08-15"), played) == "2026-27"

    def test_july_belongs_to_the_season_it_opens(self) -> None:
        # Ligue 1 has opened in July since 1993 and the provider files those
        # matches under the season that is starting, not the one that ended.
        assert feed.season_for(pd.Timestamp("2026-07-30"), None) == "2026-27"

    def test_the_spring_half_belongs_to_the_season_that_opened(self) -> None:
        assert feed.season_for(pd.Timestamp("2027-02-14"), None) == "2026-27"


class TestToFrame:
    def test_the_id_is_the_one_the_played_match_will_carry(self) -> None:
        """A fixture and the match it becomes are the same row to the archive.

        Built the long way on both sides on purpose: this asserts that the six
        parts of the natural key agree, which is the only reason a forecast
        served on Friday can be scored on Monday.
        """
        frame = feed.to_frame([row()], league_registry())
        expected = make_match_id(
            PROVIDER_NAME, "ENG_1", "2026-27", "2026-09-12", "Arsenal", "Chelsea"
        )
        assert frame.loc[0, "match_id"] == expected

    def test_a_fixture_has_no_result(self) -> None:
        frame = feed.to_frame([row()], league_registry())
        assert frame.loc[0, ["home_goals", "away_goals", "result"]].isna().all()

    def test_the_odds_are_left_out_even_when_the_file_carries_them(self) -> None:
        # The closing line is the benchmark and it is read off the
        # played row. A pre-match price under the same column name would be a
        # different number wearing it.
        priced = row() | {"B365H": "2.10", "B365D": "3.40", "B365A": "3.60"}
        frame = feed.to_frame([priced], league_registry())
        assert frame.loc[0, ["odds_home", "odds_draw", "odds_away"]].isna().all()

    def test_a_row_with_a_score_is_a_match_and_is_dropped(self) -> None:
        assert feed.to_frame([row(home_goals="2", away_goals="1")], league_registry()).empty

    def test_a_division_the_registry_does_not_carry_is_skipped(self) -> None:
        assert feed.to_frame([row(div="SP1")], league_registry()).empty

    def test_a_blank_division_is_skipped_without_being_named(self) -> None:
        assert feed.to_frame([row(div="")], league_registry()).empty

    @pytest.mark.parametrize(
        "broken",
        [
            {"date": ""},
            {"home": ""},
            {"away": ""},
            {"home": "Arsenal", "away": "Arsenal"},
        ],
    )
    def test_an_unusable_row_is_dropped(self, broken: dict[str, str]) -> None:
        assert feed.to_frame([row(**broken)], league_registry()).empty

    def test_a_date_in_an_unknown_format_raises_rather_than_emptying_the_table(self) -> None:
        """The adapter's own behaviour, inherited rather than softened.

        A blank date is an ordinary unplayed row and is dropped; a date in a
        format this project does not know is the published file changing shape,
        and dropping every row for it would look exactly like a quiet week.
        """
        with pytest.raises(ProviderFileError):
            feed.to_frame([row(date="2026-09-12")], league_registry())

    def test_fixtures_before_the_window_are_dropped(self) -> None:
        rows = [row(date="01/09/2026"), row(date="12/09/2026", home="Everton")]
        frame = feed.to_frame(rows, league_registry(), since=TODAY)
        assert list(frame["home_team"]) == ["Everton"]

    def test_the_season_comes_from_what_the_competition_is_playing(self) -> None:
        played = {"ENG_1": feed.Played(date=pd.Timestamp("2026-09-06"), season="2026-27")}
        frame = feed.to_frame([row()], league_registry(), played=played)
        assert frame.loc[0, "season"] == "2026-27"

    def test_an_empty_file_gives_an_empty_frame_with_every_column(self) -> None:
        frame = feed.to_frame([], league_registry())
        assert frame.empty
        assert "match_id" in frame.columns and "home_goals" in frame.columns

    def test_the_rows_come_back_in_date_order(self) -> None:
        rows = [row(date="14/09/2026"), row(date="12/09/2026", home="Everton")]
        frame = feed.to_frame(rows, league_registry())
        assert list(frame["date"]) == [pd.Timestamp("2026-09-12"), pd.Timestamp("2026-09-14")]


class TestLatestPlayed:
    def test_it_reports_the_newest_match_per_competition(self) -> None:
        matches = pd.DataFrame(
            {
                "competition_id": ["ENG_1", "ENG_1", "SPA_1"],
                "date": pd.to_datetime(["2026-09-01", "2026-09-06", "2026-08-30"]),
                "season": ["2026-27", "2026-27", "2026-27"],
            }
        )
        found = feed.latest_played(matches)
        assert found["ENG_1"].date == pd.Timestamp("2026-09-06")
        assert found["SPA_1"].date == pd.Timestamp("2026-08-30")

    def test_an_empty_table_has_nothing_to_carry_forward(self) -> None:
        assert feed.latest_played(pd.DataFrame(columns=["competition_id", "date", "season"])) == {}


class TestReadAndDownload:
    def test_no_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert feed.read(tmp_path / "fixtures.csv") == []

    def test_an_unreadable_file_is(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "fixtures.csv"
        path.write_text("Div,Date\nE0,12/09/2026\n", encoding="utf-8")

        def refuse(_: Path) -> list[dict[str, str]]:
            raise ProviderFileError("not csv")

        monkeypatch.setattr(feed, "read_provider_csv", refuse)
        with pytest.raises(feed.FixtureFeedError):
            feed.read(path)

    def test_a_real_file_reads_back(self, tmp_path: Path) -> None:
        path = tmp_path / "fixtures.csv"
        path.write_text("Div,Date,HomeTeam,AwayTeam\nE0,12/09/2026,Arsenal,Chelsea\n", "utf-8")
        assert feed.read(path)[0]["HomeTeam"] == "Arsenal"

    def test_the_url_is_the_published_one(self) -> None:
        assert feed.fixtures_url("https://example.test") == "https://example.test/fixtures.csv"

    def test_a_page_instead_of_a_fixture_list_is_removed_and_reported(self, tmp_path: Path) -> None:
        """The provider answers a missing file with HTML and a non-error status.

        Removed rather than left on disk: `--offline` reads whatever is there,
        and an HTML page cached as a fixture list is a command that fails
        differently every run.
        """

        class Serving:
            def download(self, _url: str, destination: Path, **_: object) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("<html><title>nope</title></html>", encoding="utf-8")

        with pytest.raises(feed.FixtureFeedError):
            feed.download(Serving(), tmp_path)  # type: ignore[arg-type]
        assert not (tmp_path / "football-data" / "fixtures.csv").exists()

    def test_a_fixture_list_is_kept(self, tmp_path: Path) -> None:
        class Serving:
            def download(self, _url: str, destination: Path, **_: object) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("Div,Date\nE0,12/09/2026\n", encoding="utf-8")

        path = feed.download(Serving(), tmp_path)  # type: ignore[arg-type]
        assert path.is_file()
