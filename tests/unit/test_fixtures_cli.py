"""The two entry points that close the loop, driven without a network or a server.

`fixtures` fetches and builds; `price` asks. Both are commands somebody will
run unattended from cron, so what is asserted here is mostly what they do when
something is not there — a provider that is down, a table that is stale, a
service that has not been restarted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from src.pipelines.features import build_features
from src.pipelines.ratings import build_ratings
from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame

TODAY = pd.Timestamp.today().normalize()


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_cli", PROJECT_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIXTURES = _load("fixtures")
PRICE = _load("price")


def _fixture_file(path: Path, *, when: pd.Timestamp, home: str = "Team 00") -> None:
    """The published file, in its own column vocabulary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        f"E0,{when:%d/%m/%Y},15:00,{home},Team 01,,,\n",
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory with a match table recent enough not to be called stale."""
    processed, features, raw = tmp_path / "processed", tmp_path / "features", tmp_path / "raw"
    processed.mkdir(parents=True)
    features.mkdir(parents=True)

    league = league_frame()
    # Re-dated so the table ends today: the command reports how far behind it
    # is, and a synthetic league that ended in 2026 would make that number a
    # property of the calendar rather than of the test.
    shifted = league.assign(date=league["date"] + (TODAY - league["date"].max()))
    shifted.to_parquet(processed / "matches.parquet", index=False)
    build_ratings(shifted).to_parquet(features / "ratings.parquet", index=False)
    build_features(shifted).to_parquet(features / "features.parquet", index=False)

    _fixture_file(raw / "football-data" / "fixtures.csv", when=TODAY + pd.Timedelta(2, "D"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


class TestFixturesCommand:
    def test_it_builds_design_rows_for_what_is_about_to_be_played(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert FIXTURES.main(["--offline"]) == 0
        written = pd.read_parquet(workspace / "features" / "upcoming.parquet")
        assert len(written) == 1
        assert written.loc[0, "home_team"] == "Team 00"
        # The row is a design row, not a fixture list: it carries what the
        # model prices, which is the whole reason the command exists.
        assert written.loc[0, "home_form_points_5"] is not None
        assert "restart it" in capsys.readouterr().out

    def test_a_provider_that_is_down_falls_back_to_the_file_already_fetched(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file fetched this morning is worth more than an empty table, and
        the fallback is announced rather than silent."""

        def refuse(*_: object, **__: object) -> Path:
            raise RuntimeError("503")

        monkeypatch.setattr(FIXTURES.feed, "download", refuse)
        assert FIXTURES.main([]) == 0
        assert len(pd.read_parquet(workspace / "features" / "upcoming.parquet")) == 1

    def test_a_provider_that_is_down_with_nothing_on_disk_is_an_error(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (workspace / "raw" / "football-data" / "fixtures.csv").unlink()

        def refuse(*_: object, **__: object) -> Path:
            raise RuntimeError("503")

        monkeypatch.setattr(FIXTURES.feed, "download", refuse)
        assert FIXTURES.main([]) == 1

    def test_a_successful_fetch_is_what_gets_read(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched = workspace / "raw" / "elsewhere.csv"
        _fixture_file(fetched, when=TODAY + pd.Timedelta(1, "D"), home="Team 02")
        monkeypatch.setattr(FIXTURES.feed, "download", lambda *_, **__: fetched)
        assert FIXTURES.main([]) == 0
        written = pd.read_parquet(workspace / "features" / "upcoming.parquet")
        assert written.loc[0, "home_team"] == "Team 02"

    def test_fixtures_past_the_window_are_not_priced(self, workspace: Path) -> None:
        """A fixture three weeks out will be priced again with another round of
        results behind it, and today's row would be the one the archive scored."""
        _fixture_file(
            workspace / "raw" / "football-data" / "fixtures.csv",
            when=TODAY + pd.Timedelta(20, "D"),
        )
        assert FIXTURES.main(["--offline"]) == 0
        assert pd.read_parquet(workspace / "features" / "upcoming.parquet").empty

    def test_a_stale_match_table_is_said_out_loud(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pricing this Saturday against a month-old table is doing something
        slightly worse than the operator thinks, not something wrong. The
        number is what says which, so it is a warning rather than a refusal.

        Read off stderr rather than ``caplog``: the command configures logging
        itself, which replaces the root handlers pytest's capture fixture
        installs."""
        matches = pd.read_parquet(workspace / "processed" / "matches.parquet")
        aged = matches.assign(date=matches["date"] - pd.Timedelta(30, "D"))
        aged.to_parquet(workspace / "processed" / "matches.parquet", index=False)
        assert FIXTURES.main(["--offline"]) == 0
        assert "30 days behind" in capsys.readouterr().err

    def test_a_dry_run_builds_nothing(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert FIXTURES.main(["--offline", "--dry-run"]) == 0
        assert "nothing built" in capsys.readouterr().out
        assert not (workspace / "features" / "upcoming.parquet").exists()

    def test_without_a_match_table_it_says_which_command_makes_one(self, workspace: Path) -> None:
        (workspace / "processed" / "matches.parquet").unlink()
        assert FIXTURES.main(["--offline"]) == 1

    def test_more_fixtures_than_fit_on_screen_are_counted_rather_than_printed(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        lines = "\n".join(
            f"E0,{(TODAY + pd.Timedelta(1, "D")):%d/%m/%Y},15:00,Team {index:02d},Team 19,,,"
            for index in range(12)
        )
        (workspace / "raw" / "football-data" / "fixtures.csv").write_text(
            "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n" + lines + "\n", encoding="utf-8"
        )
        assert FIXTURES.main(["--offline"]) == 0
        assert "and 2 more" in capsys.readouterr().out


class StubResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class StubClient:
    """The service, without a service."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []

    def __enter__(self) -> StubClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, _url: str, *, json: dict[str, Any]) -> StubResponse:
        self.sent.append(json)
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return StubResponse(answer)


@pytest.fixture
def priceable(workspace: Path) -> Path:
    assert FIXTURES.main(["--offline"]) == 0
    return workspace


def _client(monkeypatch: pytest.MonkeyPatch, *responses: Any) -> StubClient:
    stub = StubClient(*responses)
    monkeypatch.setattr(PRICE, "HttpClient", lambda **_: stub)
    return stub


class TestPriceCommand:
    @pytest.mark.usefixtures("priceable")
    def test_it_asks_the_service_about_every_fixture(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stub = _client(monkeypatch, {"predictions": [{}], "unresolved": [], "recorded": 1})
        assert PRICE.main([]) == 0
        assert len(stub.sent[0]["fixtures"]) == 1
        assert "1 priced, 1 written to the prediction log" in capsys.readouterr().out

    @pytest.mark.usefixtures("priceable")
    def test_a_fixture_the_service_cannot_find_names_the_restart(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ordinary way this fails: `make fixtures` has run and the process
        that indexes at startup has not been restarted since."""
        _client(
            monkeypatch,
            {"predictions": [], "unresolved": [{"match_id": "abc"}], "recorded": 0},
        )
        assert PRICE.main([]) == 0
        printed = capsys.readouterr().out
        assert "could not find: abc" in printed
        assert "restart it" in printed

    @pytest.mark.usefixtures("priceable")
    def test_a_service_that_does_not_answer_stops_rather_than_repeating_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _client(monkeypatch, RuntimeError("connection refused"))
        assert PRICE.main([]) == 1

    def test_batches_are_capped_at_what_the_service_accepts(
        self, priceable: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = pd.read_parquet(priceable / "features" / "upcoming.parquet")
        pd.concat([table, table.assign(match_id="second")], ignore_index=True).to_parquet(
            priceable / "features" / "upcoming.parquet", index=False
        )
        empty = {"predictions": [], "unresolved": [], "recorded": 0}
        stub = _client(monkeypatch, empty, empty)
        assert PRICE.main(["--batch", "1"]) == 0
        assert len(stub.sent) == 2

    @pytest.mark.usefixtures("workspace")
    def test_no_fixture_table_says_which_command_builds_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert PRICE.main([]) == 0
        assert "run `make fixtures`" in capsys.readouterr().out

    @pytest.mark.usefixtures("priceable")
    def test_a_dry_run_asks_for_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stub = _client(monkeypatch)
        assert PRICE.main(["--dry-run"]) == 0
        assert stub.sent == []
        assert "nothing asked for" in capsys.readouterr().out

    def test_the_table_can_be_named(self, priceable: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = _client(monkeypatch, {"predictions": [{}], "unresolved": [], "recorded": 1})
        moved = priceable / "elsewhere.parquet"
        (priceable / "features" / "upcoming.parquet").rename(moved)
        assert PRICE.main(["--upcoming", str(moved)]) == 0
        assert stub.sent


def test_the_settings_the_price_command_reads_are_the_service_defaults() -> None:
    """``API_MAX_BATCH``'s default is fifty and a larger batch is a 422, so the
    two numbers have to agree without anybody remembering to make them."""
    from src.utils.config import ApiConfig

    assert ApiConfig().max_batch == PRICE.DEFAULT_BATCH


def test_the_stub_response_shape_matches_what_the_service_returns() -> None:
    """The stub above answers three keys; this is what says they are the right
    three, so a schema change breaks the test rather than the cron job."""
    from api.schemas import BatchResponse

    assert {"predictions", "unresolved", "recorded"} == set(BatchResponse.model_fields)
