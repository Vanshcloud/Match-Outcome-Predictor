#!/usr/bin/env python
"""Fetch what is about to be played, and build the design rows to price it with.

    python scripts/fixtures.py                   # fetch, build, write
    python scripts/fixtures.py --offline         # use the file already fetched
    python scripts/fixtures.py --days 3          # a shorter window
    python scripts/fixtures.py --dry-run         # print it, write nothing

The first half of the loop the archive needs. Without it the service only
priced matches that had already been played, because the
canonical table holds results and the service prices what is in the table. So
every forecast the archive scored was in-sample, and the drift column was
empty for a structural reason rather than a broken one.

This asks :mod:`src.ingestion.fixtures` for the published fixture list — the
same provider that publishes the results, so the club names already match the
table — appends the fixtures to the canonical frame, runs the ratings and the
features over the whole thing, and writes the served columns for the fixtures
alone.

**Run `make data` first.** Fixtures are priced from the history in the table,
so a table that is a fortnight behind prices this Saturday on a fortnight-old
form window. Not a failure, and not silent: the command prints how stale the
table is and says so above the threshold.

**Then restart the service.** The API indexes its tables once, in the lifespan,
which is what makes its cache directives honest — nothing it says can change
while the process that said it is running. So a new fixture table reaches the
service the same way a new model does: `docker compose restart api`.

Exits non-zero when the provider cannot be reached and there is no file already
fetched, or when the match table is missing. Finding *no* fixtures is not an
error: a Wednesday in June is a Wednesday in June, and the table is written
empty so that the service stops offering last week's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.ingestion import fixtures as feed  # noqa: E402
from src.ingestion.registry import load_registry  # noqa: E402
from src.pipelines.fixtures import run_upcoming  # noqa: E402
from src.pipelines.tables import read_matches, resolve_tables  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.http import HttpClient  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("fixtures")

DEFAULT_DAYS = 10
"""How far ahead a fixture is worth pricing.

The provider publishes about a week and a half, and the window is a cap on that
rather than a request. Ten days rather than the whole file because a design row
is only as good as the history behind it: a fixture three weeks out will be
priced again — with another week of results in the form windows — long before
it kicks off, and the row written today would be the one the archive scored.
"""

STALE_DAYS = 7
"""How far behind the match table may be before the command says so.

A week, because the form windows are five matches long: a table that has missed
a round is pricing on a window that is one match short, which is a real effect
and a small one. Two weeks is not.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS, help="how far ahead to price, in days"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the fixture file already on disk instead of fetching",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, but write nothing")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    tables = resolve_tables(settings.paths, matches=args.matches)
    matches = read_matches(tables.matches, columns=None)
    if matches is None:
        logger.error("no match table at %s; run `make data` first", tables.matches)
        return 1

    today = pd.Timestamp.today().normalize()
    behind = (today - matches["date"].max()).days
    print(f"\n{len(matches):,} match(es) in the table, the most recent {behind} day(s) ago")
    if behind > STALE_DAYS:
        # A warning rather than a refusal. Someone pricing a fixture list
        # against a stale table is doing something slightly worse than they
        # think, not something wrong, and the number is what says which.
        logger.warning(
            "the match table is %d days behind; every form window is that stale. "
            "Run `make data` first",
            behind,
        )

    path = feed.local_path(settings.paths.raw_dir)
    if not args.offline:
        with HttpClient() as client:
            try:
                path = feed.download(client, settings.paths.raw_dir)
            except Exception as error:  # noqa: BLE001 - reported, then fall back
                # The provider being unreachable is not the same as there being
                # no fixtures, and a file fetched this morning is worth more
                # than an empty table. Falls back and says it did.
                logger.error("could not fetch the fixture list: %s", error)
                if not path.is_file():
                    return 1
                logger.warning("using the copy already at %s", path)

    rows = feed.read(path)
    fixtures = feed.to_frame(
        rows,
        load_registry(),
        played=feed.latest_played(matches),
        since=today,
    )
    if args.days is not None and not fixtures.empty:
        fixtures = fixtures[fixtures["date"] <= today + pd.Timedelta(args.days, "D")]

    print(f"{len(rows):,} row(s) in the fixture list, {len(fixtures)} of them priceable\n")
    if not fixtures.empty:
        print(
            fixtures[["date", "competition_id", "home_team", "away_team"]]
            .head(10)
            .to_string(index=False)
        )
        if len(fixtures) > 10:
            print(f"  ... and {len(fixtures) - 10} more")

    if args.dry_run:
        print("\n(dry run: nothing built, nothing written)")
        return 0

    report = run_upcoming(matches, fixtures, settings.paths.features_dir)
    print(f"\n{report.summary()}")
    if report.already_played:
        print(f"{report.already_played} fixture(s) already in the match table, dropped")
    print(f"written to {report.output}")
    print("\nThe service indexes its tables at startup: restart it to price these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
