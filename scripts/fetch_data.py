#!/usr/bin/env python
"""Download and ingest match data into the canonical table.

    python scripts/fetch_data.py                      # every competition
    python scripts/fetch_data.py --competition ENG_1  # one, repeatable
    python scripts/fetch_data.py --country England    # every English division
    python scripts/fetch_data.py --list               # show the registry

A full run is roughly 700 requests and takes about fifteen minutes at the
default one-second spacing, most of it spent discovering that a division did
not exist in 1994. It is resumable: settled seasons are cached permanently, so
a second run costs almost nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/fetch_data.py` from anywhere without an editable
# install. Prepending the project root is what makes `import src...` resolve;
# resolving it from this file rather than from the working directory is what
# makes it work when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.football_data import FootballDataProvider  # noqa: E402
from src.ingestion.registry import Competition, load_registry  # noqa: E402
from src.pipelines.ingest import run_ingest  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.http import HttpClient  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("fetch_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="competition id, e.g. ENG_1. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--country",
        action="append",
        default=[],
        metavar="NAME",
        help="every competition in this country. Repeatable.",
    )
    parser.add_argument("--list", action="store_true", help="list the registry and exit")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def select(
    registry_competitions: tuple[Competition, ...], args: argparse.Namespace
) -> tuple[Competition, ...]:
    """Resolve the CLI filters to a competition list.

    An unknown id is an error rather than an empty selection: a typo that
    silently ingests nothing looks exactly like a provider outage.
    """
    if not args.competition and not args.country:
        return registry_competitions

    known = {c.id: c for c in registry_competitions}
    selected: dict[str, Competition] = {}

    for competition_id in args.competition:
        if competition_id not in known:
            raise SystemExit(
                f"unknown competition {competition_id!r}. " f"Known: {', '.join(sorted(known))}"
            )
        selected[competition_id] = known[competition_id]

    for country in args.country:
        matched = [c for c in registry_competitions if c.country.lower() == country.lower()]
        if not matched:
            countries = sorted({c.country for c in registry_competitions})
            raise SystemExit(f"unknown country {country!r}. Known: {', '.join(countries)}")
        selected.update({c.id: c for c in matched})

    return tuple(selected.values())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    registry = load_registry()

    if args.list:
        print(f"{len(registry.competitions)} competitions\n")
        print(f"{'id':9s} {'feed':6s} {'tier':>4s}  {'country':13s} name")
        for competition in registry.competitions:
            tier = "-" if competition.tier is None else str(competition.tier)
            print(
                f"{competition.id:9s} {competition.feed.value:6s} {tier:>4s}  "
                f"{competition.country:13s} {competition.name}"
            )
        return 0

    targets = select(registry.competitions, args)
    logger.info("ingesting %d competition(s)", len(targets))

    with HttpClient(
        timeout_seconds=settings.http.timeout_seconds,
        max_retries=settings.http.max_retries,
        backoff_factor=settings.http.backoff_factor,
        user_agent=settings.http.user_agent,
        min_request_interval_seconds=settings.http.min_request_interval_seconds,
    ) as client:
        provider = FootballDataProvider(registry, settings.paths.raw_dir, client=client)
        report = run_ingest(provider, settings.paths.processed_dir, competitions=targets)

    if report.matches == 0:
        logger.error("no matches ingested")
        return 1

    print(f"\n{report.summary()}")
    print(f"written to {report.output}")
    if report.competitions_empty:
        print(f"no data found for: {', '.join(report.competitions_empty)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
