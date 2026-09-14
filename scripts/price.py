#!/usr/bin/env python
"""Ask the service to price every upcoming fixture, so the archive has something to score.

    python scripts/price.py                          # price them all
    python scripts/price.py --url http://api:8000    # a service somewhere else
    python scripts/price.py --dry-run                # say what it would ask for

The second half of the loop. `make fixtures` puts the design
rows where the service can index them; this is the part that *asks*. Nothing
before it did: the API answers when a page or a caller happens to want a
number, and a deployment nobody browses on a Friday night serves nothing, logs
nothing, and gives the archive nothing to score.

**Over HTTP, like every other client.** This does not import the model and does
not import :mod:`api`. There is exactly one process in this system that holds
the artefact, and a second one that priced a fixture directly would be a second
answer to "what does the model say" with nothing comparing the two. The
forecast is written to the prediction log by the service, as a side effect of
serving it, which is the same path a browser's request takes.

**It prices fixtures, not matches.** The ids come from the upcoming table, so
every forecast this sends is for a match that has not been played — which is
what makes it out-of-sample, which is the entire reason the archive was empty.
The service stamps ``in_sample`` from the artefact's own training window; it
will say false for these, and it said true for all 25 rows the first archive run found.

Exits non-zero when the service is not reachable or is not ready. An empty
upcoming table is not an error — it is a week with no football in it, or a
`make fixtures` that has not run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.tables import read_upcoming, resolve_tables  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.http import HttpClient  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("price")

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_BATCH = 50
"""Fixtures per request, matching ``API_MAX_BATCH``'s own default.

A batch that exceeds the service's limit is a 422 rather than a smaller batch,
so the default here is the default there. ``--batch`` is how a deployment that
lowered its limit lowers this one.
"""

TIMEOUT_SECONDS = 60.0
"""Longer than the dashboard's, and for a reason: this is the request that can
arrive at a service which has just restarted and is still indexing three
hundred thousand fixtures."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="where the service is")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="fixtures per request")
    parser.add_argument("--upcoming", type=Path, default=None, help="fixture table to read")
    parser.add_argument("--dry-run", action="store_true", help="say what it would ask for")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    tables = resolve_tables(settings.paths, upcoming=args.upcoming)
    path = args.upcoming or tables.upcoming
    upcoming = read_upcoming(path) if path is not None else None
    if upcoming is None or upcoming.empty:
        print(f"\nno fixtures at {path}; run `make fixtures`")
        return 0

    ids = [str(value) for value in upcoming["match_id"]]
    print(f"\n{len(ids)} fixture(s) to price against {args.url}")
    if args.dry_run:
        print("(dry run: nothing asked for)")
        return 0

    base = args.url.rstrip("/")
    served = recorded = 0
    unresolved: list[str] = []
    with HttpClient(timeout_seconds=TIMEOUT_SECONDS) as client:
        for start in range(0, len(ids), args.batch):
            chunk = ids[start : start + args.batch]
            payload = {"fixtures": [{"match_id": one} for one in chunk]}
            try:
                response = client.post(f"{base}/predict/batch", json=payload)
            except Exception as error:  # noqa: BLE001 - the one failure worth stopping on
                # Stopping rather than continuing. Every chunk goes to the same
                # service, so the second one is going to fail the same way, and
                # forty more identical stack traces help nobody.
                logger.error("the service did not answer: %s", error)
                return 1
            body = response.json()
            served += len(body["predictions"])
            recorded += int(body.get("recorded", 0))
            unresolved += [one.get("match_id") or "?" for one in body.get("unresolved", [])]

    print(f"{served} priced, {recorded} written to the prediction log")
    if unresolved:
        # Not a failure. It is what a service that has not been restarted since
        # `make fixtures` looks like, and saying which fixtures it could not
        # find is more useful than saying how many.
        print(f"{len(unresolved)} the service could not find: {', '.join(unresolved[:5])}")
        print("  the service indexes at startup — restart it if `make fixtures` has just run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
