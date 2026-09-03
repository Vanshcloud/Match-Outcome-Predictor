"""The dataset card: what is actually in the table, generated from the table.

A dataset card that is written by hand is a dataset card that is wrong. This
one is produced from the data every time validation runs, so the coverage
figures, the date ranges and the class balance cannot drift from what a model
would actually be trained on — and the file's checksum is printed at the top,
so the numbers are attached to specific bytes rather than to "the dataset".

The per-competition table is the part worth reading. Aggregate coverage hides
the thing that matters: 93% of recent matches in stat-capable competitions
carry shot data, and the National League carries almost none. A feature that
depends on shots is unusable in one of those competitions and fine in the other
twenty-one, and only the breakdown says which.

Every figure comes from SQL against :class:`~src.storage.duckdb_store.DuckDBStore`
rather than from pandas over a materialised frame — the card is exactly the
kind of column-scan aggregate the analytical store exists to answer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src import __version__
from src.ingestion.base import (
    BENCHMARK_COLUMNS,
    POST_MATCH_COLUMNS,
    PRE_MATCH_COLUMNS,
    TARGET_COLUMN,
)
from src.ingestion.manifest import checksum
from src.storage.duckdb_store import MATCHES_VIEW, DuckDBStore

if TYPE_CHECKING:
    from src.validation.report import ValidationReport

# One row per competition. `any_value` rather than `max` on the denormalised
# columns because they are constant within a competition — a fact the
# "competition metadata agrees with the registry" check is what guarantees.
_COVERAGE_SQL = f"""
WITH sides AS (
    SELECT competition_id, home_team_id AS team FROM {MATCHES_VIEW}
    UNION ALL
    SELECT competition_id, away_team_id FROM {MATCHES_VIEW}
),
teams AS (
    SELECT competition_id, count(DISTINCT team) AS teams FROM sides GROUP BY competition_id
)
SELECT
    m.competition_id,
    any_value(m.country)              AS country,
    any_value(m.competition)          AS competition,
    any_value(m.tier)                 AS tier,
    count(*)                          AS matches,
    count(DISTINCT m.season)          AS seasons,
    any_value(t.teams)                AS teams,
    min(m.date)                       AS first_match,
    max(m.date)                       AS last_match,
    avg(CASE WHEN m.home_shots     IS NOT NULL THEN 1.0 ELSE 0.0 END) AS stats,
    avg(CASE WHEN m.ht_home_goals  IS NOT NULL THEN 1.0 ELSE 0.0 END) AS half_time,
    avg(CASE WHEN m.odds_home      IS NOT NULL THEN 1.0 ELSE 0.0 END) AS odds,
    avg(CASE WHEN m.result = 'H'   THEN 1.0 ELSE 0.0 END)             AS home_win
FROM {MATCHES_VIEW} AS m
JOIN teams AS t USING (competition_id)
GROUP BY m.competition_id
ORDER BY country, tier NULLS LAST, m.competition_id
"""

_TOTALS_SQL = f"""
SELECT
    count(*)                                   AS matches,
    count(DISTINCT competition_id)             AS competitions,
    count(DISTINCT country)                    AS countries,
    count(DISTINCT season)                     AS seasons,
    min(date)                                  AS first_match,
    max(date)                                  AS last_match,
    avg(CASE WHEN result = 'H' THEN 1.0 ELSE 0.0 END) AS home_win,
    avg(CASE WHEN result = 'D' THEN 1.0 ELSE 0.0 END) AS draw,
    avg(CASE WHEN result = 'A' THEN 1.0 ELSE 0.0 END) AS away_win
FROM {MATCHES_VIEW}
"""


def competition_coverage(store: DuckDBStore) -> pd.DataFrame:
    """Per-competition rows, seasons, date range and per-block coverage."""
    return store.query(_COVERAGE_SQL)


def totals(store: DuckDBStore) -> pd.DataFrame:
    """One row of headline figures."""
    return store.query(_TOTALS_SQL)


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _tier(value: Any) -> str:
    """Render a nullable tier.

    Cups have no tier, and DuckDB hands the null back as pandas' ``NAType``,
    which is neither ``None`` nor a float — so the obvious ``value is None``
    test passes it straight through to ``int()`` and raises.
    """
    return "-" if pd.isna(value) else str(int(value))


def _team_count(store: DuckDBStore) -> int:
    result: Any = store.query(
        f"SELECT count(DISTINCT team) AS n FROM ("
        f"  SELECT home_team_id AS team FROM {MATCHES_VIEW}"
        f"  UNION SELECT away_team_id FROM {MATCHES_VIEW})"
    )
    return int(result.iloc[0]["n"])


def build_card(
    store: DuckDBStore,
    report: ValidationReport,
    *,
    source: Path | None = None,
) -> str:
    """Render the dataset card as Markdown.

    Args:
        store: An open store with the matches view attached.
        report: The validation run whose result is embedded. Passed in rather
            than run here so the card and the pipeline's gate can never
            disagree about what was checked.
        source: The Parquet file, for its checksum. Omitted in tests, where the
            table may be synthetic and its bytes uninteresting.
    """
    # Rows are converted to plain dictionaries at this boundary rather than
    # read off the frame. A DuckDB result has no static column types, so every
    # `row.tier` and `head["draw"]` is an opaque object to the type checker;
    # converting once says "these are dynamic query results" in one place
    # instead of scattering twenty casts through the rendering.
    head: dict[str, Any] = totals(store).to_dict("records")[0]  # type: ignore[assignment]
    coverage = competition_coverage(store)
    rows: list[dict[str, Any]] = coverage.to_dict("records")  # type: ignore[assignment]
    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Dataset card — canonical match table",
        "",
        "<!-- GENERATED by scripts/validate_data.py. Do not edit by hand: the next",
        "     validation run overwrites it, and a hand-edited figure is a figure",
        "     that no longer describes the data. -->",
        "",
        f"Generated {generated} by match-outcome-predictor {__version__}.",
        "",
        "## What this is",
        "",
        "One row per completed football match, in a single schema across every",
        "competition and both of the provider's file layouts. It is the input to",
        "every feature, model and evaluation in this repository, and the only",
        "table any of them read.",
        "",
        "## At a glance",
        "",
        "| | |",
        "|---|---|",
        f"| Matches | {int(head['matches']):,} |",
        f"| Competitions | {int(head['competitions'])} |",
        f"| Countries | {int(head['countries'])} |",
        f"| Teams | {_team_count(store):,} |",
        f"| Seasons | {int(head['seasons'])} |",
        f"| Date range | {head['first_match'].date()} to {head['last_match'].date()} |",
        f"| Home / draw / away | {_percent(head['home_win'])} / "
        f"{_percent(head['draw'])} / {_percent(head['away_win'])} |",
    ]

    if source is not None and source.is_file():
        lines += [
            f"| File | `{source.name}`, {source.stat().st_size / 1e6:.1f} MB |",
            f"| SHA-256 | `{checksum(source)}` |",
        ]

    lines += [
        "",
        "## The prediction target",
        "",
        f"`{TARGET_COLUMN}` — three ordered classes, H / D / A, from the home team's",
        "perspective. Ordered matters: predicting Away when the result was Home is a",
        "worse error than predicting Draw, which is why the Ranked Probability Score",
        "is a primary metric here and accuracy is not.",
        "",
        "## What is knowable before kick-off",
        "",
        "The single most expensive mistake available in this dataset is training on",
        "a column that does not exist until the match is over. `home_shots` is not a",
        "property of the fixture; it is a summary of the ninety minutes. Those",
        "columns are kept, because a team's shots in its *previous* matches are a",
        "legitimate feature — the distinction is temporal, not columnar, so it",
        "cannot be enforced by leaving data out.",
        "",
        f"**Pre-match ({len(PRE_MATCH_COLUMNS)} columns).** "
        + ", ".join(f"`{column}`" for column in sorted(PRE_MATCH_COLUMNS)),
        "",
        f"**Post-match ({len(POST_MATCH_COLUMNS)} columns)** — usable only through a lag. "
        + ", ".join(f"`{column}`" for column in sorted(POST_MATCH_COLUMNS)),
        "",
        "**Benchmark, not features.** "
        + ", ".join(f"`{column}`" for column in sorted(BENCHMARK_COLUMNS))
        + ". Pre-match and therefore safe, but reserved as the comparison this",
        "project measures itself against. A model trained on odds learns to copy the",
        "bookmaker: it looks excellent in validation and collapses the moment prices",
        "are unavailable.",
        "",
        "## Validation",
        "",
        report.summary() + ".",
        "",
        report.to_markdown(),
        "",
        "## Coverage by competition",
        "",
        "Aggregate coverage hides what matters. A feature that depends on shots is",
        "usable in some of these competitions and not in others, and only the",
        "breakdown says which.",
        "",
        "| Competition | Country | Tier | Matches | Seasons | Teams | From | To | Stats | Half-time | Odds | Home win |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['competition']} (`{row['competition_id']}`) | {row['country']} "
            f"| {_tier(row['tier'])} | {int(row['matches']):,} | {int(row['seasons'])} "
            f"| {int(row['teams'])} | {row['first_match'].date()} | {row['last_match'].date()} "
            f"| {_percent(row['stats'])} | {_percent(row['half_time'])} | {_percent(row['odds'])} "
            f"| {_percent(row['home_win'])} |"
        )

    lines += [
        "",
        "## Known limitations",
        "",
        "- **Completed matches only.** Rows without a full-time score are dropped at",
        "  ingest; this table cannot be used to enumerate upcoming fixtures.",
        "- **No xG, weather, injuries, lineups, travel or attendance.** None of these",
        "  is available at match level across these competitions from any free",
        "  source. The feature registry has slots so that any which becomes",
        "  available is one entry rather than a refactor.",
        "- **Statistics coverage is a timeline, not a constant.** The provider added",
        "  shot data to most European divisions around 2019/20, so a feature built on",
        "  it has a shorter usable history than the table's date range suggests.",
        "- **One provider.** Every figure here inherits that provider's errors. The",
        "  quirks found and handled are recorded in `docs/DATA_SOURCES.md`.",
        "- **Not redistributed.** No match data is committed to this repository;",
        "  `python scripts/fetch_data.py` reproduces it from the source.",
        "",
    ]
    return "\n".join(lines)


def write_card(destination: Path, card: str) -> Path:
    """Write the card, creating parent directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(card, encoding="utf-8")
    return destination
