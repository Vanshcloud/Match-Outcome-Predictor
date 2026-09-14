"""Canonical team identity.

Downstream code joins on ``home_team_id``, never on ``home_team``. That
indirection is the seam a second provider plugs into: football-data's
"Man United" and API-Football's "Manchester United" have to become the same
team without every feature module learning both spellings.

**What this deliberately is not: a fuzzy matcher.** That was a measurement, not
a preference. Across every Premier League season the provider publishes — 34
seasons, 12,724 matches — it uses just 51 distinct team strings, and "Arsenal"
in 1993 is byte-identical to "Arsenal" in 2024. The clubs appearing in exactly
one season (Barnsley, Blackpool, Luton, Oldham, Swindon) are genuine
one-season promotions, not renames.

So a similarity threshold has nothing to find on this vocabulary, and real
damage to do. The two most similar *distinct* pairs both score exactly 0.800:
"Sheffield United" against "Sheffield Weds", and "Barnsley" against "Burnley".
Any threshold low enough to catch a hypothetical rename merges Sheffield
United with Sheffield Wednesday — two clubs from the same city that have played
each other. Exact matching has no such failure mode.

So identity is exact, with an explicit alias table for the cases a second
provider will introduce. :func:`find_single_season_teams` covers the residual
risk — a genuine rename surfaces as a warning naming both spellings, rather
than silently splitting one club's history into two half-length records.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

# Exact, hand-maintained cross-provider name equivalences, keyed by
# ``(country_code, provider spelling)`` and mapping to the canonical spelling.
#
# Empty on purpose. This project currently reads one provider whose vocabulary
# is internally consistent, so every entry here would be a guess about a
# provider not yet integrated. Populated when a second results provider is added
# source, where the mappings can be verified against two real vocabularies
# instead of imagined.
ALIASES: Mapping[tuple[str, str], str] = {}


def slugify(name: str) -> str:
    """Reduce a team name to a stable, URL-safe token.

    Accents are folded rather than stripped as unknown bytes: the Turkish,
    Greek and Portuguese divisions carry names the provider has not always
    transliterated consistently, and NFKD decomposition followed by dropping
    combining marks makes "Beşiktaş" and "Besiktas" land on the same token.

    Raises:
        ValueError: If nothing survives normalisation. A team whose name is
            entirely punctuation is a parsing failure upstream, and returning
            an empty id would let it join to every other broken row.
    """
    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    slug = _NON_ALPHANUMERIC.sub("-", folded).strip("-")
    if not slug:
        raise ValueError(f"team name reduces to an empty slug: {name!r}")
    return slug


def country_code(competition_id: str) -> str:
    """Extract the three-letter country prefix from a competition id.

    ``"ENG_1"`` -> ``"ENG"``. Team identity is scoped to a country rather than
    to a competition, so a club keeps one id as it is promoted and relegated —
    which is precisely the continuity a rolling form feature depends on.
    """
    prefix, _, rest = competition_id.partition("_")
    if not rest or len(prefix) != 3 or not prefix.isupper():
        raise ValueError(f"competition id must look like 'ENG_1', got {competition_id!r}")
    return prefix


def team_id(
    competition_id: str,
    name: str,
    aliases: Mapping[tuple[str, str], str] | None = None,
) -> str:
    """Return the canonical identifier for a team.

    Readable rather than hashed — ``eng:man-united`` — because these ids appear
    in logs, error messages and dashboard URLs, and an opaque digest turns
    every debugging session into a lookup.

    Args:
        competition_id: Supplies the country scope.
        name: The provider's spelling.
        aliases: Cross-provider equivalences. Defaults to :data:`ALIASES`.
    """
    code = country_code(competition_id)
    table = ALIASES if aliases is None else aliases
    canonical = table.get((code, name.strip()), name)
    return f"{code.lower()}:{slugify(canonical)}"


def find_single_season_teams(
    rows: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Return teams that appear in exactly one season.

    This is the rename detector. A club that genuinely played one season is
    indistinguishable *here* from a club that was renamed and so appears to
    start and stop — the difference needs a human. Surfacing both as a warning
    is the honest option; the failure it prevents is a rename splitting one
    club's history into two half-length records that every rolling feature then
    computes over the wrong window.

    Args:
        rows: ``(team_id, season)`` pairs.

    Returns:
        ``{team_id: the one season}``, for the validation report.
    """
    seasons: dict[str, set[str]] = defaultdict(set)
    for identifier, season in rows:
        seasons[identifier].add(season)
    return {
        identifier: next(iter(found)) for identifier, found in seasons.items() if len(found) == 1
    }
