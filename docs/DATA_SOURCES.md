# Data sources

Everything here was measured against real files, not read off a schema. Each
quirk names the file it was found in, so a future change can tell which
real-world case it would break.

## Provider: football-data.co.uk

Free, keyless, static CSV over HTTPS. No account, no rate limit, no scraping,
no browser automation. Two file layouts, both behind one adapter.

| | Primary (`main`) | Secondary (`extra`) | Fixtures |
|---|---|---|---|
| URL | `/mmz4281/{season_code}/{div}.csv` | `/new/{country}.csv` | `/fixtures.csv` |
| Granularity | one competition-season | one country, every season | every primary division, about ten days ahead |
| Competitions | 22 | 17 (16 country files) | the primary 22 |
| History | 1993/94 onward | ~2012 onward | none — it is rewritten as matches are played |
| Per-match detail | shots, shots on target, corners, fouls, cards, referee | none | none; there has been no match |
| Odds | yes, many books | closing only | pre-match, and deliberately not read |
| Season encoding | in the URL, always split | a column, split *or* calendar | **absent** — see quirk 17 |

The third column is Milestone 20's, and it is the same provider on purpose.
A fixture has to carry the `match_id` its played row will carry a week later,
or the forecast made about it can never be joined to the result; every part of
that key — the competition, the date, the club names *as this provider spells
them* — is already shared, so there is no name matching between the two and no
alias table to keep. `src/ingestion/fixtures.py` reads it, and nothing from it
is ever written to the canonical table.

### Licence

The provider publishes no licence granting redistribution, and asks for
acknowledgement of its sources. **This repository therefore ships no match
data** — not even a test slice. The unit suite runs on synthetic fixtures
shaped like the real files, so it needs no network; tests that require real
data are marked `integration` and skip when it is absent.

## Discovered quirks

Sixteen, all load-bearing. Where a fix lives in code, the module is named.
The last three were found only by running a full ingest and asserting
invariants over all 303,517 resulting matches — the last of them by an
assumption check written two milestones later, for the feature layer.

### 1. A missing file returns HTTP 300, not 404

`/mmz4281/9596/SC3.csv` answers **HTTP 300 Multiple Choices** with an HTML
body and `content-type: text/html`. `requests.raise_for_status()` does not
treat 3xx as an error, so the download succeeds and an error page lands on
disk named `.csv`. Parsed, it becomes garbage rows.

Three distinct "missing" responses exist: `404` (clean), `300` + HTML, and a
country file that simply lacks the rows. `SeasonUnavailableError` unifies them.
→ `csv_reader.looks_like_csv`, `football_data.ensure_file`

### 2. Mixed character encodings

Most files are UTF-8. English 2004/05 contains byte `0xa0`, a cp1252
non-breaking space, and raises `UnicodeDecodeError` under UTF-8. English
2021/22 and 2024/25 additionally carry a UTF-8 BOM — without `utf-8-sig` it
becomes an invisible prefix on the first column name, so `df["Div"]` raises
`KeyError` for a column plainly visible in the header.
→ `csv_reader.decode`, fallback chain `utf-8-sig` → `cp1252`

### 3. Ragged rows, in both directions

English 2003/04: a 57-column header with 32 rows of 72 fields and 13 of 62.
English 1993/94 and 1994/95: 28-column headers with rows of 7.

**This one taught the most.** The reader originally *raised* whenever a
truncated field was non-empty, reasoning that surplus data meant a shifted row.
Running over all forty competitions falsified it at once: Italian Serie B
2003/04 has a **42-column header whose last eight names are blank**, and rows
of 49 fields carrying extra unnamed statistics. Every named column in those
rows is correctly aligned — `Atalanta 4-1 Triestina` reads perfectly. The
"misalignment" was the header being narrower than the data.

Worse, raising aborted the entire run: one malformed file out of ~700 discarded
every competition after it. The rule is now pad, truncate, and *warn*; only
named columns are read, so a field beyond the last named column cannot affect
one. The real guard is semantic and lives in the adapter.
→ `csv_reader.read_provider_csv`, `pipelines.ingest.ingest_competition`

### 4. Entirely blank rows

English 1993/94 ships 552 rows, 90 of them empty. Dropping them leaves 462 —
exactly a 22-team double round-robin, which is how the rule was confirmed
rather than assumed.

### 5. Unnamed header columns

The 1993/94 header has 7 names and 21 blanks. Collected into a dict under a
shared `""` key, all but the last are silently discarded. They are skipped.

### 6. The header grows from 21 to 120 columns

| Season | Columns | Adds |
|---|---|---|
| 1995/96 | 21 | results, half-time |
| 2000/01 | 45 | shots, corners, fouls, cards, referee, attendance |
| 2003/04 | 57 | first odds (Bet365) |
| 2009/10 | 71 | Betbrain averages |
| 2019/20 | 106 | market-average closing odds |
| 2024/25 | 120 | more books |

Columns also *disappear*: `Attendance`, `HHW`/`AHW` (hit woodwork), `HO`/`AO`
(offsides) and `HBP`/`ABP` (booking points) exist only in the 2000–2004 window.
None is carried, because a feature available for four seasons out of thirty-two
is a feature no model can use.

### 7. Two date formats

`dd/mm/yy` until 2014/15, `dd/mm/yyyy` after. Both appear in a single ingest.
The two-digit form is unambiguous here because `%y` pivots at 69 and this
provider's history runs 1993–2068.

### 8. Season semantics differ per competition, and change mid-history

The secondary feed's `Season` column is `int64` for calendar-year leagues
(Brazil, USA, Ireland, Norway) and `object` for split ones (Austria, Poland,
Romania). **Japan contains both**: the J1 League ran on calendar years through
2025 and switches to `2026/2027`. Season format is therefore read per row,
never inferred from the country.

Canonical labels: `2024-25` (split) and `2024` (calendar). Both sort
chronologically as text, which is what lets a temporal split be a comparison.

### 9. Whitespace pollution in identity columns

The provider's own data contains `'Ireland '` alongside `'Ireland'`,
`'Premier Division '` alongside `'Premier Division'`, and `' J1 League'`
alongside `'J1 League'`. Compared unstripped, a competition silently splits in
two, each with half its history.

### 10. Country files can hold more than one competition

| File | Competitions |
|---|---|
| `ARG.csv` | Liga Profesional **and** Copa de la Liga Profesional — a league and a cup |
| `SWZ.csv` | Super League **and** Challenge League — two tiers |

Unfiltered, Argentina's rows merge a league and a knockout competition into one
impossible table. → `Competition.league_filter`

### 11. Odds columns changed twice

No single triple spans the history: nothing before 2003/04, Bet365 only until
Betbrain averages appear around 2009, market averages (`AvgC*`) only from about
2019. A fallback chain picks the best available, and **all three prices must
come from the same triple** — mixing books produces an overround no real market
had, corrupting the benchmark this project measures itself against.

### 12. The provider's own notes are out of date

`notes.txt` states that free kicks conceded (`HFKC`/`AFKC`) are published in
place of fouls for the French 2nd, Belgian 1st and Greek 1st divisions.
Checked across those three divisions in 2009/10, 2015/16, 2019/20 and 2024/25:
**`HFKC` appears in no header in any era.** Those divisions simply carry no
fouls column before 2019/20 and use `HF` after. No fallback was added, because
adding one for a column that does not exist is speculation.

### 13. Unplayed fixtures are shipped alongside played ones

Every country file carries the remainder of the current season as rows with
teams, a date and no score. Failing on them makes the current season
unfetchable; keeping them puts a match with no result into the training target.
They are dropped and counted.

### 14. Impossible bookmaker prices

A real book's implied probabilities always sum above 1.0 — the excess is the
margin. Across 246,052 priced matches the median overround is **1.074**, a
textbook 7.4%. But **33 rows (0.013%)** sum *below* 1.0, which would be free
money: River Plate v Arsenal Sarandí, 2023, priced 2.39 / 5.72 / 10.84 sums to
0.685. Twenty-three of the thirty-three are in 2025-26, suggesting recent
entry errors rather than a historical format change.

The odds triple is nulled and the match kept. The result is still a perfectly
good training row; only the price is wrong, and a benchmark computed from an
impossible book would be quietly and confidently misleading.
→ `football_data.MIN_PLAUSIBLE_OVERROUND`

### 15. Shots on target exceeding shots

A shot on target is a shot, so the reverse is impossible. **9 rows in 128,498
(0.007%)** violate it — Coventry 2000/01 with 0 shots and 5 on target. Seven of
the nine are in 2000/01, the first season the provider published shot data at
all.

There is no way to tell which of the pair is wrong, so both are nulled and the
match kept — a rolling shot-accuracy feature computed over "0 shots, 5 on
target" is worse than one computed over a gap.

### 16. Five files are copies of another division, served under its name

The worst of them, because every row is individually perfect.

| File | Contains | Cost of believing the URL |
|---|---|---|
| `1993-94/P1.csv` | Spanish La Liga | 380 fabricated Portuguese matches |
| `1993-94/SC1.csv` | Spanish La Liga | 380 fabricated Scottish matches |
| `1993-94/SP2.csv` | Spanish La Liga | 380 duplicated fixtures |
| `1994-95/SP2.csv` | Spanish La Liga | 380 duplicated fixtures |
| `1995-96/SP2.csv` | Spanish La Liga | 462 duplicated fixtures |

**1,982 matches, 0.65% of the table.** Barcelona, Real Madrid and Athletic
Bilbao appeared in Portugal's and Scotland's first seasons wearing `por:` and
`sco:` team ids — forty phantom clubs whose entire history was one season, and
which the rename detector duly reported as one-season teams for two milestones
without anyone asking why.

Nothing per-row could catch it. Every copy has the right teams, the right date
and the right score; the schema is valid, the results agree with the scores,
the odds imply a real book. And because `match_id` hashes the competition, the
same fixture under two division codes gets two ids and survives deduplication.

**The file says so itself.** Every primary-feed row carries a `Div` column, and
in all five of these it reads `SP1`. The URL is only where a file was
published; the file is the data. The adapter now drops rows whose `Div` names
a different division, and keeps rows where it is blank — some early files leave
it empty on the odd row, and dropping those would trade a rare provider error
for a common one.
→ `football_data._rows_for_this_division`,
  `validation.matches` ("no fixture appears in two competitions")

Found by asking whether a team ever plays twice on one date — the assumption
every rolling feature rests on. It did, 2,444 times.

### 17. The fixture file publishes no season

Every other file says which season it is: the primary feed in its URL, the
secondary in a column. `fixtures.csv` says neither, and the season is part of
the natural key a `match_id` is built from — so a fixture filed under the wrong
label gets an id the played match will never carry, and the forecast sits in
the archive as permanently unresolved rather than failing.

`season_for` therefore asks the data rather than the calendar: a competition
that has played inside the last 30 days is mid-season, and the fixture belongs
to the season those matches belong to. Only a competition that has been idle
longer gets a label counted from the fixture's own date, with July as the
boundary — Ligue 1 has opened in July since 1993.

Measured over every ingested match by handing each one its predecessor: **19 of
270,848 labels disagree**, all of them a competition resuming after a break of
more than a month. The rule that counts months instead gets 2,175 wrong, and
743 of those are the 2019-20 season running into July 2020 across fifteen
competitions — which is exactly the case asking the data survives.

### A note on the Swiss "Challenge League"

`SWZ.csv` carries a `Challenge League` label that is **not** the second tier:
it is two matches, Thun v Sion home and away in May 2021 — the Super League
promotion/relegation playoff. Registered as a competition it would be a league
with one season and two fixtures, which no rolling feature can use and every
per-competition report would have to caveat. It is deliberately absent from the
registry. Argentina still exercises the `league_filter` mechanism, since its
file really does hold two competitions.

## Team identity

Names are **stable within this provider** — that was measured, not assumed.
Across the 34 Premier League seasons it publishes (12,724 matches) it uses just
51 distinct team strings, and "Arsenal" in 1993 is byte-identical to "Arsenal"
in 2024. Clubs appearing in exactly one season (Barnsley, Blackpool, Luton,
Oldham, Swindon) are genuine one-season promotions, not renames.

So **no fuzzy matching**. The two most similar *distinct* pairs both score
exactly 0.800 — `Sheffield United`/`Sheffield Weds` and `Barnsley`/`Burnley` —
so any threshold low enough to catch a hypothetical rename merges two clubs
from the same city that have played each other.

Identity is exact, scoped to the country (`eng:man-united`), so a club keeps
one id through promotion and relegation. An alias table exists for the second
provider and is deliberately empty until there is one to verify against.
`find_single_season_teams` reports rename candidates rather than guessing.

## Incremental re-runs

The provider serves `ETag` and `Last-Modified` on every file, and honours both
`If-None-Match` and `If-Modified-Since` with **304 Not Modified and no body**.
Change detection is therefore authoritative rather than guessed, and the
pipeline uses it instead of a file-age heuristic — age is wrong in both
directions: it re-downloads unchanged files once they get old enough, and it
misses a file that changed five minutes ago.

| File class | Ordinary run | `--revalidate` |
|---|---|---|
| Settled season (finished ≥2 years) | not contacted | conditional request |
| Current / recent season | conditional request | conditional request |
| Secondary-feed country file | conditional request, **once per run** | same |
| Known missing (cached, TTL 7 days) | not contacted | not contacted |

"Settled" means *will not grow*, not *will never change* — the provider does
revise history to correct a scoreline. `--revalidate` catches that for the
price of a 304 per file. `--forget-misses` retries everything previously
recorded as unpublished.

### Measured, over the full 39-competition ingest

| | First run | Second run |
|---|---|---|
| Files transferred | 80 (**10.20 MB**) | 0 (**0.00 MB**) |
| Revalidated (304, no body) | 6 | 60 |
| 404 misses re-probed | 2 | 0 |
| Output `sha256` | `8d489ed3…` | `8d489ed3…` |

The canonical Parquet is **byte-identical** across runs, which is what makes
the checksum in the manifest meaningful. It is also independent of the order
competitions are ingested in, because the table is sorted by
`(date, competition_id, match_id)` before writing.

### State on disk

| Path | Purpose |
|---|---|
| `data/raw/fetch_cache.json` | Per-URL `ETag`, `Last-Modified`, `sha256`, last-checked time, and whether the provider publishes it at all |
| `data/raw/manifest.json` | Checksum of every cached provider file — provenance of the **inputs** |
| `data/processed/manifest.json` | Checksum, row count and date range of the canonical table — provenance of the **output** |

Both manifests are verified by the integration suite. A changed checksum is
*reported*, never raised on: the provider genuinely does revise files, and
whether that is corruption or a correction is the caller's judgement.

Losing the fetch cache costs one slow run and nothing else — a corrupt or
version-mismatched cache is discarded rather than raising, because an
optimisation that can halt the pipeline is a liability.

## Deliberately not used

| Source | Reason |
|---|---|
| **Transfermarkt** | ToU §11.1 bans automated access *and* training models on the content. |
| **FBref / Sports Reference** | Cloudflare-blocked; HTTP 403 on every policy page. |
| **FiveThirtyEight SPI** | Feed stopped updating when 538 was wound down; usable history, no live fixtures. |
| **API-Football** | Free tier is 100 requests/day, unusable at this scale. |
