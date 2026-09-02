# Data sources

Everything here was measured against real files, not read off a schema. Each
quirk names the file it was found in, so a future change can tell which
real-world case it would break.

## Provider: football-data.co.uk

Free, keyless, static CSV over HTTPS. No account, no rate limit, no scraping,
no browser automation. Two file layouts, both behind one adapter.

| | Primary (`main`) | Secondary (`extra`) |
|---|---|---|
| URL | `/mmz4281/{season_code}/{div}.csv` | `/new/{country}.csv` |
| Granularity | one competition-season | one country, every season |
| Competitions | 22 | 17 (16 country files) |
| History | 1993/94 onward | ~2012 onward |
| Per-match detail | shots, shots on target, corners, fouls, cards, referee | none |
| Odds | yes, many books | closing only |
| Season encoding | in the URL, always split | a column, split *or* calendar |

### Licence

The provider publishes no licence granting redistribution, and asks for
acknowledgement of its sources. **This repository therefore ships no match
data** — not even a test slice. The unit suite runs on synthetic fixtures
shaped like the real files, so it needs no network; tests that require real
data are marked `integration` and skip when it is absent.

## Discovered quirks

Fifteen, all load-bearing. Where a fix lives in code, the module is named.
The last two were found only by running a full ingest and asserting
football-shaped invariants over all 305,499 resulting matches.

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

## Deliberately not used

| Source | Reason |
|---|---|
| **Transfermarkt** | ToU §11.1 bans automated access *and* training models on the content. |
| **FBref / Sports Reference** | Cloudflare-blocked; HTTP 403 on every policy page. |
| **FiveThirtyEight SPI** | Feed stopped updating when 538 was wound down; usable history, no live fixtures. |
| **API-Football** | Free tier is 100 requests/day, unusable at this scale. |
