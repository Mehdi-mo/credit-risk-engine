# Slice 2 — cleaning and exploration

## Target
- Built from `loan_status`: Charged Off + Default = 1, Fully Paid = 0.
- Unfinished loans removed (Current, Late, In Grace Period) — outcome not yet known.
- "Does not meet the credit policy" loans removed — different lending rules.
- 29,619 usable loans from a 49,658-row random sample. **Bad rate 20.0%.**

## Leakage
- 64 of 151 columns dropped; 39 of them were leakage.
- Three families: payments after funding, credit checks made later,
  and hardship/settlement programmes.
- Trap found: `fico_range_low` (at application, usable) vs
  `last_fico_range_low` (years later, forbidden).

## Data quality
- All columns arrived from Postgres as TEXT; 71 converted cleanly to numbers.
- `emp_length`: naive digit extraction merged "< 1 year" with "1 year" — fixed.
- `dti` uses 999 as an "unknown" code (11 rows) — set to missing.
- `annual_inc` uses 0 for "not provided" (4 rows) — set to missing.
- 16 joint-applicant columns dropped: 98.7% missing after filtering to
  finished loans, because joint applications only started in 2017.

## Signal found
- Grade: 6.2% (A) to 53.4% (G). Monotonic. Confirms the target is correct.
- Purpose: 7% to 30%; `small_business` highest. 80% of loans are
  debt consolidation or credit card.
- FICO: 25% down to 14%. Weaker than grade, because grade contains FICO.
- `loan_amnt / annual_inc` (engineered): 12.7% to 31.7%, monotonic.
  Strong candidate for slice 3.

## Known limitations
- No borrower below FICO 660 exists in the data — Lending Club rejected them.
  The model cannot say anything about that population.
- Small purpose categories (educational: 3 loans) need folding into `other`.