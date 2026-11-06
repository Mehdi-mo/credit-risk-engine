# Slice 3 — feature engineering

Input: `model_table.parquet` (29,619 loans, 81 features)
Output: `features_train.parquet` (20,902 rows) and `features_test.parquet` (5,378 rows),
71 model features each.

## The rule this slice is built on

Split the data first. Any step that **learns a number from a column** —
a median, a list of rare levels, a scaling factor — may only learn it from
the training half, then be applied unchanged to the test half.

Row-by-row steps (`loan_amnt / annual_inc`, parsing `"36 months"`) compute each
row from itself alone and are safe in any order.

The split is **by time**, not random: in production you always predict the
future, never a random sample of the past.

## Data restricted to 2013–2017

- **2018 dropped.** Only 1,234 finished loans, and they are mostly early
  prepayers — bad rate 14.3%, far below the truth.
- **Pre-2013 dropped.** 153 loans before 2010, no 60-month product existed,
  crisis-era lending rules. A different product.

26,280 loans remain. Train 2013-01 → 2016-08, test 2016-09 → 2017-12.

### Censoring runs in both directions

A loan only appears here if it finished. That biases recent vintages, but not
in one direction:

- **Very recent loans look too good.** They can be fully paid *early* (months),
  but a charge-off needs ~5 months of missed payments. So the newest rows are
  mostly early prepayers.
- **Moderately recent loans look too bad.** Defaults have had time to arrive,
  but loans that simply ran to maturity have not finished yet.

Train 20.0% vs test 23.7%: mostly this artefact, not risk drift. A fixed
performance window ("went bad within 18 months") would fix it, but needs payment
dates — which are leakage and were removed in slice 2. Cost of a correct decision.

## Features built

Scored by the ratio between the worst and best decile bad rate.

| feature | ratio | kept |
|---|---|---|
| `grade` (LC's own verdict, held back) | 8.6× | for slice 6 comparison |
| `loan_to_income` = `loan_amnt / annual_inc` | **2.4×** | ✅ |
| FICO | 1.8× | ✅ |
| `mths_since_recent_inq` | **1.64×** | ✅ |
| `open_to_total_acc` = `open_acc / total_acc` | **1.50×** | ✅ |
| `revol_util` | 1.40× | ✅ |
| `credit_history_months` = `issue_d − earliest_cr_line` | 1.29× | ✅ weak |
| `revol_to_income` = `revol_bal / annual_inc` | 1.19× flat | ❌ |

Also: `fico` = midpoint of `fico_range_low`/`_high` (always 4 apart — two columns,
one fact). Logs of `annual_inc`, `revol_bal`, `tot_cur_bal` (skew > 1).
`loan_amnt` not logged — skew 0.77.

### "Amount ÷ capacity" — and each amount has its own capacity

`loan_to_income` works (2.4×). `revol_to_income` does not (1.19×, flat) — because
for revolving debt the right denominator is the **credit limit**, not income.
That column already exists as `revol_util`, and it works (1.40×).

### Every "how long ago" is measured from `issue_d`

Never from today. Measuring against today would credit a 2013 borrower with 13
years of credit history that did not exist at the decision. Same sin as
`last_fico_range_low`, better disguised.

`credit_history_months` capped at 600 (50 years); 24 rows had 1950s credit lines.
Capped, not deleted — the other columns on those rows are fine. The cap comes from
domain judgement, so it is safe pre-split; a cap read off `.quantile(0.999)` would
be a learned step.

## Missing values

### Selection bias killed the delinquency family

Blank in `mths_since_last_delinq` should mean "never delinquent" — good news.
It carries almost nothing here: 20.4% vs 21.2%. And recency among those who do
have a record is flat too (1.16×).

Reason: these are loans LC **approved**. Anyone with a bad delinquency record was
already screened out, and those approved with a blemish had to be strong elsewhere.
The variation was removed before the data was written.

> In a dataset of approved applications, anything the lender screened on hard
> will look weak.

Same effect explains FICO at only 1.8× (no borrower below 660 exists) and why
`grade` dominates at 8.6× — it was formed when LC could still see the full range.
This is **range restriction**; lenders address it with reject inference, which is
not possible here (no rejected applications in the data).

Dropped: `mths_since_last_delinq`, `_recent_bc_dlq`, `_recent_revol_delinq`,
`_last_major_derog`, `_last_record`, `_recent_bc`.

### The inquiry family survived

`mths_since_recent_inq`: blank 15.8% vs present 21.3%, and recency runs 25.9%
(within a month) down to 16.0% (16–24 months). The blank group lands exactly where
the "longest ago" bucket lands — confirming blank means "no recent credit-seeking"
and sits on the same scale.

So blanks are filled with **36**, not the median. The median (~5 months) would have
placed the safest group among the riskiest. The fill value comes from what the
blank means, not from a default.

> What the lender screened on hard is dead. What the lender under-weighted is alive.

## Columns whose missingness tracks time

17 columns dropped. Lending Club added a batch of fields in 2016
(`total_bal_il`, `all_util`, `inq_fi`, `open_rv_*`, `inq_last_12m`,
`mths_since_rcnt_il`, …). Their missing rate is a step: 1.00 before, 0.00 after —
and the step lands near the train/test boundary.

Such a column is a **date in disguise**. The model learns "missing = old = safer",
which is true in training and useless for a new application.

`mths_since_rcnt_il_missing` initially looked like a feature (gap +0.046). Holding
the year fixed (2016 only) collapsed it to +0.008 on 180 rows. The test:

> When a feature looks good, ask what else it might be measuring, hold that fixed,
> and see if the signal survives.

`emp_length` was caught by the same sweep and **kept** — its missingness drifts
gradually (5.1% → 7.1% across 2013–2017), which is a change in filling-in habits,
not a schema change. An automatic threshold finds candidates; a person decides.

## Categories

| column | decision | reason |
|---|---|---|
| `purpose` | one-hot, 13 → 8 levels | `small_business` 32.5% vs `car` 15.6% |
| `home_ownership` | one-hot, 4 → 3 | MORTGAGE 17.8% < OWN 22.2% < RENT 24.2% |
| `verification_status` | one-hot, all 3 kept | 14.9% / 21.8% / 24.9% |
| `application_type` | one-hot | Joint 28.5% vs Individual 20.7%, 396 rows |
| `initial_list_status` | dropped | 21.0% vs 20.4% — no signal |
| `disbursement_method` | dropped | 37.3% gap, but only 67 rows — noise |
| `addr_state`, `zip_code` | **in the table, out of the model** | fairness |

### Verified borrowers default more

Not Verified 14.9%, Source Verified 21.8%, Verified 24.9%.

Verification is not a property of the borrower — it is a record of LC's suspicion.
They verify when something looks doubtful. Known before funding, so not leakage,
but like `grade` it carries the lender's judgement rather than the applicant's
circumstances. Fragile in production: change the verification policy and the
column's meaning changes underneath the model.

`MORTGAGE` beating `OWN` is the same flavour: a mortgage means a bank already
screened you and you are still paying.

### Geography stays out of the model

In US lending, zip code and state proxy for race closely enough that a model
using them can reproduce redlining without anyone intending it. Both are kept in
the saved files so slice 7 can test whether errors fall unevenly across groups —
bias cannot be measured with the evidence discarded.

## Folding rare levels

A fold threshold on size alone is too blunt. `small_business` (240 rows, 32.5%)
and `car` (231 rows, 15.6%) are nearly the same size and nothing alike. Fold a
level when it is small **and** unremarkable.

Folding also needs a real destination: folding `ANY` (2 rows) into a new `other`
merely renamed it. It went into `MORTGAGE`, the largest group, where 2 rows dilute
to nothing.

## Bug found and fixed in slice 2

`emp_length` in `model_table.parquet` was scrambled — 38.4% missing against 5.8%
in the source text, and a by-year pattern of 0.19 / 0.71 / 0.18 / 0.45 / 0.44.

Cause: in notebook 2, `labelled` kept the original gappy index while `model_df`
had been reset to 0,1,2,… Assigning a Series to a column aligns **by index label,
not by position**, so ~40% of rows got `NaN` and the remaining ~60% received
another borrower's value. No error, no warning.

Fixed with `.reset_index(drop=True)` on the source Series. Missing share is now
0.058 and the level counts match the raw text exactly.

Note: the slice-3 row-count assertion passed throughout. It checked that the
counts matched, which they did. An assertion only tests what you thought to ask.

## Second bug, found in slice 4: rows were not in time order

`split_by_time` used two boolean masks:

```python
train = df[df['issue_d'] <  cut].copy()
test  = df[df['issue_d'] >= cut].copy()
```

That picks the right **rows** — every train row is genuinely before the cut — but
masking preserves whatever order the frame already had. The order traced back to
a `SELECT` with no `ORDER BY`, so it was arbitrary.

Harmless until slice 4 needed `TimeSeriesSplit`, which cuts by **row position**
and assumes earlier rows are earlier in time. Folds built on arbitrary order
would have trained on a mix of 2013–2016 and scored on another mix — producing
numbers that look fine and mean nothing.

Caught by `assert train['issue_d'].is_monotonic_increasing` before any fold ran.

> Filtering by a date gives you the right rows. It does not give them to you in
> date order. Anything that depends on row position needs an explicit sort.

Fixed by sorting inside `split_by_time` (stable `mergesort`, so loans sharing an
`issue_d` keep their relative order) and resetting the index on both halves. Both
parquet files were rebuilt.

The fix changed row order and nothing else: row counts, bad rates, date ranges,
and the slice-4 AUCs (0.7156 train / 0.6975 test) all came back identical.

### The same logic lived in two places

Notebook 3 did the split with its own inline copy of the masking code, so editing
`features.py` had no effect on the files it wrote. The duplicate is why the bug
survived the slice-3 verification cell — that cell called the function, the
file-writing cell did not.

The notebook now calls `split_by_time`. One version of the logic, in the file
slice 4 onward imports.

## Handover

`features_train.parquet` / `features_test.parquet`, both checked for: identical
column sets in identical order, no missing values, no `inf`, rows in `issue_d`
order.

`NON_FEATURES = ['target', 'issue_d', 'earliest_cr_line', 'zip_code', 'addr_state']`
— in the files, never in the model.

Bad rate: train 20.05%, test 23.65%.

## Known limitations

- Measured bad rates rise toward the end of the window because of one-sided
  censoring, not necessarily because risk rose. Cannot be corrected without
  payment dates, which are leakage.
- No borrower below FICO 660 exists — LC rejected them. The model can say nothing
  about that population, and range restriction weakens every screened variable.
- `annual_inc` reaches $6,000,000 and is self-reported, often unverified.
- `grade`, `sub_grade`, `int_rate` and `installment` are deliberately held out, so
  slice 6 can compare a model that uses LC's verdict against one that does not.