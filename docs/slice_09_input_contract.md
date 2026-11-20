# Slice 9 — the input contract

Written before `src/contract.py`. That file is this document as code, and
`tests/test_contract.py` is this document as tests: one decision, one test.

## What the service takes

One loan application: a flat JSON object with **57 fields**, plus one optional
field. The names are the column names of the table the model was built from
(`model_table.parquet` plus the two dates). There is no translation layer
between "API names" and "model names" on purpose — a second vocabulary is a
second place to be wrong.

Numbers are numbers, not text. `term` is `36` or `60`, not `" 60 months"`;
`emp_length` is years as a number with 10 meaning ten or more. That parsing was
done in notebook 02 and is outside this contract.

## The question, four times

The slice plan asked what happens on a missing field, an out-of-range value, an
unseen category and a negative income, and for each one: reject, impute, or
score with a flag. The answers follow from one principle.

> **The service may do to an application only what training did to a training
> row. Anything else is refused, and nothing is ever done silently.**

A model is a function fitted to data that went through a specific treatment. If
the service treats an input in a way training never did, the model is being
asked about a kind of row it has never seen, and it will answer anyway.

### 1. A missing field

**Impute and flag — but only where training imputed.** Ten columns had blanks
in the training rows, and slice 3 filled them. The service fills the same ten,
with the same values, and lists them under `flags.imputed`.

| field | blank in training | filled with |
|---|---|---|
| `emp_length` | 5.60% | the train median, 7 |
| `dti` | 1 row of 20,902 | the train median, 17.93 |
| `revol_util` | 0.04% | the train median, 54 |
| `bc_open_to_buy` | 0.90% | the train median, 4209 |
| `bc_util` | 0.94% | the train median, 66.1 |
| `percent_bc_gt_75` | 0.93% | the train median, 50 |
| `mo_sin_old_il_acct` | 3.12% | the train median, 129 |
| `mths_since_recent_inq` | 9.76% | 36, the "longest ago" end, plus the flag feature `mths_since_recent_inq_missing` = 1 |
| `num_tl_120dpd_2m` | 3.72% | the train median, 0 |
| `pct_tl_nvr_dlq` | 0.01% | the train median, 97.7 |

**Reject everywhere else.** The other 47 fields were never blank in training.
A blank income or a blank credit score has no honest fill: any number put there
would be invented by the service, and the model would treat it as a fact.

`dti`, `revol_util` and `pct_tl_nvr_dlq` are imputable on almost no evidence —
a handful of training rows. They are kept imputable because the rule is "do
what training did", and a rule with exceptions chosen by taste is not a rule.
It is written here so nobody mistakes it for a well-supported fill.

An **unknown field name is rejected**, not ignored. Otherwise `emp_lenght: 7`
is dropped, `emp_length` is quietly imputed, and the caller never learns that
the value they sent was not used.

### 2. An out-of-range value

Two different things hide in "out of range", and they get different answers.

**Impossible → reject.** A negative balance, a credit score of 250 or 900, a
percentage above 100, a term of 48 months, a date of first credit that is after
the application date, more accounts open than ever held. These are not unusual
applicants; they are errors, and scoring an error produces a confident number
about nothing. The hard limits in the table below are facts about the world
and were not read from the data.

**Possible but outside what the model saw → score and flag.** A credit score of
620 is a real applicant. Lending Club simply never lent below 660, so the model
has no training rows there and is extrapolating in a straight line. The
application is scored, each such field is listed under
`flags.outside_training_range` with the range training saw, and
`review_recommended` is set to `true`.

Why not reject these too? Because the range is an accident of one lender's
policy, not a boundary of the world, and a service that refuses every applicant
who differs from the training book refuses exactly the people a lender most
needs a second look at. Why not score them silently? Because that is how
slice 5's row happened.

"What the model saw" is measured on the 16,134 rows the weights were fitted on
(before 2016-01), read from `models/feature_spec.json`.

**One field is clipped, not just flagged: `loan_to_income`.** Slice 5 found a
test loan at 19.5 times income, against a training maximum of 1.782, scored
1.000000 — on a loan that repaid. The guard clips the ratio to the training
maximum before scaling and sets `flags.guard_applied`. It is the only clip
because it is the only one that was measured (slice 7, and `src/guard_effect.py`
in this slice). Clipping the other 56 fields because it "seems safer" would be
a modelling change made in the service, tested on nothing.

### 3. An unseen category

**Reject.** This closes the one silent path in the whole build. pandas encodes
a level it has never seen as "no dummy column set", which is exactly how the
base level is encoded. So `purpose: "educational"` would be scored as `car`,
and `home_ownership: "NONE"` as `MORTGAGE` — the base level of each.
No error, no warning, a slightly-too-good probability.

**Five rare purposes and one home-ownership level are folded and flagged**,
because training folded them: `wedding`, `renewable_energy`, `house`, `moving`
and `vacation` are scored as `other`, and `ANY` as `MORTGAGE`. They are listed
under `flags.folded`. The allowed lists are built from the levels in
`models/feature_spec.json` and the fold table in `src/columns.py`, so they
cannot drift from the build.

### 4. A negative income

**Reject.** So is zero. `loan_to_income` is the model's third-heaviest feature
and it is `loan_amnt / annual_inc`. A negative income makes it negative, which
the model reads as an exceptionally safe borrower; zero makes it infinite.
Imputing the median instead would turn a broken application into a typical
earner. There is no version of this that is honest except saying no.

A small positive income is a different case: `annual_inc: 2000` is possible, is
below what training saw, and is scored, flagged, and guarded.

## What a rejection looks like

HTTP 422, and **every** problem at once, so an application is fixed in one
round and not one field per attempt:

```json
{"rejected": [
  {"field": "term",       "problem": "must be one of [36, 60], got 48"},
  {"field": "annual_inc", "problem": "must be at least 1, got -40000"}
]}
```

## What a flag looks like

```json
"flags": {
  "imputed": ["emp_length"],
  "outside_training_range": [{"field": "annual_inc", "value": 2000.0,
                              "train_min": 3000.0, "train_max": 6000000.0}],
  "folded": [{"field": "purpose", "value": "wedding", "scored_as": "other"}],
  "guard_applied": true,
  "credit_history_capped": false
},
"review_recommended": true
```

`review_recommended` is true when a value was outside the training range or the
guard fired. An imputed blank or a folded category does not set it: the model
was trained on rows treated exactly that way.

## The optional field: the price

`int_rate` is not part of the application and never reaches the model. Slice 3
held the price out of the features so the model would judge the applicant and
not copy Lending Club's grading; `src/decide.py` asserts that it is still not a
model column. If a caller sends it, the response adds the loan's own break-even
probability from slice 8's economics — as information. The rule in force is the
global threshold. `tests/test_contract.py` checks that changing the price from
6% to 28% does not move the probability or the decision.

## What the contract did to real applications

All 5,378 test applications were sent through it (`python -m src.service_report`).
This is reporting; nothing was changed because of it.

| | applications | share |
|---|---|---|
| rejected | 0 | 0.00% |
| a blank was imputed | 1,207 | 22.44% |
| a rare category was folded | 139 | 2.58% |
| a value outside the training range | 156 | 2.90% |
| `loan_to_income` guard applied | 2 | 0.04% |
| review recommended | 157 | 2.92% |

Zero rejections matters: a contract that turned away real applications would be
a contract for some other dataset.

The largest group outside the training range is not noise. **117 applications
ask for more than 35,000**, the largest loan in the training rows. In the test
window loans go up to 40,000: the lender's maximum moved after the training
window closed. The contract did not know that; it only knew the model had never
seen such a loan, and said so. That is a product change arriving as data drift, and it is the
first thing slice 10's monitoring should count.

## Every field

Generated from `src/contract.py` (`GET /contract` returns the same table).
"training saw" is the range in the 16,134 rows the model was fitted on; a
value outside it is scored and flagged.

| field | meaning | accepts | if blank | training saw |
|---|---|---|---|---|
| `loan_amnt` | Loan amount requested | number 1 .. no limit | reject | 1,000 to 35,000 |
| `term` | Term in months | 36 or 60 | reject |  |
| `purpose` | Purpose of the loan | car, credit_card, debt_consolidation, home_improvement, major_purchase, medical, other, small_business, wedding, renewable_energy, house, moving, vacation | reject |  |
| `application_type` | Individual or joint application | Individual, Joint App | reject |  |
| `issue_d` | Application date | date, YYYY-MM-DD | reject |  |
| `annual_inc` | Annual income | number 1 .. no limit | reject | 3,000 to 6e+06 |
| `emp_length` | Years in current job (10 = ten or more) | whole number 0 .. 10 | **impute** | 0 to 10 |
| `home_ownership` | Home ownership | MORTGAGE, OWN, RENT, ANY | reject |  |
| `verification_status` | Whether the income was verified | Not Verified, Source Verified, Verified | reject |  |
| `dti` | Debt payments as % of income | number 0 .. no limit | **impute** | 0 to 64.99 |
| `fico_range_low` | Credit score, low end of range | whole number 300 .. 850 | reject | 660 to 840 |
| `fico_range_high` | Credit score, high end of range | whole number 300 .. 850 | reject | 664 to 844 |
| `earliest_cr_line` | Date of the oldest credit account | date, YYYY-MM-DD | reject |  |
| `open_acc` | Open credit accounts | whole number 0 .. no limit | reject | 1 to 70 |
| `total_acc` | Credit accounts ever held | whole number 1 .. no limit | reject | 3 to 130 |
| `acc_open_past_24mths` | Accounts opened in the last 24 months | whole number 0 .. no limit | reject | 0 to 38 |
| `mort_acc` | Mortgage accounts | whole number 0 .. no limit | reject | 0 to 27 |
| `num_actv_bc_tl` | Active bankcard accounts | whole number 0 .. no limit | reject | 0 to 21 |
| `num_actv_rev_tl` | Active revolving accounts | whole number 0 .. no limit | reject | 0 to 31 |
| `num_bc_sats` | Bankcard accounts in good standing | whole number 0 .. no limit | reject | 0 to 32 |
| `num_bc_tl` | Bankcard accounts ever held | whole number 0 .. no limit | reject | 0 to 47 |
| `num_il_tl` | Instalment accounts ever held | whole number 0 .. no limit | reject | 0 to 101 |
| `num_op_rev_tl` | Open revolving accounts | whole number 0 .. no limit | reject | 0 to 67 |
| `num_rev_accts` | Revolving accounts ever held | whole number 0 .. no limit | reject | 2 to 86 |
| `num_rev_tl_bal_gt_0` | Revolving accounts with a balance | whole number 0 .. no limit | reject | 0 to 31 |
| `num_sats` | Accounts in good standing | whole number 0 .. no limit | reject | 1 to 70 |
| `num_tl_op_past_12m` | Accounts opened in the last 12 months | whole number 0 .. no limit | reject | 0 to 24 |
| `revol_bal` | Revolving balance | number 0 .. no limit | reject | 0 to 678,330 |
| `revol_util` | Revolving credit used, % | number 0 .. no limit | **impute** | 0 to 146.1 |
| `tot_cur_bal` | Total current balance, all accounts | number 0 .. no limit | reject | 75 to 2.47171e+06 |
| `avg_cur_bal` | Average balance per account | number 0 .. no limit | reject | 13 to 447,433 |
| `total_rev_hi_lim` | Total revolving credit limit | number 0 .. no limit | reject | 0 to 730,500 |
| `tot_hi_cred_lim` | Total credit limit, all accounts | number 0 .. no limit | reject | 2,700 to 2.64169e+06 |
| `total_bal_ex_mort` | Total balance excluding mortgage | number 0 .. no limit | reject | 0 to 1.13274e+06 |
| `total_bc_limit` | Total bankcard limit | number 0 .. no limit | reject | 0 to 246,400 |
| `total_il_high_credit_limit` | Total instalment credit limit | number 0 .. no limit | reject | 0 to 1.1192e+06 |
| `bc_open_to_buy` | Unused bankcard credit | number 0 .. no limit | **impute** | 0 to 226,942 |
| `bc_util` | Bankcard credit used, % | number 0 .. no limit | **impute** | 0 to 153.9 |
| `percent_bc_gt_75` | % of bankcards above 75% of limit | number 0 .. 100 | **impute** | 0 to 100 |
| `mo_sin_old_il_acct` | Months since oldest instalment account opened | whole number 0 .. no limit | **impute** | 1 to 439 |
| `mo_sin_old_rev_tl_op` | Months since oldest revolving account opened | whole number 0 .. no limit | reject | 13 to 683 |
| `mo_sin_rcnt_rev_tl_op` | Months since newest revolving account opened | whole number 0 .. no limit | reject | 0 to 330 |
| `mo_sin_rcnt_tl` | Months since newest account opened | whole number 0 .. no limit | reject | 0 to 151 |
| `mths_since_recent_inq` | Months since the last credit inquiry (blank = none on record) | whole number 0 .. no limit | **impute** | 0 to 24 |
| `inq_last_6mths` | Credit inquiries in the last 6 months | whole number 0 .. no limit | reject | 0 to 6 |
| `delinq_2yrs` | Delinquencies in the last 2 years | whole number 0 .. no limit | reject | 0 to 19 |
| `acc_now_delinq` | Accounts delinquent right now | whole number 0 .. no limit | reject | 0 to 3 |
| `delinq_amnt` | Amount currently past due | number 0 .. no limit | reject | 0 to 65,000 |
| `num_accts_ever_120_pd` | Accounts ever 120+ days past due | whole number 0 .. no limit | reject | 0 to 29 |
| `num_tl_120dpd_2m` | Accounts 120 days past due, last 2 months | whole number 0 .. no limit | **impute** | 0 to 1 |
| `num_tl_30dpd` | Accounts 30 days past due right now | whole number 0 .. no limit | reject | 0 to 3 |
| `num_tl_90g_dpd_24m` | Accounts 90+ days past due, last 24 months | whole number 0 .. no limit | reject | 0 to 18 |
| `pct_tl_nvr_dlq` | % of accounts never delinquent | number 0 .. 100 | **impute** | 25 to 100 |
| `collections_12_mths_ex_med` | Collections in the last 12 months | whole number 0 .. no limit | reject | 0 to 3 |
| `chargeoff_within_12_mths` | Charge-offs in the last 12 months | whole number 0 .. no limit | reject | 0 to 2 |
| `tot_coll_amt` | Total ever owed to collections | number 0 .. no limit | reject | 0 to 63,730 |
| `pub_rec` | Public records | whole number 0 .. no limit | reject | 0 to 16 |
| `pub_rec_bankruptcies` | Bankruptcies on public record | whole number 0 .. no limit | reject | 0 to 5 |
| `tax_liens` | Tax liens | whole number 0 .. no limit | reject | 0 to 16 |
| `int_rate` | Interest rate offered, % per year (optional) | number 0 .. 100 | left out |  |

## Known gaps

- **24 of the 71 model features have a weight of exactly zero** (the L1
  penalty removed them), so 17 of these fields cannot change any answer. They
  are still required, because the saved scaler works by position on all 71
  columns and the contract mirrors the artifact as it was trained. Dropping
  them is a retraining decision, not a service one.
- **Dates are used to the month.** Lending Club publishes months, so the credit
  history is whole months between two dates, exactly as in slice 3.
- **The service does not check that an application is recent.** The model was
  trained on applications from 2013 to 2015. Scoring one dated today is
  allowed, and nothing in the response can say how much the world has moved
  since. That is slice 10.
