# Slice 4 — baseline logistic regression

Input: `features_train.parquet` (20,902 rows) and `features_test.parquet` (5,378 rows),
71 model features each.
Output: `models/baseline_logreg.joblib` and
`data/processed/predictions_test_baseline.parquet`.

**Test AUC 0.6973.** L1 logistic regression, `C=0.05`, 47 of 71 features surviving.

## What a baseline is for

Not to be good. To be the number every later model has to beat, and to force the
decisions that are easy to skip: how to scale, what to measure, where the
threshold goes.

Logistic regression first because it is fast, it converges, and every weight can
be read. If a gradient-boosted model later wins by 0.01, that is worth knowing;
if it wins by nothing, the simple model ships.

## Scaling, and why it is a learned step

Columns arrived on wildly different scales — `annual_inc` in hundreds of
thousands, `open_to_total_acc` between 0 and 1. Two problems follow: the solver
walks a long thin valley and may stop early, and the default L2 penalty judges
"large weight" on the raw number, so a column is punished for its units rather
than its usefulness.

`StandardScaler` fixes both. The mean and standard deviation it subtracts are
**numbers learned from a column**, so the slice 3 rule applies unchanged: fit on
train, apply to test.

The check that it was done right: scaled train has mean 0.0 and std 1.0 by
construction; scaled **test** came out at mean −0.0066, std 1.0813. Near those
values but not on them. Test landing exactly on 0 and 1 would have meant it was
scaled with its own numbers — the leak.

## Metric: AUC, not accuracy

20% of loans default, so predicting "everyone repays" scores 80% accuracy and is
worthless.

ROC-AUC asks instead: take one random defaulter and one random non-defaulter, how
often is the defaulter ranked riskier. 0.5 is guessing, 1.0 is perfect.

| | AUC |
|---|---|
| constant prediction (train bad rate for every loan) | 0.5000 |
| L2, all 71 features | 0.6975 |
| **L1, `C=0.05`, 47 features (chosen)** | **0.6973** |

The constant model is not a joke entry. It exercises the same scoring code with a
model that cannot discriminate, so a wrong answer there would expose a bug in the
measurement rather than in the model. It returned exactly 0.5.

Train 0.7156 against test 0.6975 — a gap of 0.018. Small, and the test set is a
*later time period*, not a random slice, so the model survived both unseen rows
and a shift in time.

## Collinearity: the weights are not an explanation

The first weight table read sensibly in places — `term` largest and positive
(60-month loans riskier), `dti` positive, `fico` negative — and nonsensically in
others. `credit_history_months` carried **+0.176**: longer credit history, more
risk.

Correlation and VIF explained it.

| pair | correlation |
|---|---|
| `credit_history_months` ↔ `mo_sin_old_rev_tl_op` | +0.920 |
| `total_il_high_credit_limit` ↔ `total_bal_ex_mort` | +0.869 |

| feature | VIF |
|---|---|
| `open_acc` | 471 |
| `num_sats` | 465 |
| `total_acc` | 357 |
| `num_rev_accts` | 171 |
| `tot_hi_cred_lim` | 92 |

When two columns carry nearly the same information, the model can place a large
positive weight on one and a compensating negative weight on the other and
predict identically either way. The *pair* points the right way; the individual
numbers do not mean what they appear to.

> With VIF in the hundreds the model still predicts well, but you have lost the
> right to read a coefficient as "the effect of this feature".

Pairwise correlation alone would have missed `tot_hi_cred_lim` — roughly the sum
of `total_rev_hi_lim` and `total_il_high_credit_limit`, and carrying a higher VIF
than either part. **Correlation checks pairs; VIF checks each feature against the
whole rest of the table.**

`open_acc` and `num_sats` being near-identical is range restriction again, the
same effect that killed the delinquency family in slice 3: among *approved*
borrowers, open accounts and accounts-in-good-standing are the same number.

## Choosing L1, and choosing it on training data only

Three options: keep everything and document that weights are unreadable; drop
columns by hand; or switch to an L1 penalty, which pushes redundant weights to
exactly zero.

Hand-dropping was rejected — without a data dictionary, choosing between
`open_acc` and `num_sats` would have been a coin flip.

**The comparison was made with `TimeSeriesSplit` on the training half only.**
Picking a model by test AUC would have folded the test set into the training
process and destroyed it as an honest estimate.

| | mean CV AUC | std |
|---|---|---|
| L2 | 0.7046 | 0.0168 |
| L1 (`C=1.0`) | 0.7060 | 0.0175 |

A difference of 0.0014 against a fold-to-fold spread of 0.0168. **Tied** — the
variation within either option dwarfs the gap between them.

### Sweeping the regularisation strength

`C=1.0` zeroed only 3 features, so `C` was swept. `C` is inverted: small `C` is a
*strong* penalty.

| C | mean AUC | std | features kept |
|---|---|---|---|
| 0.001 | 0.5396 | 0.0526 | 1 |
| 0.005 | 0.6577 | 0.0810 | 20 |
| 0.010 | 0.7000 | 0.0201 | 33 |
| **0.050** | **0.7087** | **0.0167** | **47** |
| 0.100 | 0.7089 | 0.0166 | 56 |
| 0.500 | 0.7074 | 0.0168 | 65 |
| 1.000 | 0.7060 | 0.0175 | 68 |
| 10.00 | 0.7026 | 0.0189 | 71 |

Between `C=10` and `C=0.1`, 15 features leave and the score *rises* — those were
carrying noise. Below `C=0.01` the penalty starts crushing real signal and the
fold-to-fold std widens, which is its own warning.

`C=0.05` and `C=0.1` are 0.0002 apart. **Chose `C=0.05`: the smallest model in the
flat, stable part of the curve.** Not the highest score — chasing a 0.0002 margin
is chasing noise, as the L2-vs-L1 comparison already showed.

### What L1 did and did not fix

| pair | outcome |
|---|---|
| `open_acc` / `num_sats` | **both zeroed** — the worst VIF pair, gone |
| `credit_history_months` / `mo_sin_old_rev_tl_op` | +0.1005 / −0.1368 — both survived, still opposing |
| `total_il_high_credit_limit` / `total_bal_ex_mort` | −0.0806 / +0.0912 — both survived |

L1 removes a feature that adds nothing *on its own*. Two strongly-correlated
columns that each carry a little unique signal can both survive — the penalty
shrinks them rather than choosing between them. An honest partial result: the
fully redundant features went, the near-duplicates did not.

Also dropped: `annual_inc` and `log_annual_inc`, while `loan_to_income` survived.
The ratio carried the signal and the raw figure added nothing on top — slice 3's
"an amount only means something next to the capacity to carry it", confirmed by
the model.

## The threshold is not a modelling decision

The model outputs a probability; a lending decision is yes or no. Something must
draw the line.

`.predict()` silently uses 0.5. At a 20% base rate that is close to useless: it
rejected **149 of 5,378** loans and caught **6.5%** of the defaults.

The two mistakes do not cost the same. Approving someone who defaults loses
principal; rejecting someone who would have repaid loses interest. In lending the
first is several times the second, so the threshold belongs well below 0.5.

| threshold | approved | bad rate of approved | precision | recall |
|---|---|---|---|---|
| 0.10 | 21.9% | 8.5% | 0.279 | 0.921 |
| 0.15 | 43.9% | 12.2% | 0.326 | 0.774 |
| 0.20 | 61.4% | 15.5% | 0.367 | 0.598 |
| 0.25 | 73.9% | 17.7% | 0.406 | 0.447 |
| 0.30 | 82.5% | 19.2% | 0.446 | 0.331 |
| 0.40 | 92.4% | 21.4% | 0.511 | 0.164 |
| 0.50 | 97.2% | 22.7% | 0.557 | 0.065 |
| — approve everyone — | 100% | 23.7% | — | — |

`bad rate of approved` is the column that matters and no sklearn function
produces it: of the loans you said yes to, what share went bad. **The metric that
matters most is often the one you have to write yourself.**

The curve has no kink. Risk trades against volume smoothly the whole way down, so
no threshold is "correct" — the number can only come from what a default costs
against what a lost customer costs. **This slice hands over the table, not a
chosen number.**

## Calibration: ranking is not the same as being right

AUC measures order only. Halve every probability and AUC is unchanged. So a model
can rank well and still output numbers that are false as probabilities.

Ten equal-sized bins, predicted against actual:

| bin | n | predicted | actual | gap |
|---|---|---|---|---|
| 0 | 538 | 0.0506 | 0.0502 | +0.0004 |
| 2 | 538 | 0.1070 | 0.1506 | −0.0435 |
| 4 | 538 | 0.1526 | 0.2007 | −0.0481 |
| 5 | 538 | 0.1799 | 0.2584 | −0.0785 |
| 7 | 538 | 0.2559 | 0.2993 | −0.0434 |
| 9 | 538 | 0.4721 | 0.4981 | −0.0261 |

Brier score 0.1671.

Every bin but the first sits **above** the diagonal: the model under-states risk
by roughly 3–5 points throughout. Not an S-curve, not over-confidence — a steady
lean in one direction.

The cause is already documented. Train bad rate 20.05%, test 23.65%; the model
learned its sense of typical risk from the earlier half and applied it to a
period that measures riskier. Slice 3 attributes that gap to **one-sided
censoring**, not to real drift — so the miscalibration is the known data artefact
appearing exactly where theory said it would.

Practical consequence: a threshold of 0.20 is not selecting borrowers with a
one-in-five chance. It is selecting borrowers with roughly a 26% chance. The
ranking is sound; the dial is mislabelled.

A maximum predicted probability of 1.000 looked alarming, but only 5 loans sit
above 0.9 and 20 above 0.7 — a thin tail of individual rows, not a population.
The highest *bin average* is 0.472. **The alarming summary statistic was not the
problem; the quiet 4-point bias across the middle was.**

## Three silent failures, caught by asserts

Each produced plausible output and no error.

### Rows were not in time order

`TimeSeriesSplit` cuts by **row position** and assumes earlier rows are earlier in
time. The parquet files were in arbitrary order — slice 3 split by boolean mask,
which selects the right rows but preserves whatever order they arrived in.

Folds built on arbitrary order would have trained on a mix of 2013–2016 and scored
on another mix, producing numbers that look fine and mean nothing.

Caught by `assert train['issue_d'].is_monotonic_increasing` before any fold ran.
Fixed in `features.py`, files rebuilt; full write-up in
`slice_03_feature_engineering.md`. The fix changed order and nothing else — AUC,
weights and VIFs all returned identical.

### A stale variable from six steps earlier

The save-verification asserted the reloaded AUC equalled `auc_test` — and failed,
0.6973 against 0.6975.

Nothing was wrong with the saved file. `auc_test` had been computed in step 4 from
the **L2** model and never overwritten when `model_final` replaced it. The assert
was comparing two different models' scores.

> A summary number computed earlier does not update when the thing it summarises
> is replaced.

The verification now compares the probability arrays directly (`np.allclose`,
max difference 0.0) — exact, and immune to a stale summary.

### A deprecation warning that hid what it was doing

`penalty="l1"` is deprecated in scikit-learn 1.8 in favour of `l1_ratio`, and
raised `Inconsistent values: penalty=l1 with l1_ratio=0.0` — where `0.0` means
pure L2. The warning stated the conflict without saying which side won, so the
model could have been silently running L2 while labelled L1.

Re-run with `l1_ratio=1.0`: byte-identical results, so `penalty` had been winning.
The earlier numbers were correct — but were not *knowably* correct until checked.

## Handover

`models/baseline_logreg.joblib` holds the model, the scaler, **and the column
order** in one bundle. The scaler works by position, so a frame with the same
columns in a different order would be silently mis-scaled. Also stored: `C`, the
test AUC, the training window, and the scikit-learn version that wrote the file.

Verified by loading the bundle back, re-scaling raw `X_test` with the loaded
scaler, and confirming the probabilities match exactly.

`data/processed/predictions_test_baseline.parquet` — `issue_d`, `zip_code`,
`addr_state`, `target`, `p_default` for all 5,378 test loans. The geography
columns are there so slice 7 can test whether errors fall unevenly across groups.

## Known limitations

- **Probabilities run 3–5 points low.** Recalibrate on a validation split in a
  later slice — never on the test set, which would spend it.
- **`credit_history_months` and two other pairs remain collinear.** Their
  individual weights cannot be interpreted. Resolving them needs a domain
  decision about which column to keep, which needs the Lending Club data
  dictionary.
- **`bad rate of approved` is measured on finished loans only** and is inflated by
  the one-sided censoring documented in slice 3. The relative movement between
  thresholds is sound; the absolute levels are pessimistic.
- **The test set has now been used.** Every further choice must be made on
  training folds or a validation split, or this AUC stops being an honest estimate.
- `grade`, `sub_grade`, `int_rate` and `installment` are still held out for the
  slice 6 comparison.