# Slice 5 — honest evaluation

Input: `features_train.parquet` (20,902 rows) and `features_test.parquet` (5,378 rows),
71 model features each, plus `models/baseline_logreg.joblib` and
`data/processed/predictions_test_baseline.parquet` from slice 4.

Output: `models/calibrated_logreg.joblib`,
`data/processed/predictions_test_calibrated.parquet`, and three figures.

**Test AUC 0.6908, 95% interval [0.6739, 0.7067]. Average precision 0.3918
against a no-skill floor of 0.2365 — a 1.66× lift. Probabilities corrected from
5.1 points low to 0.3 points high.**

## What this slice adds

Slice 4 asked whether the model ranks. It does. Slice 5 asks three harder
questions:

- **How good is it for a rare event?** AUC divides by every non-defaulter, and
  there are 4,106 of them. A precision-recall curve ignores the ones never
  flagged.
- **Can the numbers be read as probabilities?** Slice 4 found they run 3–5
  points low. This slice fixes that with a correction learned on data the model
  has never seen.
- **How much of any number is luck?** 5,378 rows is not many. Bootstrap
  intervals say which differences are real.

## A third split, carved out of train by time

Calibration must be fitted on data the model did not learn from. The last
stretch of the training period comes out and becomes a validation set.

Cut **by date, not by row position**. A row-position cut lands mid-month and
leaves the same calendar month partly in each set. Cutting on a named date is
also self-documenting: the notebook says `2016-01-01`, which a reader can check
against slice 3's windows.

| set | period | rows | bad rate |
|---|---|---|---|
| train_inner | 2013-01 → 2015-12 | 16,134 | **18.90%** |
| valid | 2016-01 → 2016-08 | 4,768 | **23.91%** |
| test | 2016-09 → 2017-12 | 5,378 | **23.65%** |

### A prediction made in advance, and wrong

Before splitting, the expectation was that validation would sit near the train
level, so a correction learned there would only half-transfer to test.

It sat at 23.91% — beside test, not beside train. The reason is slice 3's
one-sided censoring: the 2016-and-later vintages all measure riskier for the
same artefactual reason, so valid and test share a level that train_inner does
not. That made the correction transferable, and it is why the calibration below
works as well as it does.

Recorded because it was wrong. On data with genuine drift rather than a shared
artefact, validation would not have landed there and the correction would not
have carried.

## The refit, and what it cost

Everything was refitted on `train_inner` alone: a **new** `StandardScaler` (the
slice 4 scaler learned its means from rows that are now held out — reusing it
would be a quiet leak) and a new model with `C=0.05` unchanged.

`C` was deliberately **not** re-tuned. It was chosen in slice 4 on training
folds. Re-tuning now would spend `valid` twice — once to pick the model, once to
fit the calibration — and the correction would stop being honest.

| | AUC |
|---|---|
| train_inner (in-sample) | 0.7176 |
| valid | 0.7019 |
| test | **0.6908** |

### The cost is real, and smaller than the benefit

Slice 4's model scored 0.6973 on the same test rows. The difference looked like
noise: slice 4's fold-to-fold std was 0.0167, and 0.0065 is less than half of
that.

A **paired** bootstrap says otherwise.

| | observed | 95% interval | contains zero |
|---|---|---|---|
| slice 5 AUC − slice 4 AUC, same rows | −0.0066 | [−0.0094, −0.0038] | **no** |

Pairing is why. Most of the variation in an AUC comes from which loans were
drawn; score both models on identical rows and that luck cancels in the
subtraction. The paired interval is 0.0056 wide against 0.0328 for the AUC
itself — six times narrower.

> Comparing a *difference* against the spread of a *level* is a scale error. If
> two models can be scored on the same rows, pair them.

The honest limit: the bootstrap resamples the test set, not the training set. It
proves the difference survives test-sample luck. It does not prove a different
training draw would reproduce it.

**And the trade was worth it.** 0.0066 of AUC bought a correction that closed a
5.1-point error in the probabilities. Ranking got marginally worse; the numbers
became usable.

### Feature selection: stable core, arbitrary edge

L1 kept 47 of 71 — the same count as slice 4. The count is a coincidence.

**Only 39 are common to both.** Eight left, eight arrived. The arrivals include
`application_type_Joint App`, `chargeoff_within_12_mths`, `emp_length` and
`mo_sin_old_il_acct` — all of which appear on slice 4's *dropped* list. Slice 4
recorded `application_type_Joint App` at a weight of +0.000037.

So these are not eight substantive changes. They are features sitting on the
zero boundary, and 23% less training data was enough to flip which side they
land on.

> Slice 4 established that an individual coefficient cannot be read. This goes
> further: *whether a feature was selected at all* cannot be read either.

### Feature distributions widen with time

Scaled against `train_inner`'s own means and standard deviations:

| set | mean | std |
|---|---|---|
| train_inner | −0.0000 | 1.0000 |
| valid | −0.0002 | 1.0918 |
| test | +0.0116 | 1.1678 |

Mean 0 and std 1 on train is arithmetic, not evidence. The other two landing
*near* but not *on* those values is the evidence — exactly on them would mean
each had been scaled with its own numbers.

The steady widening across time is a mild distribution shift, monotone in the
direction of travel. Not corrected here. It is the first concrete reason slice
10 exists.

## Precision-recall: the less generous mirror

ROC divides false alarms by every good loan — about 4,106 rows. A large number
of false alarms still produces a small false-positive rate. Precision divides by
the loans actually flagged, and no quiet approval appears in it anywhere.

| | average precision | no-skill floor | lift |
|---|---|---|---|
| valid | 0.4151 | 0.2391 | 1.74× |
| **test** | **0.3918** | **0.2365** | **1.66×** |

> Average precision has no 0.5 floor. Its no-skill value is the base rate.
> Quoting AP without the floor beside it says almost nothing.

What the curve costs at each operating point:

| catch this share of defaults | …and this share of what you flag is actually bad | lift |
|---|---|---|
| 10% | 53.1% | 2.25× |
| 20% | 50.0% | 2.11× |
| 30% | 45.3% | 1.92× |
| 50% | 37.5% | 1.58× |
| 75% | 32.9% | 1.39× |

Still above chance at 75% recall. "Above chance" and "good enough to lend on"
are different claims, and the PR panel is what stops a reader confusing them.

The far-left end of a PR curve swings between 0 and 1 as single rows enter and
leave the flagged set. That region is noise; the figure clips it.

## Calibration

### Two candidates, fitted on validation

- **Sigmoid (Platt)** — a one-input logistic regression on the model's raw
  log-odds. Two parameters: a slope and a shift. Fitted on `decision_function`
  rather than on the probabilities, which is both the correct form and avoids
  taking the log-odds of a probability that equals exactly 1.000.
- **Isotonic** — free-form, monotone, bends however the data says.

Theory said the sigmoid should fit: slice 4 documented a steady lean in one
direction, not an S-curve, and a shift is what a sigmoid does.

**Platt learned a slope of +0.9475 and a shift of +0.2898.** A slope near 1 means
the spread was already right. Almost all the correction went into the shift,
which is what was predicted in advance.

### The decision

| | Brier | log loss | AUC | mean predicted | lean vs 0.2365 |
|---|---|---|---|---|---|
| uncalibrated | 0.1698 | 0.5232 | 0.6908 | 0.1855 | **−0.0511** |
| **sigmoid (chosen)** | 0.1668 | **0.5131** | **0.6908** | 0.2394 | **+0.0029** |
| isotonic | **0.1664** | 0.5250 | 0.6897 | 0.2381 | +0.0016 |

The two metrics disagree, and the disagreement is the answer.

Brier calls it a tie — 0.0004 apart. Log loss puts isotonic at 0.5250, **worse
than not calibrating at all**. The cause is in isotonic's own output: it produced
6 predictions of exactly 1.0 and 8 of exactly 0.0. Log loss punishes
confident-and-wrong close to without bound; Brier squares the miss and shrugs.

Isotonic also cost 0.0011 of AUC by creating ties, and collapsed 5,378 distinct
scores into **75**.

The sigmoid keeps every distinct score, leaves the ranking identical to the
decimal, wins log loss outright, and does it with two parameters.

> A monotone map cannot reorder anything, so AUC after a sigmoid must equal AUC
> before, exactly. That is an assert, not an expectation. Isotonic is different —
> it creates ties, which can lower AUC, so that one is measured rather than
> asserted.

### What it fixed, and what it left

Ten equal-sized bins, test set:

| bin | predicted | actual | gap |
|---|---|---|---|
| 0 | 0.0720 | 0.0539 | +0.0181 |
| 1 | 0.1123 | 0.1004 | +0.0120 |
| 2 | 0.1407 | 0.1654 | −0.0247 |
| 3 | 0.1660 | 0.1601 | +0.0058 |
| 4 | 0.1927 | 0.2063 | −0.0136 |
| 5 | 0.2234 | 0.2398 | −0.0164 |
| 6 | 0.2598 | 0.2998 | −0.0401 |
| 7 | 0.3078 | 0.3011 | +0.0067 |
| 8 | 0.3798 | 0.3457 | +0.0341 |
| 9 | 0.5399 | 0.4926 | +0.0473 |

**The signs are the finding.** Uncalibrated, all ten bins were negative, running
from −0.006 to −0.104 — a bias. Now four are negative and six positive — scatter.
Turning a bias into scatter is the whole job. Mean absolute gap fell to 0.0219.

What is left is a gentle shape, not a lean: slightly pessimistic at both ends,
slightly optimistic through the middle. Bin 9 over-states by 4.7 points —
predicting 54% where the truth is 49%. **That is the bin a lender acts on**, so it
is the weakest point of the correction and it sits where decisions get made. A
two-parameter sigmoid slides and stretches but cannot bend; isotonic could have
fixed this, at the cost documented above.

The two extreme rows below are not the cause — 2 rows in 538 move that bin's
average by about 0.001.

### Calibration is a horizontal shift

Because the sigmoid never reorders, sorting by raw or by calibrated probability
produces the identical ten bins — the same borrowers in each. The `actual`
column is unchanged between the before and after tables. Only the predictions
move. The figure draws one connector per bin to show this.

## One row, total certainty, completely wrong

| | |
|---|---|
| row | 3240, issued 2017-05-01 |
| score | +33.368 (validation's highest was +3.189) |
| probability | 1.000000000000 |
| what happened | **the loan was repaid** |

One feature caused it:

```
loan_to_income    raw value 19.500    training maximum 1.782    contribution +34.51
```

A loan for 19.5 times stated annual income — roughly eleven times beyond
anything in training. Total score +33.368; that feature contributed +34.508.
Remove it and the score is about −1.14, a probability of 0.24, right on the base
rate. Every other field on the row is unremarkable: 60-month term, FICO 732, two
accounts opened in 24 months.

A ratio of 19.5 is not a borrower who asked for 19.5× their income. It is a
denominator that is wrong — a monthly figure entered as annual, or a typo. Slice
3 flagged `annual_inc` as self-reported and often unverified, and noted values up
to $6,000,000. This is the same limitation from the other end: not a number too
large, but one too small.

`loan_to_income` was slice 3's strongest engineered feature at 2.4×. Its power
and its fragility are the same property — a ratio concentrates signal into one
number, and a ratio with a bad denominator explodes.

Slice 3 had already established the remedy, for a different column:
`credit_history_months` was capped at 600 on domain judgement. The technique was
justified and then applied to only one of the two features that needed it.

### Each metric saw something different

The guard below caps `loan_to_income` at the training maximum. Nothing is
refitted — same scaler, same model, same calibrator, only the input clipped.

| | Brier | log loss | AUC |
|---|---|---|---|
| **as measured (reported)** | 0.1668 | 0.5131 | 0.6908 |
| extreme rows dropped (n=5,376) | 0.1667 | 0.5074 | 0.6907 |
| input guard applied (n=5,378) | 0.1668 | **0.5076** | 0.6908 |

> AUC and Brier saw nothing. Log loss lost 0.0055. **That single row is 1.2% of
> the total log loss, from 0.019% of the rows.** Two of three evaluation metrics
> were blind to a prediction that was certain and wrong.

The guard reaches the same log loss as discarding the rows, while keeping every
row. Maximum predicted probability falls from 1.000000 to **0.982307**.

**This was found in the test set, so nothing in the pipeline was changed.** Slice
4's limitations state that every further choice must be made on training folds or
a validation split. The table above is a labelled sensitivity check, not a
result. The reported numbers are the first row. The cap is recorded in the model
bundle under a key that says it is not applied, and slice 9 applies it as an
input guard.

Rare — 2 rows in 5,378 — and rarity is worth stating. It does not soften the
finding: the failure is **unbounded** (nothing stops the ratio being 50), and the
test set is one fixed sample while production sees new applications forever.

## Confidence intervals

2,000 bootstrap resamples of the test set, percentile method, seeded.

| metric | estimate | 95% interval | width |
|---|---|---|---|
| AUC | 0.6908 | [0.6739, 0.7067] | 0.0328 |
| average precision | 0.3918 | [0.3664, 0.4198] | 0.0534 |
| Brier | 0.1668 | [0.1606, 0.1728] | 0.0122 |
| log loss | 0.5131 | [0.4956, 0.5318] | 0.0362 |
| approved @ 0.20 | 0.4788 | [0.4658, 0.4913] | 0.0255 |
| bad rate of approved @ 0.20 | 0.1305 | [0.1174, 0.1437] | 0.0263 |

**AUC's third decimal place is meaningless.** That independently confirms slice
4's judgement calls: `C=0.05` vs `C=0.1` differed by 0.0002 and L1 vs L2 by
0.0014, both a fraction of this width. Slice 4 called them tied using
fold-to-fold variation; the bootstrap agrees by a different route.

Average precision has the widest interval of all. It leans on the top of the
ranking, where there are fewest rows, so it is inherently noisier than AUC.

Valid AUC 0.7019 [0.6850, 0.7181] against test 0.6908 [0.6739, 0.7067]: the
intervals overlap. The 0.0111 gap **cannot be distinguished** from noise — which
is not the same as there being no difference. Comparing two separate intervals is
a weak test, and unlike the model-vs-model comparison these are different loans
from different periods, so pairing is not available.

> What a bootstrap measures is sampling luck and nothing else. It says nothing
> about a different time period, a different lender, or the censoring bias in
> slice 3. A tight interval means stable, not correct.

## The approval table that means what it says

Calibrated probabilities, so a threshold of 0.20 selects borrowers with roughly a
20% chance — unlike slice 4, where 0.20 meant about 26%.

| threshold | approved | predicted bad | actual bad | 95% interval | recall | good loans rejected |
|---|---|---|---|---|---|---|
| 0.05 | 1.4% | 0.0388 | 0.0395 | [0.0000, 0.0909] | 99.8% | 4,033 (98.2%) |
| 0.10 | 11.0% | 0.0745 | 0.0556 | [0.0378, 0.0751] | 97.4% | 3,545 (86.3%) |
| 0.15 | 28.4% | 0.1060 | 0.1060 | [0.0908, 0.1216] | 87.3% | 2,739 (66.7%) |
| **0.20** | **47.9%** | **0.1338** | **0.1305** | **[0.1174, 0.1440]** | **73.6%** | **1,867 (45.5%)** |
| 0.25 | 62.7% | 0.1552 | 0.1620 | [0.1493, 0.1741] | 57.1% | 1,282 (31.2%) |
| 0.30 | 73.9% | 0.1732 | 0.1815 | [0.1695, 0.1930] | 43.3% | 855 (20.8%) |
| 0.35 | 81.5% | 0.1873 | 0.1934 | [0.1816, 0.2050] | 33.3% | 569 (13.9%) |
| 0.40 | 87.2% | 0.1995 | 0.2021 | [0.1905, 0.2133] | 25.5% | 363 (8.8%) |
| 0.50 | 94.5% | 0.2182 | 0.2192 | [0.2078, 0.2307] | 12.4% | 138 (3.4%) |
| — approve everyone — | 100% | — | 0.2365 | — | — | 0 |

`predicted bad` is the average probability the model assigned to the loans it
approved. `actual bad` is what those loans did. The two agreeing is calibration
stated in a lender's own language, and it is checkable by anyone in five seconds.

### Before and after, same thresholds

| threshold | gap, slice 4 | gap, slice 5 |
|---|---|---|
| 0.05 | +0.0106 | −0.0007 |
| 0.10 | −0.0156 | +0.0190 |
| 0.15 | −0.0245 | 0.0000 |
| 0.20 | −0.0358 | +0.0033 |
| 0.25 | −0.0402 | −0.0068 |
| 0.30 | −0.0412 | −0.0084 |
| 0.35 | −0.0417 | −0.0060 |
| 0.40 | −0.0425 | −0.0026 |
| 0.50 | −0.0426 | −0.0010 |

**Mean absolute gap: 0.0327 → 0.0053.** Six times better.

The shape matters more than the size. Slice 4's gap grows steadily more negative
as more loans are approved, settling near −0.043 — wrong by a *different amount
at every setting*, so no single correction factor would have fixed it. Slice 5's
is flat near zero across the whole range. Flat is what makes a dial usable.

The one blemish is the 0.10 row at +0.019, where the model over-states risk by
more than sampling explains. It matches bins 0 and 1 above: slightly pessimistic
at the safe end.

### No kink, and now a slope

Approved share against bad rate is close to a straight line from (0.11, 0.056) to
(0.95, 0.219) — slope about 0.196. **Every extra 10 points of approvals costs
roughly 2 points of bad rate, the whole way across.**

Slice 4 observed there was no kink. With calibrated numbers that becomes
quantitative: the trade has constant marginal cost. There is no volume at which
risk suddenly gets cheap, so no operating point can be read off the data. It can
only come from what a default costs against what a lost customer earns.

At the 0.20 row the model is worth 10.6 percentage points of bad rate against
approving everyone — while turning away 45.5% of all good borrowers. Slice 8
inherits a clean linear trade and one number to find.

## Figures

- `figures/slice05_pr_and_roc.png` — the same model in two mirrors. ROC looks
  respectable; PR never exceeds 0.55.
- `figures/slice05_calibration.png` — ten bins before and after, with a connector
  per bin showing the same borrowers sliding onto the diagonal.
- `figures/slice05_risk_vs_volume.png` — the approval trade with its bootstrap
  band. The band is a wide wedge at the left, where few loans are approved.

## Handover

`models/calibrated_logreg.joblib` holds **three** fitted objects that only work in
one order — scaler, model, calibrator — plus an `apply_order` key and the column
list. A `predict_calibrated(bundle, X_raw)` function defines that order in code
rather than in prose; slice 9 deploys it unchanged.

Also stored: `C`, the penalty, the split date, the surviving feature count, test
AUC with its interval, Brier, log loss, both data windows, the scikit-learn
version, and `loan_to_income_guard_not_applied: 1.782` — named so nobody can load
the file and assume the guard is already in the pipeline.

Verified by loading the bundle cold, rebuilding from **raw** `X_test`, and
comparing the probability **arrays**: maximum difference 0.00e+00. Slice 4's
stale-variable failure is the reason this compares arrays rather than a summary
number — two different models can share an AUC to four decimals, but not 5,378
probabilities.

`data/processed/predictions_test_calibrated.parquet` — `issue_d`, `zip_code`,
`addr_state`, `target`, `p_default_raw`, `p_default_calibrated` for all 5,378
test loans. Both probability columns are kept so slice 7 can test fairness on the
number a lender would use, and also ask whether calibration changed the picture.

Slice 4's bundle and predictions are untouched, so slice 6 can compare against
both.

## Known limitations

- **The test set has now been used twice.** Slice 6 must make every choice on
  training folds or the validation split.
- **The correction transferred because valid and test share a censoring
  artefact**, not because the method is robust to drift. On genuinely drifting
  data it would need refitting over time.
- **Bin 9 still over-states risk by 4.7 points**, and the 0.10 threshold by 1.9.
  The residual is a gentle curve — pessimistic at both ends, optimistic in the
  middle — which a two-parameter sigmoid cannot remove.
- **`loan_to_income` is unbounded and unguarded.** Two test rows extrapolated past
  anything in training; one produced a probability of 1.000 on a loan that
  repaid. Documented and measured here, fixed in slice 9.
- **Feature distributions widen steadily across time** (scaled std 1.00 → 1.09 →
  1.17). Not corrected. Slice 10's problem.
- **Which features L1 keeps is unstable at the margin** — 39 of 47 common to slice
  4's selection. Selection is not a finding about a feature.
- **Everything is still measured on finished loans only**, and inherits slice 3's
  one-sided censoring. Relative movement between thresholds is sound; absolute
  levels are pessimistic.
- `grade`, `sub_grade`, `int_rate` and `installment` remain held out for the slice
  6 comparison.
