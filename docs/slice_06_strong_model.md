# Slice 6 — strong model

Input: `features_train.parquet` (20,902 rows) and `features_test.parquet`
(5,378 rows), 71 model features each, plus `models/calibrated_logreg.joblib`
and `data/processed/predictions_test_calibrated.parquet` from slice 5.

Output: `models/calibrated_lgbm.joblib`,
`data/processed/predictions_test_slice6.parquet`, and three figures.

Library: LightGBM 4.7.0.

**Gradient boosting was fitted at defaults, tuned over 44 configurations with
a pre-declared search, extended once when it hit a grid edge, and calibrated
the same way as slice 5 — and it lost. CV 0.7053 against logistic's 0.7107;
test 0.6726 against 0.6908, a paired difference of −0.0182 with a 95% interval
of [−0.0257, −0.0105] that excludes zero. The slice 5 logistic model ships.**

---

## What this slice adds

Slice 5 produced an honest, calibrated baseline. Slice 6 asks the obvious next
question — would a more flexible model do better? — and answers no, with the
work to back it up.

Three things make the answer worth more than the answer itself:

- **The search was declared before it ran** and committed to git, so "it lost"
  cannot be confused with "I stopped looking".
- **The comparison is paired**, so the difference is measured against the
  spread of the *difference* rather than the spread of either level.
- **Lending Club's own verdict was brought back in** at the end, which turned
  slice 3's argument about range restriction into a measured quantity.

---

## The ground, reproduced

Every choice in this slice is made on `TimeSeriesSplit` folds inside
`train_inner`. `valid` is reserved for calibration; `test` is reported, never
selected on.

| set | period | rows | bad rate |
|---|---|---|---|
| train_inner | 2013-01 → 2015-12 | 16,134 | 18.90% |
| valid | 2016-01 → 2016-08 | 4,768 | 23.91% |
| test | 2016-09 → 2017-12 | 5,378 | 23.65% |

Rebuilt from the parquet files and asserted against slice 5's recorded counts
and bad rates before anything was fitted. **When a new result has to be
compared against an old one, the first task is proving the ground is
identical** — a four-row difference in `train_inner` would have invalidated
the paired bootstrap at the end without producing a single error.

### No scaler

Slice 4 and slice 5 both fitted a `StandardScaler`. Slice 6 does not, and the
reason is structural rather than a matter of taste: a tree splits on a
threshold. Divide a column by a thousand and the threshold divides with it, so
the tree is unchanged. Scaling a tree model is a no-op.

This matters again at calibration, where the logistic model's saved bundle
holds three fitted objects and the boosted model's holds two.

---

## The default fit, and how badly it lost

`LGBMClassifier` at its own defaults — 31 leaves, learning rate 0.1, 100 trees
— with nothing set but `objective`, `random_state`, `n_jobs` and `verbose`.

| | mean CV AUC | std |
|---|---|---|
| logistic (slice 5's model, re-measured on these folds) | **0.7107** | 0.0128 |
| LightGBM, defaults | 0.6830 | 0.0191 |

Per-fold difference: `−0.0446, −0.0246, −0.0283, −0.0240, −0.0170`.
**Mean −0.0277. Zero folds of five won.**

### Class weighting was considered and rejected

Every tutorial on imbalanced data reaches for `is_unbalance` or
`scale_pos_weight`. At an 18.9% base rate the model trains without them, and
AUC is rank-based so weighting barely moves it. What weighting *does* do is
inflate every predicted probability — which would have to be undone in the
calibration step immediately afterwards. Recorded here because "we didn't
think of it" and "we considered it and here's why not" look identical in a
notebook otherwise.

### The yardstick, and a scale error avoided

LightGBM's own fold-to-fold std is 0.0191. Judged against that, −0.0277 looks
only marginally outside the noise.

The spread of the *paired difference* is 0.0094. Judged against that, it is
three times out, with every fold agreeing — and a sign test on five folds all
falling the same way is about a 1-in-16 event under the null.

> Comparing a *difference* against the spread of a *level* is a scale error.
> Slice 5 found it comparing two models on the test set. It recurs here at the
> cross-validation level, with the same resolution.

### The folds share a month at every boundary

`TimeSeriesSplit` cuts by **row position**, not by date. Printing the fold
windows makes the consequence visible:

| fold | train | scored |
|---|---|---|
| 1 | 2,689 rows, 2013-01 → **2013-12** | 2,689 rows, **2013-12** → 2014-07 |
| 2 | 5,378 rows, 2013-01 → **2014-07** | 2,689 rows, **2014-07** → 2015-01 |
| 3 | 8,067 rows, 2013-01 → **2015-01** | 2,689 rows, **2015-01** → 2015-05 |
| 4 | 10,756 rows, 2013-01 → **2015-05** | 2,689 rows, **2015-05** → 2015-09 |
| 5 | 13,445 rows, 2013-01 → **2015-09** | 2,689 rows, **2015-09** → 2015-12 |

The boundary month appears on **both sides of every fold**. Since `issue_d` is
monthly and there are roughly 450 loans a month here, up to about 9% of each
scored fold shares a calendar month with rows the model trained on.

Kept as-is for consistency with slice 4, which used the same splitter. The
effect should be mild — `issue_d` is not a feature, and slice 3 already
removed the seventeen columns whose missingness was a date in disguise — but
it is the same objection slice 5 raised when it chose a named date over a row
position, and it is recorded rather than discovered later.

### The overfit gap

| | in-sample | CV | gap |
|---|---|---|---|
| LightGBM default | 0.9304 | 0.6830 | **0.2474** |

The model memorised the training set and carried almost none of it forward.
That is textbook high variance — and high variance is the *tunable* kind of
bad, which is what made step 4 worth running rather than stopping here.

---

## Extrapolation: what a tree cannot do

Slice 5 found that test row 3240 scored **1.000000000000** — total certainty —
on a loan that was repaid, because `loan_to_income` was 19.5 against a
training maximum of 1.782. The claim to test: a tree cannot fail that way,
because it only ever asks whether a value is above a threshold, and 19.5
answers that the same way 1.9 does.

Tested on a **synthetic row**, so no held-out data was spent. An ordinary
borrower from `train_inner` (the row closest to the median `loan_to_income` of
0.207), copied twice, with one number changed.

| model | at 0.207 | at 19.5 | change |
|---|---|---|---|
| logistic (slice 5) | 0.077021 | **1.000000** | +0.922979 |
| LightGBM (default) | 0.045647 | **0.254413** | +0.208767 |

Raw log-odds moved **−2.927 → +31.604**, a swing of **+34.531**. Slice 5
measured the feature's contribution on the real row 3240 at **+34.51**. These
are not quite the same quantity — slice 5 measured from the scaled mean, this
from the median — but the two agree to within 0.02. The mechanism is
confirmed, on a different row, by a different route.

Two things follow:

- **Calibration cannot fix this.** The 1.000000 is the figure *after* the
  sigmoid. Feed slope 0.9475 and shift +0.2898 a score of +31.6 and the result
  rounds to 1 at any printable precision. A sigmoid corrects a bias; it cannot
  correct an out-of-range input. Two different failures needing two different
  fixes.
- **The tree's answer is bounded, not absent.** 0.254 against a base rate of
  0.189 says "somewhat riskier than average", which is a sane thing to say
  about a borrower whose stated income is probably a typo. The point is not
  that the tree ignores the value, but that it says "high, like any high
  value" and stops.

The base row was ordinary in `loan_to_income` by construction and unremarkable
but not median elsewhere — `dti` 10.52 against a median of 17.88, `revol_util`
77.0 against 55.5. That does not affect the comparison, which is controlled:
the same row scored twice, so its quirks appear on both sides and cancel in
the change column. It does mean the two *levels* describe this borrower rather
than a typical one.

---

## Step 4 — tuning search, declared 2026-11-16

Declared before any tuning run and committed to git before it was executed.
Not edited afterwards.

### Why we are searching

The default LightGBM lost to the slice 5 logistic model on all five folds
(0.6830 vs 0.7107, mean paired difference −0.0277). In-sample 0.9304 against
CV 0.6830 — a gap of 0.2474. The model memorises. The search cuts complexity.

### Fixed, not searched

`objective='binary'`, `random_state=42`, `n_jobs`, `verbose=-1`.

`n_estimators=2000` as a ceiling only; early stopping decides the real count.

Early stopping: 50 rounds, metric `auc`, watching the last 20% of each fold's
training rows, split by position so it stays latest in time.

No class weighting, for the reason recorded above.

### The grid — 4 × 4 × 2 = 32 combinations, 160 fits

| parameter | values | default |
|---|---|---|
| `num_leaves` | 4, 8, 16, 31 | 31 |
| `min_child_samples` | 20, 50, 100, 200 | 20 |
| `learning_rate` | 0.02, 0.05 | 0.1 |

The defaults are inside the grid on purpose, so the known reference point is
part of the search rather than outside it.

### Stopping rule

All 32 combinations run. The loop does not stop early for any reason,
including a combination beating 0.7107.

### Selection rule

Highest mean fold AUC. If several sit within one fold-to-fold standard
deviation of the best, take the **simplest** of those — fewest `num_leaves`
first, then largest `min_child_samples`. Same rule slice 4 used to pick
`C=0.05` over `C=0.1` when they were 0.0002 apart.

### Win condition

Measured as a paired per-fold difference against `scores_logreg`, not as a
comparison of two means:

- positive in all 5 folds, **or**
- positive in 4 of 5 **and** mean difference larger than the standard
  deviation of the differences.

A higher mean AUC alone is **not** a win.

### Edge rule

If the selected combination sits on an edge of the grid, the grid was too
narrow. That permits **one** extended search, declared in a new dated section
before running.

### The lose case

If nothing meets the win condition, the slice 5 calibrated logistic model
ships as the slice 6 model. Boosting is reported as tried, tuned and beaten.
Slices 7–10 proceed with logistic. This was committed in writing in
`docs/slice_04_baseline_model.md` before slice 6 began.

### Known caveat

The winner's CV AUC is the maximum of 32 noisy values, so it is optimistic.
CV is the selection device here, not the result. The honest number is the test
AUC, measured once.

---

## Step 4 — extended search, declared 2026-11-16 (amendment)

The first search selected `num_leaves=4`, `min_child_samples=200`,
`learning_rate=0.02` — all three on a grid edge. The edge rule above permits
**one** extended search, declared before running. This is it. No further
extension is permitted.

### The extended grid — 2 × 3 × 2 = 12 combinations

| parameter | values | first search |
|---|---|---|
| `num_leaves` | 2, 3 | 4, 8, 16, 31 |
| `min_child_samples` | 200, 400, 800 | 20 … 200 |
| `learning_rate` | 0.01, 0.02 | 0.02, 0.05 |

`num_leaves=2` is a decision stump — one split per tree, so no interactions at
all. If that wins, the signal in this data is additive.

### Everything else unchanged

Same fixed parameters, same folds, same early stopping, same selection rule,
same win condition, same lose case as the declaration above.

### If the winner is again on an edge

Stop. The extension is spent. Report the result as it stands, with the edge
recorded as a limitation.

---

## What the search found

### Early stopping has to live inside the fold

`cross_val_score` splits each fold into two pieces: train and score. Early
stopping needs a **third** piece to watch while fitting, and there is nowhere
in `cross_val_score` to put it. So the fold loop is hand-written: the last 20%
of each fold's training rows, by position so they stay latest in time, become
the early-stopping watcher; the fold's scoring rows are never watched.

The shortcut — letting early stopping watch the fold's scoring rows — runs
fine, is faster to write, and inflates every combination in the grid by
roughly the same amount, so nothing in the output ever looks wrong. An assert
that the three pieces run forward in time and never overlap guards it.

### The first search

| rank | `num_leaves` | `min_child_samples` | `learning_rate` | mean AUC | std | trees |
|---|---|---|---|---|---|---|
| 1 | 4 | 20 | 0.02 | **0.7053** | 0.0125 | 270 |
| 2 | 4 | 200 | 0.02 | **0.7053** | 0.0136 | 239 |
| 3 | 4 | 50 | 0.05 | 0.7046 | 0.0136 | 79 |
| 4 | 4 | 100 | 0.02 | 0.7041 | 0.0149 | 219 |
| 5 | 4 | 20 | 0.05 | 0.7040 | 0.0134 | 97 |
| … | | | | | | |
| — | 31 | 20 | 0.05 | *(the only one below the tie band)* | | |

Seven of the top ten have `num_leaves=4`; the other three have 8. Nothing with
16 or 31 appears. **Zero combinations reached the 2,000-tree ceiling** — the
maximum any fold used was 418 — so the fixed parameter was adequate, which is
stated rather than hoped.

Tuning raised CV from 0.6830 to 0.7053, a gain of **+0.0223**.

| | in-sample | CV | gap |
|---|---|---|---|
| default | 0.9304 | 0.6830 | 0.2474 |
| **tuned** | 0.7249 | 0.7053 | **0.0196** |

The gap fell by a factor of twelve. For comparison, slice 4's logistic model
had a train-to-test gap of 0.018. **The tuned boosted model now generalises
about as cleanly as the linear one** — the memorising is gone.

But the win condition is measured on the paired difference, not the mean:

| | per fold | mean | spread | folds won |
|---|---|---|---|---|
| tuned − logistic | `−0.0140, +0.0019, −0.0098, +0.0001, −0.0052` | −0.0054 | 0.0060 | 2 of 5 |

**Win condition not met.** The character of the loss has changed completely —
step 2 was three spreads out on five folds of five, this is less than one
spread out on a 2–3 split, with fold 4 a dead heat to four decimals. It is a
statistical tie that failed to become a win, which under the declared rule is
not a win.

### The extension, and the edge that wasn't

All three selected parameters sat on grid edges, in the direction of less
complexity, so the edge rule fired and the one permitted extension was
declared and run.

| | `num_leaves` | `min_child_samples` | `learning_rate` | mean AUC |
|---|---|---|---|---|
| best of extension | 2 | 200 | 0.02 | 0.7010 |
| second | 2 | 400 | 0.02 | 0.7002 |
| third | 3 | 200 | 0.02 | 0.6978 |
| **first search winner** | **4** | **200** | **0.02** | **0.7053** |

**Every one of the twelve scored below 0.7053.** `num_leaves=4` was a genuine
optimum, not a wall. The edge rule did its job and came back negative, which
closes the question properly instead of leaving it open.

Two details worth keeping:

- **`num_leaves=2` reached 0.7010.** That is a stump — one split per tree, so
  no interactions are representable at all — landing within noise of the
  4-leaf model and of logistic. Three different levels of flexibility, one
  answer.
- **`min_child_samples=800` destabilised the model.** Fold-to-fold std rose to
  0.036 and the worst fold fell to 0.6223: too few splits permitted for some
  folds to build a usable model at all.

### The selection rule misfired, and the declaration had a gap

Applied to the extension alone, the declared rule selected
`num_leaves=2, min_child_samples=800` at **0.6919**, whose first fold is
0.6223. That is not a tie-break between equals.

The cause is in the rule's own wording. The tolerance band is *one standard
deviation of the best entry*. In the first search the best entry's std was
0.0125, a narrow band. In the extension it was 0.0199, and the wider band
swallowed 11 of 12 — including configurations too unstable to use.

> A tie-break whose tolerance is set by the best entry's own variability is
> not safe. When the best entry is noisy, the band widens exactly when it
> should tighten.

Noted, not retroactively changed.

The declaration also **never specified whether the extension's selection pools
with the first search or replaces it**. Three readings, three winners:

| reading | selected | CV AUC | win condition |
|---|---|---|---|
| extension alone, rule as written | 2 / 800 / 0.02 | 0.6919 | fails |
| both searches pooled, rule as written | 2 / 400 / 0.02 | 0.7002 | fails |
| highest mean AUC across both | 4 / 200 / 0.02 | 0.7053 | fails |

**All three fail against logistic's 0.7107, so the conclusion is unchanged
under every reading.** That is the strongest form a result can take: it does
not depend on a judgement call that could have gone the other way.

Reported figures use **4 / 200 / 0.02, 239 trees** — the highest score
boosting reached — because the comparison should be generous to the model that
lost.

---

## Calibration

Fitted on `valid`, same method as slice 5, for a reason that is not about this
model: AUC is rank-based and needs no calibration, but Brier and log loss do.
Comparing a calibrated model against an uncalibrated one on those metrics
would flatter the calibrated one for free. **Same method on both, or the
comparison means nothing.**

The sigmoid is fitted on the raw log-odds rather than on probabilities. That
the raw scores really are the log-odds behind `predict_proba` is asserted, not
assumed — `sigmoid(raw) == predict_proba` to within 1e-9 — because the whole
correction would be fitted on the wrong input otherwise.

### The lean was predicted and arrived

`train_inner` is 18.90% bad and `valid` is 23.91%, so the model should
under-predict by around 5 points for the same one-sided censoring reason slice
5 documented. It came in at **−0.0582**; slice 5's logistic model was −0.0511
on test. Same size, same direction, same cause.

### The slope went the other way

| | slope | shift |
|---|---|---|
| slice 5, logistic | 0.9475 | +0.2898 |
| slice 6, LightGBM | **1.1183** | +0.5477 |

Slice 5's slope is *below* 1 — the raw scores were slightly over-spread and
needed flattening. This one is *above* 1: the scores were too **compressed**
and needed stretching apart.

That is what heavy regularisation does. Four leaves, 239 trees and a 0.02
learning rate shrink every prediction toward the middle, and the calibrator
has to undo some of it. The predicted range says the same thing — **0.0465 to
0.5451**. The tuned model never predicts more than 55% risk on any of the
4,768 validation loans, where the logistic model reached 1.000000 on one.

### Sigmoid over isotonic, despite isotonic winning every metric

| method | Brier | log loss | AUC | mean pred | lean | distinct |
|---|---|---|---|---|---|---|
| uncalibrated | 0.1718 | 0.5228 | 0.6826 | 0.1809 | −0.0582 | 4,754 |
| **sigmoid (chosen)** | 0.1675 | 0.5111 | 0.6826 | 0.2391 | 0.0000 | 4,754 |
| isotonic | **0.1661** | **0.5067** | **0.6876** | 0.2391 | 0.0000 | **31** |

Isotonic wins Brier, log loss *and* AUC — the last by +0.0050, which is
unusual enough to explain. Collapsing scores into ties can raise AUC when the
fine ordering being destroyed was noise. Slice 5 saw the opposite: there,
isotonic lost log loss outright by producing predictions of exactly 0 and 1.

Sigmoid chosen anyway, for four reasons, the last decisive:

1. **The table is in-sample for the calibrator.** Isotonic was fitted on
   `valid` and is scored on `valid`. A free-form monotone fit flatters itself
   there; two parameters cannot.
2. **31 distinct values out of 4,754.** Slice 8 sets approval thresholds on
   these numbers, and 31 steps is a very coarse dial.
3. The AUC gain is the kind that disappears on new data, for reason 1.
4. **Slice 5 calibrated the logistic model with a sigmoid.** A Brier
   difference between a sigmoid-calibrated model and an isotonic-calibrated
   one measures the calibrators, not the models.

A monotone map cannot reorder anything, so AUC after the sigmoid must equal
AUC before, exactly. That is an assert. Isotonic creates ties, so its effect
on AUC is measured rather than asserted.

### What the correction fixed, and what it left

Ten equal-sized bins on `valid`. Mean absolute gap **0.0582 → 0.0127**.

| bin | gap before | gap after |
|---|---|---|
| 0 | −0.0023 | **+0.0173** |
| 1 | −0.0152 | **+0.0140** |
| 2 | −0.0372 | −0.0019 |
| 3 | −0.0447 | −0.0038 |
| 4 | −0.0616 | −0.0141 |
| 5 | −0.0714 | −0.0161 |
| 6 | −0.0667 | −0.0022 |
| 7 | −0.1022 | −0.0255 |
| 8 | −0.0799 | **+0.0137** |
| 9 | −0.1009 | **+0.0189** |

Before: every bin negative, worsening almost monotonically as risk rises. The
uncalibrated model is worst exactly where decisions get made.

After: positive at both ends, negative through the middle.

**Now compare slice 5's residual for the logistic model after its sigmoid:**
`+0.0181, +0.0120, −0.0247, +0.0058, −0.0136, −0.0164, −0.0401, +0.0067,
+0.0341, +0.0473`. Positive at both ends, negative through the middle. The
same shape.

> Two models with nothing in common — a linear model and 239 boosted trees,
> one over-spread and one under-spread — leave the **identical residual
> curve** after calibration. The shape is not a property of either model. It
> is a property of the sigmoid, which can slide and stretch but cannot bend.

Slice 5 stated this as a limitation of two-parameter calibration. Slice 6
demonstrates it independently on a different model family.

Bin 9, the bin a lender acts on, sits at +0.0189 against slice 5's +0.0473 —
but that comparison is not fair, because 0.0127 is measured on the set the
calibrator was *fitted* on while slice 5's 0.0219 was measured on test.

---

## The test set

Both bundles loaded cold from disk. The logistic chain had to be rewritten,
because `predict_calibrated` only ever existed inside notebook 05 — see
limitations.

**Cross-check first:** the freshly computed logistic probabilities against the
`p_default_calibrated` column saved in slice 5. Maximum difference **0.00e+00**
across all 5,378 rows. Every slice 5 figure reproduced exactly — AUC 0.6908,
AP 0.3918, Brier 0.1668, log loss 0.5131, lean +0.0029 — from cold files,
through a rewritten function, after a kernel restart.

### Point metrics

| | AUC | AP | Brier | log loss | lean | max pred |
|---|---|---|---|---|---|---|
| logistic | **0.6908** | **0.3918** | **0.1668** | 0.5131 | +0.0029 | **1.0000** |
| LightGBM tuned | 0.6726 | 0.3865 | 0.1682 | **0.5124** | −0.0004 | 0.6877 |

No-skill AP floor: 0.2365.

### The paired bootstrap

2,000 resamples, percentile method, seeded, zero skipped for a single class.
One index array per resample, applied to both models.

| metric | difference | 95% interval | contains zero |
|---|---|---|---|
| **AUC** | **−0.0182** | **[−0.0257, −0.0105]** | **no** |
| average precision | −0.0054 | [−0.0182, +0.0077] | yes |
| Brier | +0.0014 | [−0.0003, +0.0029] | yes |
| log loss | −0.0007 | [−0.0142, +0.0082] | yes |

**Worse at ranking, indistinguishable on everything else.** The AUC loss is
real and survives test-sample luck. It is also three times larger than the CV
gap of −0.0054, so the test set confirmed the decision more strongly than the
device that made it.

### The log loss win is noise, and why that matters

LightGBM's log loss is lower — and the mechanism behind it is real. Maximum
prediction 1.0000 against 0.6877; slice 5 measured that single extrapolating
row at **1.2% of the entire test log loss**. Bounded predictions genuinely
protect the boosted model from a failure the linear model cannot avoid.

But the interval is [−0.0142, +0.0082]. The protection is real and **not
decisive**: LightGBM gives the advantage back elsewhere, and the net
difference is indistinguishable from zero.

Two further points of precision. Slice 5 also measured the logistic model
*with* its input guard applied: **0.5076**, better than LightGBM's 0.5124. So
the margin is entirely the one failure slice 5 documented and deliberately did
not fix. Guarded, the logistic model wins that metric too.

### What pairing is worth depends on how similar the models are

| | unpaired AUC interval | paired interval | ratio |
|---|---|---|---|
| slice 5 (its model vs slice 4's) | 0.0328 | 0.0056 | **5.9×** |
| slice 6 (LightGBM vs logistic) | 0.0336 | 0.0152 | **2.2×** |

Slice 5 paired two nearly identical models — same features, same family, one
refitted on 23% less data. Their predictions moved together almost perfectly,
so the sampling luck cancelled almost completely.

Slice 6 pairs a linear model against 239 boosted trees. Genuinely different
models, less correlated errors, so less cancels.

> The narrowing factor from pairing measures how similar the two models are.
> Near-twins cancel almost everything; different model families cancel much
> less. Pairing is always worth doing, but how much it buys is not a constant.

---

## Lending Club's own verdict

`grade`, `sub_grade`, `int_rate` and `installment` were held out in slice 3 for
this comparison.

### Recovering them took care

None of the four are in `model_table.parquet` — notebook 2 dropped them along
with `issue_d`. They were recovered from `sample_50k.parquet` by **position**,
repeating notebook 2's `loan_status` filter, then replaying notebook 03's year
restriction and `features.py`'s stable mergesort and quantile cut.

`mergesort` is load-bearing: it is the only stable sort in pandas, so loans
sharing an `issue_d` keep their relative order. Any other sort returns the
same rows in a different order.

Verified row-for-row on five columns nothing transformed — `issue_d`,
`target`, `loan_amnt`, `annual_inc`, `revol_bal` — rather than on row counts
alone. Slice 2's `emp_length` bug passed a row-count assert while 40% of its
values were scrambled; **an assert only tests what you thought to ask**.

`int_rate` and `installment` arrive as text and need converting. `grade` and
`sub_grade` are encoded as ordinals, because A is genuinely safer than B: the
order carries information that one-hot would discard, at 40 columns' cost.

### The ladder

| | features | CV AUC | std |
|---|---|---|---|
| **A** — LC's verdict only | 4 | 0.7029 | 0.0173 |
| **B** — the slice 5 feature set | 71 | 0.7107 | 0.0128 |
| **C** — both | 75 | **0.7235** | 0.0147 |

B reproduced 0.7107 exactly, which is what proves the row alignment rather
than merely suggesting it.

| | per fold | mean | folds won |
|---|---|---|---|
| **C − A** — the features add to LC | `0.0345, 0.0061, 0.0261, 0.0159, 0.0208` | **+0.0207** | **5/5** |
| **C − B** — LC adds to the features | `0.0062, 0.0140, 0.0153, 0.0192, 0.0095` | **+0.0128** | **5/5** |
| A − B — LC alone vs features alone | `−0.0283, 0.0079, −0.0108, 0.0032, −0.0113` | −0.0079 | 2/5 |

### Four columns match seventy-one

A − B is less than one spread from zero and splits 2–3. **A statistical tie.**
Lending Club's underwriters compressed essentially the same predictive power
into a single letter that this slice's 71 engineered features achieve.

### But the two are not redundant

If `grade` were only a summary of what the features already contain, C would
equal the better of A and B. It does not — C beats **both**, each on five
folds out of five.

Slice 3 explains why. `grade` was assigned when Lending Club could see the
**whole applicant pool, including everyone they rejected**. The feature set
describes only the people who got through, which is why FICO scores just 1.8×
and the entire delinquency family is flat. Slice 3 named this range
restriction and argued it.

> **`C − B = +0.0128` on 5 folds of 5 is a measurement of range restriction** —
> a lower bound on how much predictive information was destroyed before this
> dataset was written.

Grade's own strength, on `train_inner`:

| grade | loans | bad rate |
|---|---|---|
| A | 2,660 | 5.15% |
| B | 4,571 | 11.42% |
| C | 4,686 | 20.76% |
| D | 2,446 | 28.95% |
| E | 1,236 | 37.95% |
| F | 442 | 43.89% |
| G | 93 | 50.54% |

Worst to best: **9.8×**. Slice 3 recorded 8.6× measured by decile across the
whole train half; this is by grade level on `train_inner` alone. Different
window, different method, both correct.

### Who the model is for decides whether any of this may be used

None of the four columns are leakage — all are known before funding. But they
are not the applicant's circumstances either. They are **Lending Club's own
judgement**, the same objection slice 3 raised about `verification_status`.

- **An investor** picking among loans LC has already listed: `grade` is on the
  screen. Using it is correct, and `C − A = +0.0207` says this engine adds
  real value on top of it.
- **The lender** deciding approve or decline: `grade` does not exist yet.
  Assigning it *is* the decision. A model reading `grade` to decide approval
  is reading an answer it is supposed to produce.

This project's stated goal is the second. **So the shipped model remains B at
0.7107, not C at 0.7235** — the most accurate model built in this slice is one
that cannot be used, and the reason is a question about *when* information
exists, not how well it predicts.

*(The caveat above was written before the comparison was run, but committed
after. The argument stands on its own terms and does not rest on the result.)*

---

## Figures

- `figures/slice06_overfit_gap.png` — in-sample against cross-validated AUC
  for the default and tuned models, with logistic's level marked. Tuning
  closed the gap from 0.2474 to 0.0196 without closing the deficit.
- `figures/slice06_paired_bootstrap.png` — the distribution of 2,000 paired
  AUC differences, with the 95% interval shaded and zero marked. The interval
  sits entirely to the left of zero.
- `figures/slice06_verdict_ladder.png` — A, B and C with every individual fold
  plotted, so the 5-of-5 consistency is visible rather than asserted.

---

## Handover

`models/calibrated_lgbm.joblib` holds **two** fitted objects — model then
calibrator — plus `apply_order`, the column list, the exact constructor
parameters as used (239 trees, not the 2,000 ceiling), the Platt slope and
shift, the CV and test metrics, the paired difference and its interval, and
both library versions.

It also carries `cv_note`, recording in the file itself that this model lost
and is not what ships, so nobody loading it has to find the doc first. And
`loan_to_income_guard_not_applied: 1.782`, the same key and convention as
slice 5's bundle, so slice 9 treats both identically.

There is **no scaler**, and the bundle says so under
`no_scaler_on_purpose` rather than leaving a reader to wonder.

Verified by loading cold and comparing the probability **arrays** — maximum
difference 0.00e+00 across 4,768 rows. Slice 4's stale-variable failure is why
this compares arrays rather than a summary number.

`data/processed/predictions_test_slice6.parquet` — `issue_d`, `zip_code`,
`addr_state`, `target`, `p_logreg`, `p_lgbm_tuned` for all 5,378 test loans,
so slice 7 can test fairness on both models and ask whether model choice
changes the picture.

**Slice 5's bundle and predictions are untouched. The slice 5 calibrated
logistic model is the slice 6 model.**

---

## Known limitations

- **The test set has now been used three times.** Every use has been for
  reporting; the decision was made on CV folds. Slices 7–10 must keep that
  discipline or the 0.6908 stops meaning anything.
- **The fold boundaries share a calendar month**, up to roughly 9% of each
  scored fold. `TimeSeriesSplit` cuts by row position; a date-based splitter
  would fix it, at the cost of consistency with slice 4.
- **Fold 1 trains on 2,689 rows, and the early-stopping watcher leaves about
  2,150.** At a 19% bad rate that is roughly 100 bad loans deciding when to
  stop. Fold 1 is the noisiest of the five and some of the fold-to-fold spread
  is this, not real instability.
- **The tie-break rule is unsafe when the best entry is noisy**, as documented
  above. Not retroactively changed; worth replacing before it is used again.
- **The declaration did not say whether the extension pools with the first
  search.** Three readings give three winners, all of which lose, so the
  conclusion is unaffected — but the ambiguity was real and should not recur.
- **The winner's CV AUC is the maximum of 44 noisy values** and is therefore
  optimistic. The test AUC is the honest number.
- **`loan_to_income` remains unguarded in the logistic pipeline.** Documented
  in slice 5, measured again here, fixed in slice 9. LightGBM needs no guard —
  not because it is careful, but because a tree cannot leave the range it was
  shown.
- **Neither `predict_calibrated` function lives in `src/`.** Both exist only
  as notebook cells and had to be rewritten to run this slice. Slice 5's
  handover says its function is "deployed unchanged" at slice 9, which is not
  possible as things stand. Both belong in `src/`.
- **Everything is still measured on finished loans only**, and inherits slice
  3's one-sided censoring. Relative movement between models is sound; absolute
  levels are pessimistic.
- **`grade` is excluded from the shipped model on a judgement about the use
  case, not a measurement.** An investor-facing version of this engine would
  legitimately include it and would score roughly 0.7235 on these folds.
