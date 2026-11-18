# Slice 7 — explainability and fairness

Input: `models/calibrated_logreg.joblib` (shipped), `models/calibrated_lgbm.joblib`
(slice 6), `features_train.parquet` / `features_test.parquet`, and
`data/processed/predictions_test_slice6.parquet`.

Output: `src/predict.py`, four processed files, three figures, and two findings
that cut in opposite directions.

**Withholding geography worked: the mean score gap between census regions is
0.025 log-odds and no fairness metric separates them. Withholding credit history
did not, because a feature correlated 0.917 with it sits in the model — producing
a 13.4 point approval gap and 12.3 points more creditworthy borrowers refused.
Same dataset, same 71 columns, opposite outcome.**

## What this slice adds

Slices 4 to 6 asked whether the model works. This one asks two different
questions:

- **Why does it say what it says**, per borrower rather than per coefficient.
- **Who pays when it is wrong**, and whether leaving a variable out of the
  feature list actually protects the people that variable describes.

## Step 0 — the scoring chain moved into `src/`

`predict_calibrated` lived only in a notebook 05 cell. By slice 6 it had already
been retyped by hand, under a second name, in notebook 06 cell 11 — the comment
on that cell says so outright. Two copies of one chain in two notebooks.

Both chains now live in `src/predict.py`, which adds four things to the copies it
replaced:

- an assert that the bundle's `apply_order` is the chain the function implements,
  so a bundle saved with a different order fails loudly instead of scoring wrong
- a column-presence check that names what is missing
- `apply_guard=False` on the logistic chain, so slice 9 turns the
  `loan_to_income` cap on by flipping a flag rather than editing a deployed
  function. No guard on the boosted chain, and that is a decision: a tree cannot
  extrapolate past its training range, so clipping the input changes nothing
  there — the same borrower who scores 1.000000 under the logistic model scores
  0.25 under the boosted one
- a `__main__` self-check: `python -m src.predict` loads both bundles cold,
  scores raw `features_test.parquet`, and asserts the arrays match the published
  probabilities

| check | result |
|---|---|
| logistic reproduces `p_logreg` | max difference 0.00e+00 over 5,378 rows |
| LightGBM reproduces `p_lgbm_tuned` | max difference 0.00e+00 over 5,378 rows |
| guard off → on, max probability | 1.000000 → 0.982307, 2 of 5,378 rows move |

Those are slice 5's recorded numbers, so matching them tests the chain rather
than restating it. Slice 5's claim that the function is "deployed unchanged" at
slice 9 is now true rather than aspirational, and slice 10 inherits a one-command
regression test.

## Step 1 — SHAP on the shipped model, and what it does not add

For a linear model SHAP is not an approximation. It is one multiplication:

```
contribution of feature j on row i  =  coefficient[j] x scaled_value[i, j]
```

The reason it collapses that far is specific to this model. SHAP measures a
contribution relative to the mean of a background set. The scaler was fitted on
`train_inner`, so the scaled mean of `train_inner` is exactly zero — largest
absolute column mean 9.78e-16 — and the subtraction disappears.

Two consequences, both asserted rather than assumed:

| check | result |
|---|---|
| contributions + intercept = `decision_function` | max difference 2.13e-14 |
| hand formula = `shap`'s linear explainer | max difference 2.22e-16 |
| `shap` base value vs model intercept | −1.6291 vs −1.6291 |
| features with a zero coefficient contributing exactly zero | 24 of 71 |

A further consequence worth recording: the Platt correction is itself linear in
the score (slope +0.9475, shift +0.2898). So scaler → model → calibrator is **one
linear map all the way to calibrated log-odds**, and only the final sigmoid into
probability is curved. Contributions in calibrated log-odds are these multiplied
by 0.9475; nothing else changes. Calibration rescales the explanation without
reordering any part of it.

### So what does SHAP buy here?

**Not a new importance ranking.** Average absolute SHAP for a linear model equals
`|coefficient| x mean|scaled value|` exactly — verified to 5.55e-17 — so it ranks
features by the coefficient, adjusted only for how spread out each feature is.
Step 3 measures the resulting rank correlation at 0.984.

**The per-row view, and row 3240 is why that matters.** Slice 5's extreme row:

| | contribution |
|---|---|
| `loan_to_income` | **+34.508** |
| `term` | +0.727 |
| `application_type_Joint App` | +0.284 |
| `fico` | −0.210 |
| `acc_open_past_24mths` | −0.178 |
| ...66 others plus a base of −1.629 | |
| **total score** | **+33.368** |

One feature contributes more than the entire prediction; everything else on the
row nets to −1.139 pulling the other way. A coefficient cannot show that, because
it is one number for 5,378 different people. Slice 5 worked this row out by hand
and recorded +34.51 and "about −1.14". A general-purpose method reproduces both
to the decimal, across two slices and two methods.

**A shared unit.** Log-odds, which the boosted model also produces. Without it,
step 3's cross-model comparison would not be possible at all.

## The three silent defaults

This is the finding that matters most, and it is not about any feature.

**Three times in one slice, a `shap` call quietly did something other than what
it was asked.** Every time the output was internally consistent and passed its own
additivity check. Nothing in the result flagged it.

| # | call | what it did | how it showed |
|---|---|---|---|
| 1 | `LinearExplainer(model, train_inner)` | resampled 16,134 background rows down to 100 | base value −1.5476 against an intercept of −1.6291; per-feature values off by up to 1.73e-02 |
| 2 | `TreeExplainer(model, data=background)` | kept 100 of the 1,000 rows we chose, and not the first 100 | base value −1.64932 against a background mean of −1.60842 |
| 3 | `shap_interaction_values` on an interventional explainer | ignored the mode entirely | collapse identity off by 8.24e-01; reconstruction off by ~1.8 |

The third is the worst, and the library says so itself. `_tree.py` line 843:

```python
# assert self.feature_perturbation == "tree_path_dependent", "Only feature_perturbation = "tree_path_dependent" is supported for SHAP interaction values right now!"
```

The guard that would have rejected the call is commented out rather than
implemented. And line 923 reuses a stale base value:

```python
self.expected_value = getattr(self, "expected_value", phi[0, -1, -1, 0])
```

The interaction matrix carries its own base value, but if the explainer already
has one from a previous `shap_values` call, the matching one is thrown away.

### How the reference point moves

The background is not a formality. For the boosted model:

| background | mean raw margin |
|---|---|
| `shap`'s hidden 100 | −1.64932 |
| our 1,000, seed 42 | −1.60842 |
| all 16,134 `train_inner` | −1.56899 |

0.08 of log-odds across that range — roughly 1.5 points of probability near the
base rate.

> Every one of these produced plausible numbers that satisfied every internal
> check. The only reason any of them surfaced is that each result was compared
> against something computed independently. **An explanation that passes its own
> additivity test can still be measured against a reference nobody chose.**

### What we did instead

Build the masker explicitly, then **assert it was obeyed**:

```python
masker = shap.maskers.Independent(background, max_samples=len(background))
explainer = shap.TreeExplainer(model, data=masker, feature_perturbation='interventional')
assert np.asarray(explainer.data).shape[0] == len(background)
```

Recorded for the boosted model: interventional, background 1,000 rows of
`train_inner`, seed 42, base value −1.6084 — which equals the mean raw margin
over that background exactly, as it must.

Interaction values use a separate `tree_path_dependent` explainer with the
interactions called first, so the base value comes from the matrix itself. Its
base is −1.568996, which equals the mean raw margin over all 16,134 `train_inner`
rows (−1.56899567). That mode is not an arbitrary reference — it is the full
training distribution, carried inside the trees.

**The two parts therefore sit on different reference points, deliberately.** The
values are interventional and comparable with step 1; the interactions are path
dependent because the library supports nothing else.

## Step 2 — interactions, and why the headline number is misleading

| check | result |
|---|---|
| matrix symmetric | max \|M − Mᵀ\| = 1.12e-16 |
| collapse gives back the SHAP values | max difference 9.99e-16 |
| all cells + base = raw margin | max difference 5.77e-15 |

| | absolute attribution |
|---|---|
| main effects (diagonal) | 6,624.8 |
| interactions (off-diagonal) | 2,174.4 |
| **interaction share** | **24.71%** |

A quarter of all attribution mass is pairwise. Taken alone that reads as
substantial structure a linear model cannot represent — and it would be the
opposite of the truth.

### Three measurements say it is worthless

**Delete every interaction and the ranking does not move.**

| | test AUC |
|---|---|
| full boosted model | 0.6726 |
| main effects only | 0.6725 |
| **cost of removing every interaction** | **+0.0001** |

**82% of it cancels inside each row.** Absolute interaction mass per row 0.4043;
net after signs cancel 0.0731. Net interactions have a standard deviation of
0.0929 against 0.5820 for main effects — a ratio of 0.16.

**It is identical on train and test.** 24.93% on a seeded 5,378-row sample of
`train_inner` against 24.71% on test, a difference of 0.22 points. Fitted noise
would be far larger on train.

### What it actually is

Stability across train and test rules out overfitting, so this is a property of
the representation rather than of the data. To approximate a smooth additive
function with depth-2 trees, boosting builds trees that split on one feature and
then another; each such tree, read alone, contains a product term. Sum 239 of
them with different split orders and those product terms cancel. What is left is
an additive function wearing a non-additive costume.

That also explains the thinness: the strongest pair
(`acc_open_past_24mths` × `loan_to_income`, 94.5) is 1.07% of all absolute
effect, spread across roughly 2,485 possible pairs. `term` appears in 5 of the
top 10 pairs and `acc_open_past_24mths` in 5 of 10 — both are frequent early
splits, which is what produces pair entries mechanically.

> **A SHAP interaction share measures how a model represents itself, not what it
> uses.** Computing only the share — the usual practice — would have produced the
> opposite conclusion. The one-line ablation is what settles it.

### Against slice 6

Slice 6 inferred additivity indirectly: tuning drove the model to 4 leaves and a
2-leaf stump still reached 0.7010. Slice 7 measures it. The caveat stated before
computing still holds in one direction — 4 leaves was *chosen* by tuning, so a low
share would have partly restated that choice. The share came back high, and the
ablation is what makes the test informative rather than circular.

No claim is made that the pair list reflects real structure. Given 82%
cancellation and zero AUC cost, the honest reading is that it is representation
artefact throughout.

## Step 3 — four importance measures, two models

| | abs_coef | shap_logreg | shap_lgbm | lgbm_gain |
|---|---|---|---|---|
| **abs_coef** | 1.000 | 0.984 | 0.469 | 0.473 |
| **shap_logreg** | 0.984 | 1.000 | 0.493 | 0.497 |
| **shap_lgbm** | 0.469 | 0.493 | 1.000 | 0.999 |
| **lgbm_gain** | 0.473 | 0.497 | 0.999 | 1.000 |

Spearman rank correlation. Two blocks of near-perfect agreement; everything
between them around 0.47.

> **The choice of importance method barely matters. The choice of model does.**
> Practitioners argue about gain versus SHAP versus permutation. Within a model
> those agree at 0.98 and 0.999. Across two models scoring within 0.018 AUC of
> each other, agreement is 0.47.

A prediction made before computing this was wrong and is recorded as such: gain
counts where splits happen rather than what they do to the output, so it was
expected to disagree with the boosted model's own SHAP values. It is 0.999.

### Where the agreement breaks

| top k | shared |
|---|---|
| 5 | 5 of 5 |
| 10 | 5 of 10 |
| 20 | 11 of 20 |
| 30 | 18 of 30 |

Unanimous on the top five — `term`, `loan_to_income`, `acc_open_past_24mths`,
`fico`, `dti`, with `term` rank 1 in all four measures — and then not one of
positions 6 to 10 is shared.

The logistic model uses 47 features, LightGBM 25, and 20 are common to both.

**Used by LightGBM, discarded by L1** — 5 of the 24, carrying 2.4% of total gain:

| feature | shap_lgbm | gain | rank |
|---|---|---|---|
| `bc_open_to_buy` | 0.0364 | 643.8 | 9 |
| `annual_inc` | 0.0039 | 56.8 | 17 |
| `bc_util` | 0.0034 | 70.4 | 19 |
| `num_actv_rev_tl` | 0.0032 | 75.6 | 20 |
| `total_rev_hi_lim` | 0.0030 | 79.1 | 21 |

**Used by the logistic model, ignored entirely by LightGBM**, with their logistic
ranks: `num_rev_tl_bal_gt_0` (8), `log_revol_bal` (9), `total_bal_ex_mort` (12),
`verification_status_Source Verified` (16), `verification_status_Verified` (17),
`total_il_high_credit_limit` (18), `home_ownership_OWN` (19),
`purpose_credit_card` (21), `purpose_other` (22), `percent_bc_gt_75` (25).

### Why they disagree

Two mechanisms, both visible in those lists.

**One-hot categoricals live in one model and are invisible to the other.** A
binary column makes a weak split, so trees skip it; a linear model picks up a
steady shift for free. `verification_status`, `purpose_*` and `home_ownership_OWN`
are all in that family.

**Correlated features get a different representative each time.**
`num_rev_tl_bal_gt_0`, `log_revol_bal` and `total_bal_ex_mort` all measure
revolving credit usage. LightGBM uses none of them and ranks `bc_open_to_buy`
ninth instead. Both models found the same signal and picked a different proxy.

> Slice 5 showed that *whether a feature was selected* is unstable between two
> runs of the same model. It is unstable between model families too, and the
> mechanism is correlation. **A feature's absence from one model's list is not
> evidence about that feature.**

## Step 4 — group definitions, fixed before any outcome was computed

### What the data forbids

- **No race, ethnicity or gender**, and no surname, so BISG-style proxying is
  impossible. The question US fair-lending law actually asks cannot be answered
  with this dataset. That is a limitation of the work, not a finding about the
  model.
- **Zip is unusable.** 723 three-digit prefixes in test, not one reaching 100
  loans.
- **Most states are unusable.** 19 have fewer than 50 test loans.

### The groups, and why each

| grouping | groups | the model sees it? | reason |
|---|---|---|---|
| census region | Northeast 1,023 / Midwest 916 / South 1,978 / West 1,461 | **no** | tests whether withholding geography worked. Every row lands somewhere, every group is large |
| the five substantial states | CA 792, TX 501, NY 424, FL 407, IL 188, other 3,066 | **no** | finer geography. The other 45 states are reported as unmeasurable, not averaged in |
| credit file thickness | thick 4,480 / thin 898 | **yes** | credit invisibility is a recognised fair-lending concern. This is the disparate-impact side rather than the blindness side |

`credit_history_months < 120` — ten years, a round domain number, chosen the way
slice 3 chose its 600-month cap rather than read off a quantile. A threshold that
produced fewer than about 400 thin-file rows was to be moved and the move
documented. It produced 898, so it stands as chosen.

> **Reading group sizes off test is fine; reading outcomes is not.** Sample size
> is a property of the data, not of the answer. Knowing CA has 792 loans is how
> we know a state-level cut is viable at all.

Also asserted before computing anything: `addr_state` and `zip_code` are not in
the model's feature list. The premise of the whole section is checked rather than
assumed.

## Steps 5 and 6 — the results

### Geography: nothing detectable

Every region overlaps every other region on approval rate, FPR, TPR, AUC and
calibration. FPR runs 0.448 to 0.470 with intervals about ±0.03 wide. The
Northeast-minus-West contrast is not distinguishable from zero on any of the five
metrics.

**Indistinguishable is not equal.** At roughly 1,000 rows per region a gap of up
to about 5 points could hide inside these intervals. The claim is "no disparity
large enough to detect at this sample size", and that is the strongest claim the
data supports.

### Credit file thickness: large, clean, completely separated

Calibrated logistic model, threshold 0.20:

| | n | bad rate | approval rate | FPR | AUC | calibration gap |
|---|---|---|---|---|---|---|
| thick (≥10y) | 4,480 | 0.227 | 0.501 [0.487, 0.516] | 0.435 [0.419, 0.451] | 0.691 | +0.008 |
| thin (<10y) | 898 | 0.284 | **0.367 [0.336, 0.398]** | **0.558 [0.521, 0.598]** | 0.684 | −0.024 |

Tested as differences rather than by eye:

| metric | thin − thick | 95% interval | real |
|---|---|---|---|
| approval rate | −0.1336 | [−0.1688, −0.0984] | **yes** |
| FPR | +0.1229 | [+0.0812, +0.1639] | **yes** |
| TPR | +0.1048 | [+0.0510, +0.1576] | **yes** |
| AUC | −0.0073 | [−0.0492, +0.0319] | no |
| calibration gap | −0.0325 | [−0.0648, −0.0020] | **yes** |

Borrowers with under ten years of credit history are approved **13.4 points less
often**, and **55.8% of the ones who would have repaid are refused**, against
43.5% for everyone else.

The model is not worse at the job — AUC is indistinguishable. The approved books
perform the same — bad-rate-of-approved 0.139 against 0.129, overlapping.

Calibration deserves care. Each group's own gap contains zero, but the
*difference* does not, at −0.0325 [−0.0648, −0.0020]. Comparing two intervals
would have missed it; testing the difference catches it — the same scale error
slice 5 documented. The direction matters: the model **under**-predicts risk for
thin-file borrowers, so it is being relatively generous to them in its
probabilities and they are still refused far more. Were it perfectly calibrated
for them they would fare worse.

> The access gap is a single threshold meeting a higher base rate (28.4% against
> 22.7%), not the model overstating thin-file risk. When base rates differ, no
> classifier can hold equal calibration and equal false-positive rates at once.
> **"Is this model fair?" has no single answer here.** By calibration it is; by
> equal opportunity it is not. Which definition applies is a policy choice, and
> the threshold is the lever — which hands slice 8 a sharper problem than a
> profit curve.

### Two exploratory results, labelled as such

The *groups* were pre-registered. The TX and FL *contrasts* were chosen after
seeing the step 5 table, which is choosing on test. Thin-versus-thick and
Northeast-versus-West follow directly from the group definitions and are
confirmatory; these two are not.

- **Texas**: FPR +0.0721 [+0.0154, +0.1238] and approval −0.0502 [−0.0957,
  −0.0037] against the rest of the country. Both exclude zero, and the two are
  mechanically linked rather than independent.
- **Florida**: calibration gap −0.0581 [−0.1019, −0.0135]; the model
  under-predicts Florida risk.

With four contrasts and five metrics there are 20 intervals, so roughly one
should exclude zero by chance.

### Why the two geographies differ: the decomposition

Score equals base plus the sum of contributions, and the base is the same for
everyone, so the difference in mean score between two groups is **exactly** the
sum of the differences in mean contribution. Asserted to 1e-8.

| | mean score gap |
|---|---|
| Northeast minus West | **+0.0247 log-odds** |
| thin minus thick | **+0.1984 log-odds** |

Eight times larger. What carries the thin-file gap:

| feature | contribution to the gap | correlation with `credit_history_months` |
|---|---|---|
| `total_acc` | +0.1138 | +0.269 |
| `term` | −0.1033 | +0.047 |
| `mo_sin_old_rev_tl_op` | +0.0787 | **+0.917** |
| `log_revol_bal` | +0.0608 | +0.191 |
| `home_ownership_RENT` | +0.0587 | −0.212 |
| **`credit_history_months`** | **−0.0005** | — |

The top five carry +0.2087 of the +0.1984 total; the other 66 net to about −0.01.

### The finding

| | weight | SHAP rank | contribution to the gap |
|---|---|---|---|
| `credit_history_months` | **0.0003** | 45 of 71 | −0.0005 |
| `mo_sin_old_rev_tl_op` | **0.0586** | 14 | +0.0787 |

`mo_sin_old_rev_tl_op` is months since the oldest revolving account was opened.
It correlates with credit history length at **0.917**.

**The variable that defines the group is not in the model. A near-copy of it is,
at two hundred times the weight.**

> **Excluding a variable protects nothing unless you check what correlates with
> it.** Slice 3 kept geography out on principle and never verified that decision
> against the correlation structure. It turned out to be safe — nothing among the
> 71 proxies for region. The credit-age case is the same mechanism with the
> opposite outcome, in the same dataset.

Nothing went wrong to produce this. L1 saw two correlated features and kept the
stronger one; slice 3 had already rated `credit_history_months` weak at 1.29×,
and slice 5 had already shown selection is unstable at the margin. Credit history
is a legitimate underwriting factor and the model is entitled to use it. The
point is narrower: *had* it been dropped for fairness reasons, nothing would have
been achieved and nobody would have known.

Thin-file status is also not a protected class. It is a fair-lending concern
because it correlates with age and recent arrival, not a legal category.

## Step 7 — the model choice changes the fairness picture

| thin − thick | logistic (shipped) | LightGBM (slice 6) |
|---|---|---|
| approval rate | −0.1336 [−0.1688, −0.0984] | −0.0499 [−0.0869, −0.0151] |
| FPR | +0.1229 [+0.0812, +0.1639] | +0.0547 [+0.0132, +0.0962] |
| calibration gap | −0.0325 [−0.0648, −0.0020] | −0.0470 [−0.0798, −0.0165] |
| mean score gap | +0.1984 | +0.0729 |
| **test AUC** | **0.6908** | **0.6726** |

Both disparities are real. The model slice 6 rejected has less than half of one.

Tested properly — and here the comparison **can** be paired, because both models
score identical rows:

| | logistic minus LightGBM | 95% interval |
|---|---|---|
| FPR gap | **+0.0682** | [+0.0372, +0.1001] |
| approval gap | **−0.0838** | [−0.1116, −0.0567] |

Both exclude zero, and the intervals are far tighter than comparing the two
separate gaps would allow — slice 5's paired bootstrap, applied a third time.

### Why the boosted model has less of it

| | logistic carriers | LightGBM carriers |
|---|---|---|
| 1 | `total_acc` +0.1138 | `term` −0.0944 |
| 2 | `term` −0.1033 | `mo_sin_old_rev_tl_op` +0.0529 |
| 3 | `mo_sin_old_rev_tl_op` +0.0787 | `tot_hi_cred_lim` +0.0365 |
| 4 | `log_revol_bal` +0.0608 | `avg_cur_bal` +0.0303 |
| 5 | `home_ownership_RENT` +0.0587 | `fico` +0.0239 |

`total_acc` — the logistic model's single largest carrier — is gone; LightGBM
ranks it 22nd. `log_revol_bal` is gone entirely. The one carrier common to both
is `mo_sin_old_rev_tl_op`, the 0.917 twin.

**The sparser model is fairer on this axis by accident.** LightGBM uses 25
features against 47, so it picks up fewer of the correlated proxies. Nobody
designed that; it falls out of using fewer features.

Region holds for LightGBM too — every interval overlaps — so the geography result
replicates across model families rather than resting on one model.

> **The better-ranking model is the less equal one.** Slice 6 chose on AUC alone,
> correctly, because fairness was not in scope. Slice 7 shows that choice carried
> a measurable cost that nobody looked for at the time.

## Figures

- `figures/slice07_importance.png` — the two models' attribution shares, top 15.
  Identical top five, then the orange bars vanish.
- `figures/slice07_fairness_fpr.png` — FPR with bootstrap intervals, both models,
  by region and by file thickness. The left panel overlaps everywhere; the right
  panel separates.
- `figures/slice07_gap_decomposition.png` — what carries each gap, both panels on
  one x-axis so the eight-fold difference is visible rather than asserted.

## Handover

New files:

- `src/predict.py` — both scoring chains plus a self-check. Run
  `python -m src.predict` from the repo root.
- `data/processed/shap_logreg_test.parquet` — 5,378 × 75. Per-row contributions
  in log-odds, plus `score` (exact: base + row sum) and `p_logreg` as a readout
  that is **not** part of the sum.
- `data/processed/shap_lgbm_test.parquet` — same shape, interventional,
  background 1,000 rows of `train_inner`, seed 42, base −1.6084.
- `data/processed/fairness_groups_test.parquet` — the three group labels per row.
- `data/processed/importance_comparison.csv`,
  `fairness_logreg_test.csv`, `fairness_lgbm_test.csv`,
  `fairness_contrasts.csv`.

For slice 8: the threshold is the lever on the thin-file gap, and the trade is no
longer only risk against volume. It is risk against volume against who gets
refused.

For slice 9: `predict_calibrated(bundle, X_raw, apply_guard=True)` is the
deployed call. The flag is the only change needed.

## Known limitations

- **The test set has now been used four times**, every time for reporting. No
  choice in this slice was made on it, and the two exploratory contrasts are
  labelled above.
- **Race, ethnicity and gender are absent from the dataset**, and no surname
  exists for proxying. The central fair-lending question is unanswerable here.
  Everything in this slice is about groups the data can define.
- **Zip-level fairness is impossible at this sample size** — 723 prefixes, none
  reaching 100 loans. 19 states are likewise unmeasurable and are reported as
  such rather than pooled.
- **"No detectable regional disparity" is bounded by the intervals**, not by
  zero. A gap of up to roughly 5 points could sit inside them.
- **The interaction analysis uses a different reference point from the SHAP
  values** — path dependent against interventional — because the library supports
  nothing else. Rankings and shapes are comparable; levels are not.
- **The 24.71% interaction share is a property of the representation.** It should
  not be quoted without the ablation beside it.
- **Thin-file status is a proxy for concerns the data cannot measure directly**
  (age, recent arrival), not a protected class in itself.
- **Everything inherits slice 3's one-sided censoring.** Relative movement between
  groups is sound; absolute levels are pessimistic.
- **The `loan_to_income` guard is still not applied** to any number in this slice,
  by design. Slice 9 applies it.
