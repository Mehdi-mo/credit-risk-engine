# Slice 6 — strong model

Input: `features_train.parquet` (20,902 rows) and `features_test.parquet`
(5,378 rows), 71 model features each, plus `models/calibrated_logreg.joblib`
and `data/processed/predictions_test_calibrated.parquet` from slice 5.

Splits reproduced exactly from slice 5: train_inner 16,134 (2013-01 → 2015-12,
bad rate 18.90%), valid 4,768 (2016-01 → 2016-08, 23.91%), test 5,378
(2016-09 → 2017-12, 23.65%). Split date 2016-01-01.

Library: LightGBM 4.7.0.

---

## Step 4 — tuning search, declared 2026-11-16

Declared before any tuning run. Not edited afterwards.

### Why we are searching

The default LightGBM lost to the slice 5 logistic model on all five folds
(0.6830 vs 0.7107, mean paired difference −0.0277). In-sample 0.9304 against
CV 0.6830 — a gap of 0.2474. The model memorises. The search cuts complexity.

### Fixed, not searched

`objective='binary'`, `random_state=42`, `n_jobs`, `verbose=-1`.

`n_estimators=2000` as a ceiling only; early stopping decides the real count.

Early stopping: 50 rounds, metric `auc`, watching the last 20% of each fold's
training rows, split by position so it stays latest in time.

No class weighting (`is_unbalance` / `scale_pos_weight`): it inflates every
probability and step 5 is calibration.

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
AUC at step 6, measured once.

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
