# Credit Risk Engine

Predicting the probability that a Lending Club borrower defaults, and turning
that probability into an approve/reject decision.

Built in ten slices, each finished and written up before the next starts. The
point of the project is not a high score — it is a model whose numbers can be
trusted, with every decision and every failure documented.

**Slices 1–5 are complete.** The model below is the honest baseline that slice 6
has to beat.

---

## Where it stands

| | |
|---|---|
| Test AUC | **0.6908**, 95% interval [0.6739, 0.7067] |
| Average precision | **0.3918** against a no-skill floor of 0.2365 — a 1.66× lift |
| Probabilities | corrected from **5.1 points low** to **0.3 points high** |
| At a 20% threshold | approve 47.9% of applications, bad rate **13.05%** [11.7%, 14.4%] vs 23.65% for approving everyone |

Measured on 5,378 loans issued 2016-09 to 2017-12, by a model trained only on
loans issued before 2016-01. `grade`, `sub_grade`, `int_rate` and `installment`
are deliberately held out — the model never sees Lending Club's own verdict on
the borrower.

### Calibration

A model can rank well and still output numbers that are false as probabilities.
This one under-stated risk by about 5 points everywhere. A sigmoid correction,
fitted on a validation split the model never trained on, fixes it:

![Calibration before and after](figures/slice05_calibration.png)

Each pair of points is the same group of borrowers. The correction never
reorders anyone — it only moves the label. Every one of the ten bins sat above
the diagonal before; afterwards they scatter around it.

### Risk against volume

![Risk against volume](figures/slice05_risk_vs_volume.png)

The trade is close to linear: every extra 10 points of approvals costs roughly
2 points of bad rate, the whole way across. There is no kink, so no operating
point can be read off the data — it can only come from what a default costs
against what a lost customer earns. That is slice 8.

---

## What this project is actually about

Most of the work is in not fooling yourself.

**Leakage.** The raw data has 151 columns, many of which only exist *after* a
loan goes bad — recoveries, total principal received, last payment date. Finding
and removing them is a slice of its own.

**Time, never random.** Every split is by issue date. The test set is a later
period, not a random sample, so the model is always judged on predicting
forward.

**Nothing learned from held-out data.** Medians, rare-level lists, scaling
factors, the regularisation strength, the calibration mapping — each is fitted
on training data only and applied unchanged.

**Intervals, not point estimates.** 5,378 test rows is not many. Bootstrap
intervals show that the third decimal place of an AUC is meaningless here, which
retroactively justifies several "these two options are tied" calls.

**Failures are written down, not cleaned up.** Four silent bugs and one data
blow-up are documented with what they looked like, why they produced plausible
output, and what caught them:

- a pandas index-alignment bug that scrambled one column into another borrower's
  values — no error, no warning
- a stale summary variable comparing two different models' scores
- parquet rows in arbitrary order, which would have made time-series
  cross-validation meaningless while still looking fine
- a deprecation warning that stated a conflict without saying which side won
- one loan with a `loan_to_income` ratio of 19.5 — eleven times beyond anything
  in training — that the model scored at probability 1.000000000000, and which
  was repaid

---

## The slices

| # | slice | status |
|---|---|---|
| 1 | Data into Postgres — SQL, loading, schemas | done |
| 2 | Clean and explore — missing data, EDA, leakage hunt | done |
| 3 | Feature engineering — encoding, selection bias, time splits | done |
| 4 | Baseline model — logistic regression, L1, collinearity | done |
| 5 | **Honest evaluation — PR curves, calibration, bootstrap intervals** | **done** |
| 6 | Strong model — gradient boosting, tuning, cross-validation | next |
| 7 | Explainability — SHAP, feature importance, fairness checks | planned |
| 8 | Business value — expected loss, approval thresholds, profit curve | planned |
| 9 | Deployment — API, Docker, dashboard | planned |
| 10 | Monitoring — drift detection, retraining, CI | planned |

Write-ups live in [`docs/`](docs/). They carry the reasoning, not just the
results:

- [`slice_03_feature_engineering.md`](docs/slice_03_feature_engineering.md) —
  why the delinquency features are dead (the lender already screened on them),
  why verified borrowers default *more*, and why geography stays out of the
  model but stays in the files
- [`slice_04_baseline_model.md`](docs/slice_04_baseline_model.md) — why a VIF in
  the hundreds means you have lost the right to read a coefficient, and why the
  threshold is not a modelling decision
- [`slice_05_honest_evaluation.md`](docs/slice_05_honest_evaluation.md) — why
  two of three evaluation metrics were blind to a prediction that was certain
  and wrong

---

## Layout

```
credit-risk-engine/
  data/raw/         downloaded CSV, never edited      (gitignored)
  data/processed/   files the code creates            (gitignored)
  notebooks/        02 explore, 03 features, 04 baseline, 05 evaluation
  src/              load_raw.py, columns.py, features.py
  models/           baseline_logreg.joblib, calibrated_logreg.joblib
  figures/          the plots this README uses
  docs/             one write-up per slice
```

The model bundle holds three fitted objects that only work in one order —
scaler, model, calibrator — so it also carries an `apply_order` key, the column
list, and a `predict_calibrated(bundle, X_raw)` function that defines the order
in code. It is verified by loading it cold, rebuilding from raw features, and
comparing all 5,378 probabilities, not a summary number.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The data is the Lending Club accepted-loans CSV from Kaggle (~2.2M rows). Put
the `.csv.gz` in `data/raw/`, set the Postgres connection details in a `.env`
file, then:

```bash
python src/load_raw.py
```

Then run the notebooks in order. Each one writes the files the next one reads.

## Known limitations

- Only finished loans are in the data, which biases recent vintages in both
  directions. The absolute bad rates are pessimistic; the relative movement
  between thresholds is sound.
- No borrower below FICO 660 exists — Lending Club rejected them. The model can
  say nothing about that population, and every variable the lender screened on
  hard looks weaker than it is.
- `loan_to_income` is unbounded. A cap is measured and documented but
  deliberately not applied, because the problem was found in the test set.
  Slice 9 applies it as an input guard.
- Feature distributions widen steadily across time. Not corrected. Slice 10.
