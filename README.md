# Credit Risk Engine

Predicting the probability that a Lending Club borrower defaults, and turning
that probability into an approve/reject decision.

Built in ten slices, each finished and written up before the next starts. The
point of the project is not a high score — it is a model whose numbers can be
trusted, with every decision and every failure documented.

**All ten slices are complete.** The model below is a logistic regression. Slice 6
tried to beat it with gradient boosting, did the work properly, and failed —
which is why it is still here. Slice 7 then found that the model it beat is
measurably fairer than it is. Slice 8 priced the decision in euros, and found
that the fairness gap slice 7 measured belonged to the threshold rather than to
the model. Slice 9 put it behind an API, and found that the model as saved
could not be served at all. Slice 10 watched it over two years of later
applications, and found that the one-number alarm stayed silent while a new
product arrived and was scored 11 points too high.

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
against what a lost customer earns. Slice 8 measures both.

---

## Does a stronger model help? No.

Gradient boosting usually wins on tabular data. Here it did not, and slice 6 is
the record of finding that out properly rather than assuming it either way.

LightGBM at its defaults lost on all five cross-validation folds. It was then
tuned across **44 configurations under a search declared in writing and
committed to git before it ran** — the grid, the stopping rule, what counts as
a win, and what happens if it loses. When the winner landed on a grid edge, the
one extension the declaration permitted was declared and run. It came back
worse, confirming the edge was a real optimum and not a wall.

| | CV AUC | in-sample | gap |
|---|---|---|---|
| LightGBM, defaults | 0.6830 | 0.9304 | 0.2474 |
| LightGBM, tuned | 0.7053 | 0.7249 | 0.0196 |
| **logistic (ships)** | **0.7107** | — | — |

Tuning worked — the overfitting gap fell by a factor of twelve, leaving the
boosted model generalising as cleanly as the linear one. It still lost.

![Paired bootstrap of the AUC difference](figures/slice06_paired_bootstrap.png)

On the test set, scored on identical rows and compared with a paired bootstrap:
**−0.0182 AUC, 95% interval [−0.0257, −0.0105]**. The interval excludes zero, so
the loss survives sampling luck. Average precision, Brier and log loss are all
indistinguishable between the two models. Worse at ranking, no better anywhere.

Two things fall out of the attempt that are worth more than the result:

**The boosted model cannot blow up.** The logistic model scores one test loan at
probability 1.000000 — a borrower whose stated income makes the loan 19.5× their
annual earnings, eleven times past anything in training, and the loan was repaid.
That single row is 1.2% of the model's entire test log loss. A tree cannot make
that mistake: above its highest split there is nothing left to say, so the same
borrower scores 0.25 instead of 1.00. Demonstrated on a synthetic row, so no
held-out data was spent proving it.

**Regularisation and a linear model converge.** Tuned down to four leaves, the
boosted model lands within 0.005 of logistic regression, with a near-identical
generalisation gap. A one-split-per-tree stump — no interactions representable
at all — reaches 0.7010. Three very different levels of flexibility, one answer.
The usable signal in this data is close to additive, and slice 3 already
explained why.

---

## Four columns of Lending Club's verdict match seventy-one of mine

`grade`, `sub_grade`, `int_rate` and `installment` were held out from slice 3
onward, for one comparison at the end of slice 6.

![The verdict ladder](figures/slice06_verdict_ladder.png)

| | features | CV AUC |
|---|---|---|
| Lending Club's verdict alone | 4 | 0.7029 |
| This project's feature set | 71 | 0.7107 |
| Both together | 75 | **0.7235** |

Alone, the two are a statistical tie — professional underwriters compressed the
same predictive power into a single letter. But they are **not redundant**:
together they beat either one, each on five folds out of five.

That gap is a measurement of something slice 3 could only argue. Lending Club
assigned `grade` while looking at the **whole applicant pool, including everyone
they rejected**. This dataset contains only the people who got through, which is
why FICO scores just 1.8× here and the entire delinquency family is flat. The
+0.0128 that `grade` adds on top of 71 engineered features is a lower bound on
how much predictive information was destroyed before the data was written.

**And the shipped model still cannot use it.** Not because it is leakage — all
four columns are known before funding — but because of *when*. An investor
picking among already-listed loans can see the grade. A lender deciding whether
to approve cannot: assigning the grade **is** the decision. This project is the
lender's seat, so the most accurate model built in slice 6 is one that may not
be used.

---

## What a decision is worth

Slice 8 turns the probability into euros. Every economic parameter is measured
from realised cash flows on **training loans only**, written to
`models/economics.json`, and only then applied.

The cash-flow columns slice 2 stripped out of the modelling table were never
removed from the raw sample, so loss given default did not have to be assumed:

| parameter | measured on train | varies with risk? |
|---|---|---|
| Loss given default | **0.8973** | **no** — 0.889 to 0.905 across all seven grades |
| Exposure at default | 0.651 | **yes** — 0.529 for A, 0.790 for F |
| Margin on a repaid loan | +0.182 | **yes** — +0.096 for A, +0.422 for G |

Riskier borrowers default *earlier*, so more principal is still outstanding when
they go. Recovery on that principal is about ten cents in the euro and does not
care who the borrower was.

### One threshold is the wrong shape

![Break-even by grade](figures/slice08_break_even_by_grade.png)

Approve while `p < margin / (margin + loss)`. Because loss per euro is nearly
flat while margin triples from A to G, the break-even probability triples too —
**0.185 to 0.567**. Slice 5's single 0.20 threshold is about right for grade A
and far too strict for everything below it.

### Choosing the threshold without looking at an outcome costs 4,000 euros

| | |
|---|---|
| threshold chosen on predicted profit | 0.43 |
| threshold that was best on true outcomes | 0.45 |
| profit captured by the blind choice | **99.89%** |

This is the money version of the calibration curve above, and it is the clearest
argument in the project for having calibrated at all. An uncalibrated score can
be ranked; it cannot be put into a break-even formula and come out right.

### The provably optimal rule loses anyway

Approving exactly the loans with positive expected profit **is** the maximum of
the predicted-profit curve — no global threshold can beat it there, and none
does (+219k). On the true outcomes it loses by 68k, because the model
over-predicts risk by 5.8 points in one disagreement set and under-predicts by
8.0 in the other. A global threshold only needs the ranking right near one cut
point. The per-loan rule needs the probability right everywhere, and spends its
entire edge in the tails where calibration is weakest.

Kept as a negative result, with a testable prediction attached: improve tail
calibration and the ordering should flip.

### There is no one right threshold

![Sensitivity surface](figures/slice08_sensitivity.png)

Across loss-given-default from 0.70 to 1.00 and margin from 0.6× to 1.4×, the
best threshold runs **0.21 to 0.68**. The margin assumption moves it more than
the loss assumption — and margin is the parameter measured from a book that
prepaid early, so it is also the one most likely to be wrong next vintage.

### The fairness gap belonged to the threshold

![Fairness frontier](figures/slice08_fairness_frontier.png)

| threshold | thin-file approval | thick-file | gap |
|---|---|---|---|
| 0.18 (worst) | 27.7% | 42.5% | −14.7 pts |
| 0.20 (slice 5) | 36.7% | 50.1% | −13.4 pts |
| **0.43 (profit-maximising)** | **90.6%** | **89.9%** | **+0.7 pts** |

Slice 7's 13.4 point access gap reproduces exactly at the threshold where it was
measured — and vanishes at the threshold slice 8 chose for entirely unrelated
reasons. Forcing equal access at 0.20 instead does not cost money, it **makes
39k more**, because 0.20 is too strict for everyone and the thin-file group is
where the strictness lands hardest.

That alignment is specific to this book at this threshold and is not a general
result — the two objectives genuinely conflict whenever base rates differ, which
slice 7 proved for this data. What can be said is that the gap this project
worried about was a symptom of a badly chosen operating point, not a price that
had to be paid.

### What the euro totals are not

![Three profit curves](figures/slice08_profit_curves.png)

Realised cash on the test book is negative at every threshold, and that is an
artifact. Slice 3's censoring dropped unresolved loans, which bites harder the
later the vintage: repaid loans ran 26.1 months in train and **11.9 in test**, so
the test book's good loans are a sample of fast prepayers that were cut off
before they had paid their interest.

So profit is computed three ways, not two — predicted, true-outcome-with-train-
economics, and fully realised — which separates calibration error from the
censoring distortion instead of blending them into one misleading number.
Relative comparisons between thresholds are sound. The euro totals are not a
forecast.

---

## Serving it

![The dashboard](figures/slice09_dashboard.png)

The screenshot is the "Stretched" example: a 35,000 loan over 60 months on an
income of 42,000. Declined at 91.6%, with the four factors that drove it.

To try it yourself you only need Docker. No Python, no data:

```bash
git clone https://github.com/Mehdi-mo/credit-risk-engine.git
cd credit-risk-engine
docker build -t credit-risk-engine .
docker run --rm -p 8000:8000 credit-risk-engine
```

Then open http://localhost:8000 and click the example buttons. Without Docker,
from a Python environment with `requirements-service.txt` installed:

```bash
uvicorn service.app:app --port 8000
```

A loan application goes in as JSON. The decision, the probability, the
threshold it was compared against, the reasons, and the version of every file
that produced them come out.

### The model could not be served

`predict_calibrated` takes 71 engineered features. An application is an income,
a loan amount and a credit file. Turning one into the other needs 59 medians,
four category level lists and a choice of which columns get a log — and none of
them had ever been saved. They existed as variables while notebook 03 ran. The
bundle held a scaler, a model and a calibrator, and nothing outside that
notebook could build their input.

Slice 9 writes them to `models/feature_spec.json` and then proves the record is
right by rebuilding all 26,280 saved feature rows from raw through the function
the service calls. Notebook 03's own rebuild check had compared shapes and
column names; a wrong median would have passed it.

### "Reproduces training exactly" has three meanings

The skew test feeds every raw test row to the service as the JSON a caller
would send, and demands the saved features and the published probabilities
back.

| the same application, scored… | difference |
|---|---|
| as a batch, on the machine that wrote the files | **0.0** |
| by the same code on another operating system | 2.2e-16 in the probability, 1.8e-15 in one feature |
| on its own, instead of inside a batch | up to 3.3e-16, on 27.5% of rows |

The second and third are the maths library rounding the last bit differently.
Neither can move a decision — the closest any test probability comes to the
threshold is 2.8e-4 — but that was measured, not assumed, and the tolerance is
a documented setting whose default is exact.

### The service holds no second copy of anything

`src/decide.py` is the route from application to decision and computes nothing
itself: contract, feature build, scoring chain, economics and reasons are each
another file's function, the same ones the notebooks and scripts use. One test
reads the service's source and fails if a step of the pipeline has been retyped
there.

### What happens to a bad application

Decided and written down before the code
([`slice_09_input_contract.md`](docs/slice_09_input_contract.md)), from one
rule: the service may do to an application only what training did to a training
row, and nothing is done silently.

| | |
|---|---|
| blank where training had blanks (10 fields) | filled as training filled it, and flagged |
| blank anywhere else | rejected |
| impossible value — a negative income, a term of 48 | rejected |
| possible, but outside what the model saw | scored, flagged, review recommended |
| a category the model never saw | rejected — pandas would silently score it as the base level |

None of the 5,378 test applications is rejected. 117 are flagged for asking for
more than 35,000, the largest loan in training: in the test window loans go up
to 40,000. The maximum moved after training ended, and the contract caught it
as data.

### The guard changed two probabilities and no decisions

The `loan_to_income` cap slice 5 measured is now on. Log loss goes 0.5131 →
0.5076, the largest probability 1.0000 → 0.9823, and the AUC does not move. Not
one of 5,378 decisions changes, at 0.20, at 0.43 or under the per-loan rule. It
is a calibration fix, not a decision fix.

### Three declines in four have the same main reason

![What the declines are made of](figures/slice09_decline_reasons.png)

`POST /reasons` returns the factors that raised and lowered an applicant's risk,
carried through the calibrator so they add up exactly to the probability in the
same response. Over the 536 declined test applications, the main reason is the
60-month term in 75%. The policy the threshold produces is close to "a 60-month
loan needs something else going for it" — true to the model, and not yet a
useful thing to tell an applicant.

The service ships slice 8's threshold of 0.43, and never without its
assumption: it is right only if the measured margin is, and every response says
so.

---

## Watching it

Slice 10 replays every application from 2016 and 2017 through the service, in
date order, as if it had been running since training ended. The alert
thresholds were written down before the replay was run.

![Drift over time](figures/slice10_drift.png)

### The summary alarm never rang

Over eight quarters the PSI of the score stays below 0.02, against a warning
level of 0.10, and the approval rate stays between 86% and 92%. A dashboard
with those two lines would have been green for two years.

Underneath, joint applications went from 10 of 16,134 training loans to about
one application in nine.

| | live applications | predicted | observed | declined |
|---|---|---|---|---|
| individual | 9,760 | 23.3% | 23.6% | 8.6% |
| joint | 386 | **39.4%** | **28.0%** | **38.9%** |

The per-feature PSI raised the alert (0.625). About half of the gap is a weight
learned from ten loans. The rest is most likely that the model scores a
two-income application on one income: the joint income is in the raw data and
was never used. Found on later data, so it is written down and not repaired.

### Outcomes come too late to monitor on

| months after a quarter ends | outcomes known | bad rate a monitor sees | in the end |
|---|---|---|---|
| 6 | 21.5% | 7.5% | 23.8% |
| 12 | 46.3% | 17.5% | 23.8% |
| 24 | 87.1% | 24.1% | 23.8% |

Early repayers are known at once; a default is booked months after the last
payment. For the first year a monitor would conclude the book is far safer
than it is.

### Retraining on schedule would have made it worse

The fit was moved out of the notebook into one function, which rebuilds the
shipped model to 15 decimal places. Then a walk-forward test from 2017-04-01:

| | AUC on the next 9 months | predicted | observed |
|---|---|---|---|
| shipped model | 0.6970 | 24.2% | 23.3% |
| retrained on outcomes known that day | 0.6958 | **15.6%** | 23.3% |
| retrained with hindsight | 0.6974 | 25.0% | 23.3% |

With perfect hindsight, retraining buys nothing. Done the way it would really
have been done, it learns from the early repayers that defaults are rare and
loses its calibration. So the policy triggers a review by a person, not an
automatic retrain.

Every push runs the tests and builds the Docker image in GitHub Actions.

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

**Decisions declared before the results exist.** Slice 6's tuning search was
written down and committed to git before it ran, including what would count as a
win and what would happen if nothing did. In a finished notebook, "the model
lost" and "I stopped looking when I was ahead" are indistinguishable. A
timestamp is the only thing that separates them.

**Intervals, not point estimates.** 5,378 test rows is not many. Bootstrap
intervals show that the third decimal place of an AUC is meaningless here, which
retroactively justifies several "these two options are tied" calls. Where two
models can be scored on the same rows, the comparison is paired — and how much
that narrows the interval turns out to depend on how similar the two models are
(5.9× for two logistic variants, 2.2× for logistic against boosted trees).

**Failures are written down, not cleaned up.** Five silent bugs, one data
blow-up and one flawed rule of my own are documented with what they looked like,
why they produced plausible output, and what caught them:

- a pandas index-alignment bug that scrambled one column into another borrower's
  values — no error, no warning
- a stale summary variable comparing two different models' scores
- parquet rows in arbitrary order, which would have made time-series
  cross-validation meaningless while still looking fine
- a deprecation warning that stated a conflict without saying which side won
- one loan with a `loan_to_income` ratio of 19.5 — eleven times beyond anything
  in training — that the model scored at probability 1.000000000000, and which
  was repaid
- a tie-break rule I wrote myself whose tolerance came from the best entry's own
  noise, so the band widened exactly when it should have tightened, and selected
  an unstable model

---

## The slices

| # | slice | status |
|---|---|---|
| 1 | Data into Postgres — SQL, loading, schemas | done |
| 2 | Clean and explore — missing data, EDA, leakage hunt | done |
| 3 | Feature engineering — encoding, selection bias, time splits | done |
| 4 | Baseline model — logistic regression, L1, collinearity | done |
| 5 | Honest evaluation — PR curves, calibration, bootstrap intervals | done |
| 6 | **Strong model — gradient boosting, pre-registered tuning, paired tests** | **done** |
| 7 | Explainability — SHAP, feature importance, fairness checks | done |
| 8 | **Business value — measured LGD, per-loan break-even, profit curves, fairness frontier** | **done** |
| 9 | **Deployment — input contract, train/serve skew test, reasons endpoint, Docker, dashboard** | **done** |
| 10 | **Monitoring — request log, drift replay, label delay, walk-forward retraining, CI** | **done** |

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
- [`slice_06_strong_model.md`](docs/slice_06_strong_model.md) — the declared
  search, why the win condition was not met, why two completely different models
  leave the identical residual curve after calibration, and why the most
  accurate model built cannot be used
- [`slice_07_explainability.md`](docs/slice_07_explainability.md) — why
  withholding geography worked and withholding credit history did not, and three
  SHAP calls that silently moved the reference point while passing their own
  checks
- [`slice_08_business_value.md`](docs/slice_08_business_value.md) — why loss
  given default did not have to be assumed, why two profit curves were not
  enough, and why the provably optimal decision rule still loses
- [`slice_09_deployment.md`](docs/slice_09_deployment.md) — why the saved model
  could not be served, what "reproduces exactly" turned out to mean, and why a
  guard that fixed the worst prediction in the project changed no decision
- [`slice_09_input_contract.md`](docs/slice_09_input_contract.md) — reject,
  impute or score with a flag, decided field by field before any code
- [`slice_10_monitoring.md`](docs/slice_10_monitoring.md) — why a stable score
  hid a new product, why early outcomes lie, and why retraining on what was
  known on the day breaks the model

---

## Layout

```
credit-risk-engine/
  data/raw/         downloaded CSV, never edited      (gitignored)
  data/processed/   files the code creates            (gitignored)
  notebooks/        02 explore, 03 features, 04 baseline,
                    05 evaluation, 06 strong model, 07 explainability
  src/              load_raw.py, columns.py, features.py, predict.py,
                    money.py, economics.py, profit.py,
                    contract.py, decide.py, reasons.py, feature_spec.py,
                    score_test.py, guard_effect.py, service_report.py,
                    tolerance.py, requestlog.py, monitor.py, retrain.py
  service/          app.py (the API), dashboard.html
  tests/            test_skew.py, test_contract.py, test_service.py,
                    test_monitor.py
  models/           baseline_logreg.joblib, calibrated_logreg.joblib,
                    calibrated_lgbm.joblib, economics.json,
                    feature_spec.json, policy.json, monitor_policy.json
  .github/          the CI workflow
  figures/          the plots this README uses
  docs/             one write-up per slice
  Dockerfile        the service, with no data in the image
```

The shipped bundle holds three fitted objects that only work in one order —
scaler, model, calibrator — so it also carries an `apply_order` key, the column
list, and a `predict_calibrated(bundle, X_raw)` function that defines the order
in code. It is verified by loading it cold, rebuilding from raw features, and
comparing all 5,378 probabilities, not a summary number.

`calibrated_lgbm.joblib` is the losing model, kept so the comparison is
reproducible. It holds two fitted objects rather than three — a tree splits on
thresholds, so scaling it is a no-op — and records in the file itself that it is
not the model that ships.

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

Slices 7 and 8 run from `src/` instead, each asserting its own inputs:

```bash
python -m src.predict     # reproduces every published probability, cold
python -m src.money       # price and realised cash per loan
python -m src.economics   # fits the economics on train, writes economics.json
python -m src.profit      # curves, sensitivity, fairness, five figures
```

Slice 9 adds the service and its checks:

```bash
pip install -r requirements-service.txt   # fastapi, uvicorn, pytest, httpx

python -m src.score_test       # guard on; writes predictions_test_slice9.parquet
python -m src.feature_spec     # writes feature_spec.json, proves the rebuild
python -m src.guard_effect     # what the guard changed downstream
python -m src.service_report   # the service on the whole test book
python -m pytest tests -q      # 127 tests, the skew test among them

uvicorn service.app:app --port 8000

docker build -t credit-risk-engine .
docker run --rm -p 8000:8000 credit-risk-engine
```

Slice 10 adds the monitor and the retraining test:

```bash
python -m src.monitor          # replays 2013-2017, tables and two figures
python -m src.retrain          # rebuilds the shipped model, then walks forward
```

## Known limitations

- Only finished loans are in the data, which biases recent vintages in both
  directions. The absolute bad rates are pessimistic; the relative movement
  between thresholds is sound.
- No borrower below FICO 660 exists — Lending Club rejected them. The model can
  say nothing about that population, and every variable the lender screened on
  hard looks weaker than it is. Slice 6 puts a lower bound on the cost of this.
- `loan_to_income` was unbounded until slice 9, which applies the cap slice 5
  measured as an input guard. The headline table above is the unguarded model
  as slice 5 published it; with the guard on, average precision is 0.3925 and
  log loss 0.5076, and no decision changes.
- The service scores an application outside the training range and flags it. A
  flag is "the model is guessing", not "this applicant is risky".
- The reasons say what the model weighed against a typical borrower, not what
  would change the answer.
- Feature distributions drift steadily across time. Slice 10 measures it; the
  score stays stable because the shifts cancel, which is not the same as safe.
- Absolute profit is not a real number. Censoring makes the euro totals a
  statement about a snapshot, not a forecast. Slice 8 measures the size of that
  distortion rather than hiding it.
- Only funded loans exist, so slice 8's 90% approval rate is 90% of a pool
  Lending Club had already screened. Nothing here speaks to applicants it turned
  away.
- Joint applications are scored on one applicant's income and are
  over-predicted by about 11 points. Slice 10 found it and did not repair it.
- The monitoring replay only holds loans that had finished by the snapshot,
  about half of the 2016-2017 applications.
- The test set has now been used seven times, each time for reporting rather
  than selection.
