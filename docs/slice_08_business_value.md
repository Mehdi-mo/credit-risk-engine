# Slice 8 — business value

Input: `data/processed/sample_50k.parquet` (the raw 151 columns, still intact),
`model_table.parquet`, `predictions_test_slice6.parquet`,
`fairness_groups_test.parquet`.

Output: `src/money.py`, `src/economics.py`, `src/profit.py`,
`models/economics.json`, four processed files, five figures, and three findings
that each contradict something the slice set out to do.

**Loss given default did not have to be assumed — the cash-flow columns slice 2
removed from the modelling table were never removed from the raw sample, so LGD
is 0.8973 measured, flat across every grade. The predicted-versus-realised check
slice 8 was built around is confounded by vintage censoring and had to be split
into three curves before it meant anything. And the per-loan break-even rule,
which is provably the best possible rule under the model's own probabilities,
loses to a single tuned threshold on the true outcomes — because it leans hardest
exactly where the model is least calibrated.**

## What this slice adds

Slices 4 to 7 asked whether the model works, why it says what it says, and who
pays when it is wrong. This one asks what a decision is worth, and it is the
first slice where the answer is denominated in euros rather than in AUC.

Three things have to be true before that question can be asked honestly:

- the economics have to be written down **before** any result is computed, so
  the numbers cannot be chosen to flatter the conclusion
- every parameter has to be learned from **train rows only**, like every other
  learned quantity in seven previous slices
- the answer has to be reported as a **range across the assumption**, not as a
  single number, because the assumption is the weakest part of the chain

## Step 0 — the money came back, and so did a bug

The slice plan said loss given default and exposure at default could only be
assumed, because `recoveries`, `total_rec_prncp`, `last_pymnt_d` and the rest of
the leakage family were removed in slice 2.

They were removed from the **modelling table**. `sample_50k.parquet` still holds
all 151 raw columns, including every cash-flow field and `int_rate` and
`installment` as well. Using them for the economics is not leakage: it is the
same outcome information the target is built from, used for evaluation and never
as a feature. Using the price for the economics while excluding it from the
model is a decision, recorded here rather than left implicit — a lender setting
the price knows the price, even when the scorecard is not allowed to.

### Matching the rows: position, not a join

The first attempt joined on eight columns — `issue_d`, `zip_code`, `addr_state`,
`loan_amnt`, `term`, `annual_inc`, `dti`, `revol_bal`. Both sides were unique on
that key and it still lost 4 of 5,378 rows, because slice 2 nulled 11 `dti`
values the raw sample still carries.

The positional path loses nothing. `sample_50k` filtered to the three resolved
statuses and reindexed is `model_table` row for row:

| check | result |
|---|---|
| resolved-status rows vs `model_table` | 29,619 = 29,619 |
| `loan_amnt`, `annual_inc`, `revol_bal`, `fico_range_low`, `total_acc`, `open_acc`, `delinq_2yrs` | agree positionally at 1.0000 |
| `target` rebuilt from `loan_status` | 1.0000 |
| after slice 3's 2013–2017 window | 26,280 |
| `issue_d.quantile(0.8)` | 2016-09-01, the outer split |
| splits reproduced | 16,134 / 4,768 / 5,378, bad rates 0.1890 / 0.2391 / 0.2365 |

`src/money.py` carries all six of those counts as named constants and asserts
each one, so a silent change upstream fails loudly instead of producing a
plausible wrong file. The raw `id` comes forward as a real key.

### The bug the identity caught

The first version of `net_cash` was
`total_pymnt + recoveries − collection_recovery_fee − funded_amnt`. That is
wrong: `total_pymnt` **already contains** `recoveries`. Checked across all
26,280 rows, `total_pymnt = prncp + int + late + recoveries` holds exactly apart
from one rounding cent.

| | double-counted | corrected |
|---|---|---|
| train bad-loan net, share of funded | −0.3146 | **−0.3936** |
| defaults that still returned a profit | 628 | 350 |
| loans returning more than 100% after charging off | 1 | **0** |

The last row is how the error was identified rather than argued about. A loan
cannot charge off and return 108% of principal; that impossible row was the
double count showing through, and it disappears rather than merely shrinking.
Good loans were never affected — all 20,818 of them have zero recoveries.

Two asserts now stand on this: one that `total_pymnt` really is the sum of its
four parts, and one that the loss decomposition below holds to 1e-6.

## Step 1 — the economics, written down before anything was computed

A repaid loan returns `margin_pct × funded`. A defaulted loan loses
`(ead_pct × LGD − int_pct) × funded` — interest already collected is not a loss.
That second expression is an identity, not a fit:

```
net_pct  =  int_pct  −  ead_pct × LGD
```

verified to 1e-6 on every defaulted loan. It is written in this form on purpose,
so **LGD stays a single knob** the sensitivity analysis can turn. A directly
measured net loss would be one number with nothing to vary.

All four parameters, fitted on the 16,134 train rows and saved to
`models/economics.json` before a single curve was drawn:

| parameter | value | measured or assumed |
|---|---|---|
| `margin_pct` | −0.00837 + 0.01387 × rate + 0.0666 × is60 | measured, R² 0.461, n 13,084 |
| `ead_pct` | +0.46612 + 0.00773 × rate + 0.1384 × is60 | measured, R² 0.147, n 3,050 |
| `int_pct` | +0.00042 + 0.01039 × rate + 0.0618 × is60 | measured, R² 0.294, n 3,050 |
| `LGD` | 0.8973, flat | measured, n 3,050 |

### LGD is flat. EAD is not.

The expectation going in was that loss would vary with risk. It does, but not
where expected. Across all seven grades LGD sits between **0.889 and 0.905**;
across term, 0.891 against 0.905; across interest-rate quintiles, 0.887 to
0.902. Standard deviation across all fourteen grade × term cells is 0.0153.
Recovery on defaulted principal is about ten cents in the euro and it does not
care who the borrower was.

**Exposure** is what moves: 0.529 for grade A against 0.790 for grade F, and
0.574 for 36-month loans against 0.741 for 60-month ones. Riskier borrowers
default *earlier*, so more principal is still outstanding when they go. The
relationship is real; it just lives in a different term of the product than
expected.

### Why straight lines instead of cell means

Cell means by grade × term were the alternative and were rejected: G/36-month
has 14 train loans and A/60-month has 60, so those cells are noise. Against the
fourteen cells that do have data, the fitted lines are off by **0.0078** on
margin and **0.0264** on loss, on average, and the cells they fit worst are the
thin ones. The sensitivity surface also needs something smooth to move.

### Prepayment, and why the formula was not used

Scheduled interest — `int_rate × term` — overstates what arrives, because these
borrowers prepay. Train 36-month loans that repaid ran **25.7 months of 36**.
The measured margin already contains that. Measurement beating the formula is a
finding, not a footnote: anyone pricing this book from the coupon would be wrong
by roughly a third of the term.

## Step 2 — the threshold comes out of the arithmetic

Approve while the expected gain still covers the expected loss:

```
(1 − p) × margin  =  p × loss      →      p* = margin / (margin + loss)
```

Per loan, because the price is per loan. A single global threshold is the
special case where every loan is priced the same, and they are not:

| grade | margin | loss | break-even p |
|---|---|---|---|
| A | 0.090 | 0.395 | **0.185** |
| B | 0.147 | 0.388 | 0.273 |
| C | 0.207 | 0.386 | 0.347 |
| D | 0.275 | 0.374 | 0.422 |
| E | 0.364 | 0.365 | 0.499 |
| F | 0.447 | 0.364 | 0.551 |
| G | 0.468 | 0.357 | **0.567** |

Loss per euro lent is nearly flat — higher exposure is offset by the higher
interest collected before default. Margin triples. So the break-even point
triples with it, from 0.185 to 0.567, and slice 5's single 0.20 threshold turns
out to be roughly right for grade A and far too strict for everything below it.

Figure: `figures/slice08_break_even_by_grade.png`

## Step 3 — three curves, because two would have lied

The plan was to compute profit two ways, predicted and realised, and read
agreement as confirmation that calibration holds in money. On this data that
comparison is confounded.

Slice 3's censoring dropped loans that had not yet resolved, and it bites harder
the later the vintage:

| split | good loan returns | months on book | bad loan returns | EAD |
|---|---|---|---|---|
| train (2013–15) | +0.1822 | 26.1 | −0.3936 | 0.651 |
| valid (2016 H1) | +0.1388 | 20.6 | −0.4832 | 0.700 |
| **test (2016 H2 – 17)** | **+0.1060** | **11.9** | **−0.6109** | **0.792** |

The test book's repaid loans are not a sample of repaid loans. They are a sample
of **fast prepayers**, because nothing else could finish before the data was
cut. Its defaults died at 10.9 months with 79% of principal outstanding.

So the gap between predicted and realised money holds two different errors at
once, and three curves separate them, all on the same approved loans:

| curve | what it multiplies | what it is for |
|---|---|---|
| **A** predicted | model probability × train economics | **choosing**; touches no outcome |
| **B** business | true outcome × train economics | **scoring**; the money measure |
| **C** censored | true outcome × this book's own cash | **diagnosis** only |

A against B is calibration error: identical economics both sides, only the
probability differs from truth. B against C is the censoring distortion:
identical outcomes both sides, only averaged euros differ from actual ones.

| rule | approval | bad rate | A | B | C |
|---|---|---|---|---|---|
| approve everyone | 100.0% | 0.2365 | 3,380k | 3,486k | −6,062k |
| slice 5's 0.20 | 47.9% | 0.1305 | 2,341k | 2,217k | −184k |
| global 0.43, chosen on A | 90.0% | 0.2082 | 4,087k | **3,906k** | −3,398k |
| per-loan break-even | 85.7% | 0.2117 | **4,306k** | 3,838k | −3,407k |
| global 0.45, B optimum (hindsight) | 91.4% | 0.2112 | 4,071k | 3,910k | −3,664k |

Curve C is negative everywhere and falls monotonically, which would say "approve
almost nobody". That is an artifact of the snapshot, not advice. The test book
genuinely did lose money — its good loans were cut off before they had paid
their interest — and that is a fact about when the data was collected, not about
the policy.

Figure: `figures/slice08_profit_curves.png`

## Step 4 — calibration, measured in money

The threshold is chosen on curve A, which never sees an outcome. Then curve B
says what that choice was worth:

| | |
|---|---|
| threshold chosen on A | 0.43 |
| threshold that was best on B | 0.45 |
| B profit at the chosen threshold | 3,906k of a possible 3,910k |
| **captured** | **99.89%** |

Choosing the threshold without looking at a single outcome cost **4,000 euros
out of 3.9 million**. This is the money version of slice 5's calibration curve,
and it passes. It is also the strongest practical argument in the project for
having calibrated the model at all: an uncalibrated score can be ranked, but it
cannot be put into `p*` and have the arithmetic come out right.

## Step 5 — the per-loan rule is provably optimal, and still loses

Approving exactly the loans whose expected profit is positive **is** the maximum
of curve A. No global threshold can beat it there, and none does: 4,306k against
4,087k, +219k.

On curve B it loses, 3,838k against 3,906k, −68k.

The two rules disagree about 475 of 5,378 loans, and the model is wrong about
both groups in the direction that hurts:

| disagreement | n | predicted | actual | error |
|---|---|---|---|---|
| global approves, per-loan rejects | 353 | 0.3412 | 0.2833 | **+0.058** |
| per-loan approves, global rejects | 122 | 0.4773 | 0.5574 | **−0.080** |

In the first group the model over-predicts risk — grade A loans in that set have
a mean predicted probability of 0.242 against an actual default rate of 0.078 —
so the per-loan rule rejects borrowers who were fine. In the second it
under-predicts, so the rule approves borrowers who were not.

**A global threshold only needs the ranking to be right near one cut point. The
per-loan rule needs the probability itself to be right everywhere, and it spends
its entire theoretical edge in the two tails where calibration is weakest.** The
decile table confirms where that is: the top decile predicts 0.540 against an
actual 0.493, and the middle deciles under-predict by two to four points.

This is a negative result and it is kept. The per-loan rule is the better
*theory* and the worse *policy* on this model, and the reason is measurable
rather than rhetorical. If slice 9 or 10 improves calibration in the tails, the
ordering should flip — which makes this a testable prediction rather than an
excuse.

## Step 6 — the deliverable is a surface, not a number

Best global threshold across a grid of LGD from 0.70 to 1.00 and a margin
multiplier from 0.6× to 1.4×, with the optimum always taken on curve A:

| moving | threshold range |
|---|---|
| the whole grid | **0.21 to 0.68** |
| LGD alone, margin at measured | 0.38 to 0.60 |
| margin alone, LGD at 0.90 | 0.24 to 0.55 |

There is no one right threshold. The **margin** assumption moves the answer more
than the loss assumption does, which is uncomfortable in the right way: margin
is the parameter measured from a book that prepaid early, so it is also the one
most likely to be wrong on a fresh vintage. The loss assumption, the one the
slice plan expected to be fragile, turns out to be the sturdier of the two.

Figure: `figures/slice08_sensitivity.png`

## Step 7 — does the money agree with slice 6?

Slice 6 chose the logistic model on AUC, by a paired-bootstrap difference of
−0.0182 [−0.0257, −0.0105] against LightGBM. Run both through the same curve:

| model | threshold chosen on A | B profit | approval |
|---|---|---|---|
| logistic (shipped) | 0.43 | **3,906k** | 90.0% |
| LightGBM (slice 6) | 0.44 | 3,721k | 91.5% |

The money agrees. Boosting does not win here either, and it loses by about 185k
on a 3.9M book — a 4.7% difference, larger in relative terms than the AUC gap
suggested. Nothing in this slice reopens slice 6's verdict.

Figure: `figures/slice08_model_comparison.png`

## Step 8 — the fairness frontier, and a result that went the other way

Slice 7 found a 13.4 point approval gap against thin-file borrowers at slice 5's
0.20 threshold — 36.7% against 50.1% — caused by `mo_sin_old_rev_tl_op`, which
correlates 0.917 with the credit-history variable that was deliberately left
out.

Measured across the whole threshold range, that gap is not a fixed property of
the model:

| threshold | thin-file approval | thick-file approval | gap |
|---|---|---|---|
| 0.18 (worst) | 27.7% | 42.5% | **−14.7 pts** |
| 0.20 (slice 5) | 36.7% | 50.1% | −13.4 pts |
| **0.43 (profit-maximising)** | **90.6%** | **89.9%** | **+0.7 pts** |

The gap is a property of **strictness**. It peaks at 0.18, and the
profit-maximising threshold is loose enough that it closes on its own — slightly
in the thin-file group's favour. Slice 7's finding reproduces exactly at the
threshold where it was measured, and does not survive the threshold this slice
chose for entirely unrelated reasons.

Forcing equal access at the strict 0.20 threshold instead, by raising the
thin-file group's own threshold to 0.230:

| | |
|---|---|
| thin-file approval | 36.7% → 50.1% |
| B profit | **rises by 39k, +1.75%** |

Equal access is not bought here, it is **found**. At 0.20 the policy is too
strict for everyone, and the thin-file group is simply where the strictness
lands hardest, so relaxing it for that group moves money in the same direction
as fairness. That alignment is specific to this book at this threshold and
should not be generalised — the two objectives genuinely conflict whenever base
rates differ, which slice 7 already proved for this data (28.4% against 22.7%).
What can be said is that **the gap this project worried about was a symptom of a
badly chosen operating point, not a cost that had to be paid.**

Figure: `figures/slice08_fairness_frontier.png`

## What this data cannot support

- **Only funded loans exist.** Every threshold here is evaluated inside the
  population Lending Club already approved. Nothing in this slice says anything
  about applicants it turned away, and the 90% approval rate is 90% of an
  already-screened pool.
- **Absolute profit is not a real number.** The censoring in step 3 makes it a
  statement about a snapshot. Relative comparisons between thresholds are sound;
  the euro totals are not a forecast.
- **Curve B is a counterfactual.** It asks what these loans would have been
  worth had they behaved like train loans. That is the right question for
  comparing policies and the wrong one for forecasting a return.
- **The 2 loans with `out_prncp > 0`** have cash that is not final. Left in,
  documented, too small to matter.

## Rules kept

| rule | how |
|---|---|
| no decision made on test | threshold chosen on curve A, which uses no outcome; the B optimum is reported and labelled hindsight |
| anything found on test is documented, not fixed | the censoring in step 3, the calibration failure in step 5 |
| rows in `issue_d` order before time-based work | asserted in `money.py` and again in `profit.py` |
| learned numbers come from train only | asserted: the economics fit refuses a frame containing a row dated 2016 or later |
| assumptions written before results | `models/economics.json` is written by step 1 and read by step 3 |
| results reported as a range | step 6 |

The test set has now been used five times, every time for reporting.

## Known debt carried into slice 9

- `loan_to_income` is still unbounded; the guard value 1.782 sits in both
  bundles and `src/predict.py` still has `apply_guard=False`. Slice 9 turns it
  on, and should re-run `src/profit.py` afterwards, because 2 rows move and both
  are near the threshold.
- Slice 7's interaction values still use a path-dependent reference. Untouched.
- The margin parameter is the weakest number in this slice and the one the
  answer is most sensitive to. Slice 10's monitoring should watch realised
  margin per vintage, not just score drift.
- The per-loan rule should be re-tested whenever calibration changes. If tail
  calibration improves, it should overtake the global threshold; if it does not,
  that prediction was wrong and the reasoning in step 5 needs revisiting.

## Reproducing

```
python -m src.money       # builds data/processed/money_table.parquet
python -m src.economics   # fits and writes models/economics.json
python -m src.profit      # curves, sensitivity, fairness, five figures
```

Each one asserts its own inputs. `src/profit.py` refits the economics itself, so
it is safe to run alone; running `src/economics.py` first is for reading the fit
diagnostics.
