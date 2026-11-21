"""Retraining: can the model be rebuilt, and would rebuilding it have helped? (slice 10)

TWO QUESTIONS, IN THIS ORDER

1. CAN THE SHIPPED MODEL BE REBUILT FROM THE DATA AT ALL?
   The fit lives in notebook 05: a scaler, an L1 logistic regression with
   C=0.05, a sigmoid calibrator fitted on the next eight months. A model that
   can only be made by re-running a notebook cannot be retrained on a
   schedule. fit_bundle() is that recipe as one function, and the first thing
   this file does is refit on the original rows and demand the shipped model
   back - the weights, the calibrator, and every test probability.

2. WOULD A RETRAINED MODEL HAVE BEEN BETTER?
   A walk-forward test. Stand on 2017-04-01, retrain with what was there,
   and score the nine months that follow (2017-04 to 2017-12) against the
   model that was actually shipped.

   "What was there" has two honest meanings, and the gap between them is the
   point of the exercise:

     as it happened   only loans whose outcome was already KNOWN on
                      2017-04-01. A loan from 2016 that is still being repaid
                      has no label yet. This is the model that could really
                      have been built that day.
     with hindsight   every loan issued before 2017-04-01, with the outcome
                      this snapshot knows. No one had this on the day. It is
                      the ceiling, not an option.

WHAT IS FIXED IN ADVANCE
- The recipe is the shipped one. C stays 0.05: slice 4 chose it and nothing
  here re-tunes it.
- The calibration rows are the most recent fifth of the training window by
  application date, cut at a month boundary - slice 5's rule.
- The comparison is a paired bootstrap of the AUC difference on the same
  rows, plus calibration, as in slice 6.
- One subgroup is reported: joint applications. It is named here because the
  drift monitor raised an alert on that feature, before this file was run.
- NOTHING IS SHIPPED. Both challengers are fitted on rows from the test
  period and scored on the rest of it, so neither can be promoted on this
  evidence: that would be choosing on test. They exist only to answer the
  question. models/ is not touched.

Run with:  python -m src.retrain
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .predict import GUARD_COLUMN, ORDER_LOGREG, predict_calibrated

C = 0.05                                  # slice 4's choice, not re-tuned
RETRAIN_DATE = pd.Timestamp('2017-04-01')
CHARGE_OFF_LAG_MONTHS = 5                 # a default is booked ~150 days after the last payment
JOINT = 'application_type_Joint App'
SEED = 42


def fit_bundle(train, calib, columns):
    """Slice 5's recipe: scaler -> L1 logistic -> sigmoid calibrator.

    train   rows the scaler and the weights are fitted on
    calib   later rows, used only to fit the two numbers of the calibrator
    """
    assert train['issue_d'].max() < calib['issue_d'].min(), \
        'the calibration rows must come after the training rows'

    scaler = StandardScaler().fit(train[columns])

    def scaled(df):
        return pd.DataFrame(scaler.transform(df[columns]), columns=columns, index=df.index)

    model = LogisticRegression(l1_ratio=1.0, solver='liblinear', C=C,
                               max_iter=2000, random_state=42)
    model.fit(scaled(train), train['target'])

    calibrator = LogisticRegression()
    calibrator.fit(model.decision_function(scaled(calib)).reshape(-1, 1), calib['target'])

    return {
        'scaler': scaler, 'model': model, 'calibrator': calibrator,
        'apply_order': ORDER_LOGREG, 'columns': list(columns),
        # the guard is a training fact: the largest ratio the weights ever saw
        'loan_to_income_guard_not_applied': float(train[GUARD_COLUMN].max()),
        'n_features_used': int((model.coef_[0] != 0).sum()),
    }


def split_recent_fifth(rows):
    """Slice 5's rule: calibrate on the most recent ~20% by date, cut at a month."""
    cut = rows['issue_d'].quantile(0.8).to_period('M').to_timestamp()
    return rows[rows['issue_d'] < cut], rows[rows['issue_d'] >= cut], cut


def paired_auc_difference(y, p_new, p_old, draws=2000, seed=SEED):
    """AUC(new) - AUC(old) on the same rows, with a 95% bootstrap interval."""
    y, p_new, p_old = np.asarray(y), np.asarray(p_new), np.asarray(p_old)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _line(name, y, p, extra=''):
    print(f'  {name:16s} AUC {roc_auc_score(y, p):.4f}   predicted {np.mean(p):6.1%}   '
          f'observed {np.mean(y):6.1%}   gap {np.mean(p) - np.mean(y):+6.1%}   '
          f'Brier {brier_score_loss(y, p):.4f}   log loss {log_loss(y, p):.4f}{extra}')


if __name__ == '__main__':
    import joblib

    from .economics import PROC, ROOT

    shipped = joblib.load(ROOT / 'models' / 'calibrated_logreg.joblib')
    columns = shipped['columns']
    split = pd.Timestamp(shipped['split_date'])

    train_file = pd.read_parquet(PROC / 'features_train.parquet')
    test_file = pd.read_parquet(PROC / 'features_test.parquet')
    rows = pd.concat([train_file, test_file], ignore_index=True)
    assert rows['issue_d'].is_monotonic_increasing, 'rows are not in issue_d order'

    # ---- 1. rebuild the shipped model ------------------------------------
    print('---- 1. can the shipped model be rebuilt from the data? ----')
    inner, valid = train_file[train_file['issue_d'] < split], train_file[train_file['issue_d'] >= split]
    assert (len(inner), len(valid)) == (16_134, 4_768)
    rebuilt = fit_bundle(inner, valid, columns)

    d_coef = np.abs(rebuilt['model'].coef_ - shipped['model'].coef_).max()
    d_cal = max(abs(rebuilt['calibrator'].coef_[0, 0] - shipped['calibrator'].coef_[0, 0]),
                abs(rebuilt['calibrator'].intercept_[0] - shipped['calibrator'].intercept_[0]))
    d_p = np.abs(predict_calibrated(rebuilt, test_file, apply_guard=True)
                 - predict_calibrated(shipped, test_file, apply_guard=True)).max()
    same_zeros = np.array_equal(rebuilt['model'].coef_ == 0, shipped['model'].coef_ == 0)

    print(f'  weights:        largest difference {d_coef:.2e}   '
          f'same {rebuilt["n_features_used"]} of {len(columns)} features kept: {same_zeros}')
    print(f'  calibrator:     largest difference {d_cal:.2e}   '
          f'(slope {rebuilt["calibrator"].coef_[0, 0]:.4f}, '
          f'shift {rebuilt["calibrator"].intercept_[0]:.4f})')
    print(f'  probabilities:  largest difference {d_p:.2e} over {len(test_file):,} test rows')
    print(f'  guard value:    {rebuilt["loan_to_income_guard_not_applied"]:.6f} rebuilt, '
          f'{shipped["loan_to_income_guard_not_applied"]:.6f} shipped')
    # an iterative solver is not bit-exact across machines, so 1e-6 and not 0.0
    assert same_zeros and d_coef < 1e-6 and d_cal < 1e-6 and d_p < 1e-6, \
        'fit_bundle() does not rebuild the shipped model'
    assert abs(rebuilt['loan_to_income_guard_not_applied']
               - shipped['loan_to_income_guard_not_applied']) < 1e-9
    print('  the recipe in this file is the shipped model')

    # ---- 2. walk forward --------------------------------------------------
    print(f'\n---- 2. standing on {RETRAIN_DATE.date()}: retrain, then score what follows ----')
    money = pd.read_parquet(PROC / 'money_table.parquet')
    assert np.array_equal(money['target'].to_numpy(), rows['target'].to_numpy()) and \
        np.array_equal(money['issue_d'].to_numpy(), rows['issue_d'].to_numpy()), \
        'money table and features are not in the same row order'
    known = money['last_pymnt_d'].fillna(money['issue_d']) + pd.to_timedelta(
        np.where(money['target'] == 1, CHARGE_OFF_LAG_MONTHS * 30.44, 0), unit='D')
    rows['known'] = known.to_numpy()

    past = rows[rows['issue_d'] < RETRAIN_DATE]
    future = rows[rows['issue_d'] >= RETRAIN_DATE]
    as_happened = past[past['known'] <= RETRAIN_DATE]

    print(f'  loans issued before the date:           {len(past):,}   bad rate {past["target"].mean():.1%}')
    print(f'  ... whose outcome was known on the day: {len(as_happened):,}   '
          f'bad rate {as_happened["target"].mean():.1%}   '
          f'({len(as_happened) / len(past):.0%} of them)')
    recent = past[past['issue_d'] >= '2016-01-01']
    recent_known = as_happened[as_happened['issue_d'] >= '2016-01-01']
    print(f'  of the {len(recent):,} issued since 2016-01, known on the day: {len(recent_known):,} '
          f'({len(recent_known) / len(recent):.0%}), bad rate {recent_known["target"].mean():.1%} '
          f'against {recent["target"].mean():.1%} in the end')
    print(f'  scored on: {len(future):,} loans issued {future["issue_d"].min().date()} to '
          f'{future["issue_d"].max().date()}')

    challengers = {}
    for name, data in (('as it happened', as_happened), ('with hindsight', past)):
        tr, ca, cut = split_recent_fifth(data)
        challengers[name] = fit_bundle(tr, ca, columns)
        b = challengers[name]
        print(f'  {name}: weights on {len(tr):,} loans before {cut.date()}, calibrated on '
              f'{len(ca):,} after; keeps {b["n_features_used"]} features; calibrator slope '
              f'{b["calibrator"].coef_[0, 0]:.3f} shift {b["calibrator"].intercept_[0]:+.3f}; '
              f'joint applications in its training rows: {int(tr[JOINT].sum())}')

    y = future['target'].to_numpy()
    p = {'shipped': predict_calibrated(shipped, future, apply_guard=True)}
    for name, b in challengers.items():
        p[name] = predict_calibrated(b, future, apply_guard=True)

    print(f'\n  on the {len(future):,} loans that followed:')
    for name in p:
        extra = ''
        if name != 'shipped':
            d, lo, hi = paired_auc_difference(y, p[name], p['shipped'])
            extra = f'\n  {"":16s} AUC against shipped {d:+.4f}  95% interval [{lo:+.4f}, {hi:+.4f}]'
        _line(name, y, p[name], extra)

    is_joint = (future[JOINT] == 1).to_numpy()
    print(f'\n  joint applications only ({is_joint.sum()} of them):')
    for name in p:
        print(f'  {name:16s} predicted {p[name][is_joint].mean():6.1%}   '
              f'observed {y[is_joint].mean():6.1%}   '
              f'declined at 0.43: {(p[name][is_joint] >= 0.43).mean():6.1%}   '
              f'weight on the joint flag {({"shipped": shipped, **challengers}[name])["model"].coef_[0][columns.index(JOINT)]:+.4f}')
    print(f'  everyone else:   declined at 0.43 by the shipped model: '
          f'{(p["shipped"][~is_joint] >= 0.43).mean():.1%}')

    out = pd.DataFrame({'issue_d': future['issue_d'].to_numpy(), 'target': y,
                        'joint': is_joint, 'p_shipped': p['shipped'],
                        'p_as_it_happened': p['as it happened'],
                        'p_with_hindsight': p['with hindsight']})
    out.to_parquet(PROC / 'slice10_walk_forward.parquet', index=False)
    print('\nwrote data/processed/slice10_walk_forward.parquet')
    print('nothing was written to models/: neither challenger is a candidate to ship')
