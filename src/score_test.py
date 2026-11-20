"""Score the test set with the shipped chain and write the probabilities (slice 9).

WHY THIS FILE EXISTS
Slice 6 wrote data/processed/predictions_test_slice6.parquet from notebook 06,
where the logistic chain had been retyped by hand. src/predict.py exists to stop
exactly that, so the guarded rescore does not go back through the notebook - it
imports the chain instead. This file is the only place the test set is rescored.

WHAT CHANGES AND WHAT DOES NOT
The loan_to_income guard is ON here. Slice 5 found one test row at
loan_to_income 19.5 against a training max of 1.782, scored 1.000000 on a loan
that repaid. Slice 7 measured what clipping does: the maximum probability goes
1.000000 -> 0.982307 and 2 of 5,378 rows move. Both numbers are asserted below,
so this file fails if the guard is not the small change slice 7 said it was.

Nothing is learned here. The guard value comes out of the bundle, the
probabilities come out of the chain in src/predict.py, and slice 6's file is
left untouched so its published numbers stay reproducible from disk.

THE OUTPUT
data/processed/predictions_test_slice9.parquet
  p_logreg             guard ON. What slice 9 ships and what src/profit.py reads
  p_logreg_unguarded   guard OFF. Not for scoring - it is here so the before and
                       after comparison is a column that can be queried later,
                       not a number that was printed once
  p_lgbm_tuned         unchanged. A tree cannot extrapolate past its training
                       range, so the guard does not apply to it
"""

import joblib
import numpy as np
import pandas as pd

from .economics import PROC, ROOT
from .predict import predict_calibrated, predict_calibrated_lgbm
from .tolerance import FLOAT_TOL

PUBLISHED_PATH = PROC / 'predictions_test_slice6.parquet'
OUT_PATH = PROC / 'predictions_test_slice9.parquet'

# what slice 7 measured when the guard is turned on. These are the assertions
# that make this a check and not just a rerun.
EXPECTED_MOVED = 2
EXPECTED_MAX_GUARDED = 0.982307

# carried through so the output file can check its own row order later
IDENTITY = ['issue_d', 'zip_code', 'addr_state', 'target']


if __name__ == '__main__':
    X = pd.read_parquet(PROC / 'features_test.parquet')
    published = pd.read_parquet(PUBLISHED_PATH)

    # row order FIRST. A reordering shows up as a probability difference and
    # sends you looking for the wrong bug - slice 4's lesson.
    assert np.array_equal(X['target'].values, published['target'].values), \
        'features_test and slice 6 predictions are not in the same row order'

    b_log = joblib.load(ROOT / 'models' / 'calibrated_logreg.joblib')
    b_lgb = joblib.load(ROOT / 'models' / 'calibrated_lgbm.joblib')

    guard_value = b_log['loan_to_income_guard_not_applied']
    print(f'guard value from the bundle: loan_to_income clipped at {guard_value:.6f}')
    print(f'rows: {len(X):,}\n')

    p_unguarded = predict_calibrated(b_log, X)                      # slice 5 / 6
    p_guarded = predict_calibrated(b_log, X, apply_guard=True)       # slice 9
    p_lgbm = predict_calibrated_lgbm(b_lgb, X)

    # ---- 1. nothing except the guard may have changed --------------------
    # exact on the machine that wrote slice 6's file; see src/tolerance.py for
    # the one case where it is not
    diff_log = np.abs(p_unguarded - published['p_logreg'].values).max()
    diff_lgb = np.abs(p_lgbm - published['p_lgbm_tuned'].values).max()

    print(f'unguarded logistic vs slice 6   max difference {diff_log:.2e}')
    print(f'lightgbm vs slice 6             max difference {diff_lgb:.2e}')
    assert diff_log <= FLOAT_TOL, 'the unguarded logistic chain no longer reproduces slice 6'
    assert diff_lgb <= FLOAT_TOL, 'the lightgbm chain no longer reproduces slice 6'

    # ---- 2. the guard is the small change slice 7 measured ---------------
    moved = np.flatnonzero(p_guarded != p_unguarded)

    print(f'\nguard off -> max probability {p_unguarded.max():.6f}')
    print(f'guard on  -> max probability {p_guarded.max():.6f}   '
          f'({len(moved)} of {len(X):,} rows move)')

    assert len(moved) == EXPECTED_MOVED, \
        f'{len(moved)} rows moved, slice 7 measured {EXPECTED_MOVED}'
    assert round(float(p_guarded.max()), 6) == EXPECTED_MAX_GUARDED, \
        f'max guarded probability is {p_guarded.max():.6f}, slice 7 measured ' \
        f'{EXPECTED_MAX_GUARDED}'

    # the rows themselves, because slice 9 has to record what moved
    print('\n---- the rows the guard moves ----')
    for i in moved:
        print(f'  row {i:5d}   loan_to_income {X["loan_to_income"].values[i]:7.3f} '
              f'-> {guard_value:.3f}    '
              f'p {p_unguarded[i]:.6f} -> {p_guarded[i]:.6f}    '
              f'actual outcome {"default" if X["target"].values[i] == 1 else "repaid"}')

    # ---- 3. write it -----------------------------------------------------
    out = X[IDENTITY].copy()
    out['p_logreg'] = p_guarded
    out['p_logreg_unguarded'] = p_unguarded
    out['p_lgbm_tuned'] = p_lgbm
    out.to_parquet(OUT_PATH, index=False)

    print(f'\nwrote {OUT_PATH.relative_to(ROOT)}   {out.shape[0]:,} rows x '
          f'{out.shape[1]} columns')
    print(f'columns: {list(out.columns)}')
