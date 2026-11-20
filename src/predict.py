"""Scoring chains for the credit risk engine (slice 7, guard switched on by slice 9).

Each saved bundle holds fitted objects that only work in one order.
These functions ARE that order, written in code rather than in prose,
so a notebook and the slice 9 service cannot drift apart - which had
already started happening: the logistic chain was retyped by hand in
notebook 06 because it only ever lived in a notebook 05 cell.

Nothing here learns anything. Every number comes out of the bundle.
"""

import numpy as np
import pandas as pd

GUARD_COLUMN = 'loan_to_income'

# the chains these functions implement, checked against each bundle
ORDER_LOGREG = ['scaler.transform',
                'model.decision_function',
                'calibrator.predict_proba[:, 1]']

ORDER_LGBM   = ['model.predict_proba(X, raw_score=True)',
                'calibrator.predict_proba(raw.reshape(-1,1))[:, 1]']


# ---- shared checks ------------------------------------------------------

def _check_order(bundle, expected):
    """A bundle saved with a different chain must fail loudly, not score wrong."""
    found = bundle.get('apply_order')
    if found != expected:
        raise ValueError(f"bundle apply_order is {found}, "
                         f"but this function implements {expected}")


def _select_columns(bundle, X_raw):
    """The bundle's columns, in the bundle's order, as a safe copy.

    Selecting by name already raises a KeyError on its own; this names
    the missing columns instead, which is what you want once a service
    is calling it with somebody else's dataframe.
    """
    missing = set(bundle['columns']) - set(X_raw.columns)
    if missing:
        raise ValueError(f"X_raw is missing {len(missing)} of the bundle's "
                         f"columns: {sorted(missing)}")

    # the bundle's order, because the scaler and the coefficients work by
    # POSITION. .copy() so the guard below never writes into the caller's frame
    return X_raw[bundle['columns']].copy()


# ---- the shipped model: scaler -> model -> calibrator -------------------

def scale_features(bundle, X_raw, apply_guard=False):
    """The first link of the logistic chain: select, guard, scale.

    Split out in slice 9 because the reasons endpoint needs the scaled values
    (a contribution is coefficient x scaled value) and must not rebuild them
    its own way. predict_calibrated calls this, so there is still one chain.
    """
    _check_order(bundle, ORDER_LOGREG)
    X = _select_columns(bundle, X_raw)

    if apply_guard:
        # slice 5 found one test row at loan_to_income 19.5 against a training
        # max of 1.782 - probability 1.000000 on a loan that repaid, 1.2% of
        # the whole test log loss. Clip BEFORE scaling: the scaler is what
        # turns 19.5 into a +34.5 contribution. Slice 9 turns this on.
        X[GUARD_COLUMN] = X[GUARD_COLUMN].clip(
            upper=bundle['loan_to_income_guard_not_applied'])

    return pd.DataFrame(                           # keep the names, or sklearn warns
        bundle['scaler'].transform(X),
        columns=bundle['columns'],
        index=X.index,
    )


def predict_calibrated(bundle, X_raw, apply_guard=False):
    """Raw features in, honest probability out. Slice 5's calibrated logistic.

    X_raw may carry extra columns; only the bundle's are used.
    apply_guard is off by default, so this reproduces slice 5 exactly.
    The service and src/score_test.py pass apply_guard=True (slice 9).
    """
    X_scaled = scale_features(bundle, X_raw, apply_guard)
    score = bundle['model'].decision_function(X_scaled)            # log-odds
    return bundle['calibrator'].predict_proba(score.reshape(-1, 1))[:, 1]


# ---- the slice 6 model: model -> calibrator -----------------------------

def predict_calibrated_lgbm(bundle, X_raw):
    """Raw features in, honest probability out. Slice 6's tuned LightGBM.

    No scaler: trees split on thresholds, so scaling is a no-op.
    No apply_guard either, and that is a decision, not an oversight - a tree
    cannot extrapolate past its training range, so clipping the input changes
    nothing. The same borrower that scores 1.000 here scores 0.25 there.
    """
    _check_order(bundle, ORDER_LGBM)
    X = _select_columns(bundle, X_raw)

    raw = np.asarray(bundle['model'].predict_proba(X, raw_score=True))
    if raw.ndim == 2:              # the shape differs between lightgbm versions
        raw = raw[:, 1]
    return bundle['calibrator'].predict_proba(raw.reshape(-1, 1))[:, 1]


# ---- self-check: does this file still reproduce the published numbers? --

if __name__ == '__main__':
    import joblib
    from .tolerance import FLOAT_TOL
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]     # runs from any directory
    PROC = ROOT / 'data' / 'processed'

    # the full frame, not just the 71 - the functions select what they need,
    # which is also what a real caller would hand them
    X         = pd.read_parquet(PROC / 'features_test.parquet')
    published = pd.read_parquet(PROC / 'predictions_test_slice6.parquet')

    # row order FIRST. a reordering shows up as a probability difference and
    # sends you looking for the wrong bug
    assert np.array_equal(X['target'].values, published['target'].values), \
        'features_test and the published predictions are not in the same row order'

    checks = [
        ('logistic (shipped)', 'calibrated_logreg.joblib',
         predict_calibrated,      'p_logreg'),
        ('lightgbm (slice 6)', 'calibrated_lgbm.joblib',
         predict_calibrated_lgbm, 'p_lgbm_tuned'),
    ]

    for name, filename, score_fn, column in checks:
        bundle = joblib.load(ROOT / 'models' / filename)
        p      = score_fn(bundle, X)
        diff   = np.abs(p - published[column].values).max()

        print(f'{name:20s} max difference {diff:.2e}  over {len(p):,} rows')
        # compare the ARRAYS, never a summary number - slice 4's lesson
        assert diff <= FLOAT_TOL, f'{name} no longer reproduces {column}'

    # and prove the guard is a real switch, not decoration
    b_log   = joblib.load(ROOT / 'models' / 'calibrated_logreg.joblib')
    guarded = predict_calibrated(b_log, X, apply_guard=True)
    plain   = predict_calibrated(b_log, X)
    n_moved = int((guarded != plain).sum())

    print(f'\nguard off -> max probability {plain.max():.6f}')
    print(f'guard on  -> max probability {guarded.max():.6f}   '
          f'({n_moved} of {len(plain):,} rows move)')
    print('\nboth chains reproduce every published probability exactly')