"""Why an application got the probability it got (slice 9).

Slice 7 worked out the explanation for the shipped model and proved it: for a
linear model on scaled features, with the training rows as the reference,

    contribution of a feature = its coefficient x its scaled value

and the contributions plus the intercept add up to the model's score exactly.
It is arithmetic, not an estimate, and slice 7 checked it against the shap
library. This file is that formula as a function the service can call. It
takes the scaled values from src/predict.py - it does not scale anything
itself - so an explanation can never be about different numbers than the ones
that were scored.

TWO THINGS SLICE 9 ADDS

1. The contributions are carried through the calibrator. Slice 7 explained the
   raw score. The service returns the calibrated probability, so the reasons
   returned with it are scaled by the calibrator's slope and the base value
   takes its shift. They then add up to the log-odds of the probability in
   the same response. The self-check and the tests assert that.

2. Features are grouped into factors a person would recognise. The seven
   purpose dummies are one fact about the applicant ("the loan is for a small
   business"), not eight. Adding contributions inside a group is exact,
   because the whole explanation is a sum.

WHAT A CONTRIBUTION MEANS
"Compared with a typical borrower in the training data, this factor moved the
log-odds of default by this much." Positive raises the risk. It describes the
model, not the world: it says what the model weighed, not what would happen if
the applicant changed that factor.
"""

import numpy as np
import pandas as pd

from .columns import CAT_FOLD_RARE
from .contract import FIELDS
from .features import CREDIT_HISTORY_CAP
from .predict import GUARD_COLUMN, scale_features

# features that are one fact told more than once -> one factor
FACTOR_OF = {
    'log_annual_inc': 'annual_inc',
    'log_revol_bal': 'revol_bal',
    'log_tot_cur_bal': 'tot_cur_bal',
    'mths_since_recent_inq_missing': 'mths_since_recent_inq',
}
CATEGORY_FACTORS = ['purpose', 'home_ownership', 'verification_status', 'application_type']

# factors the build creates, which the contract therefore has no label for
DERIVED_LABELS = {
    'loan_to_income': 'Loan amount compared with annual income',
    'credit_history_months': 'Length of credit history, in months',
    'open_to_total_acc': 'Share of all accounts that are still open',
    'fico': 'Credit score',
}


def factor_of(feature):
    """Which factor a model feature belongs to."""
    if feature in FACTOR_OF:
        return FACTOR_OF[feature]
    for cat in CATEGORY_FACTORS:
        if feature.startswith(cat + '_'):
            return cat
    return feature


def label_of(factor):
    if factor in DERIVED_LABELS:
        return DERIVED_LABELS[factor]
    return FIELDS[factor]['label']


def contributions(bundle, features, apply_guard=False, calibrated=True):
    """One row per application, one column per model feature, plus the base.

    calibrated=False gives slice 7's numbers (log-odds of the raw score).
    calibrated=True gives log-odds of the probability the service returns.
    """
    X_scaled = scale_features(bundle, features, apply_guard)
    phi = X_scaled * bundle['model'].coef_[0]          # (rows, 71) * (71,)
    base = float(bundle['model'].intercept_[0])

    if calibrated:
        slope = float(bundle['calibrator'].coef_[0, 0])
        shift = float(bundle['calibrator'].intercept_[0])
        phi, base = phi * slope, base * slope + shift

    return phi, base


def by_factor(phi):
    """Add the feature contributions up into factors. Exact: it is a sum."""
    groups = pd.Series({col: factor_of(col) for col in phi.columns})
    return phi.T.groupby(groups, sort=False).sum().T


def typical_values(bundle):
    """The reference point of every explanation: the mean training borrower.
    Read from the scaler, which was fitted on exactly those rows."""
    return dict(zip(bundle['columns'], bundle['scaler'].mean_))


def explain(bundle, features, clean_rows, apply_guard=False, top=4):
    """The reasons for each application, in the order of features' rows.

    clean_rows: the validated applications (src/contract.py), for the
    categories, which the features only hold as dummies.
    """
    phi, base = contributions(bundle, features, apply_guard)
    factors = by_factor(phi)
    typical = typical_values(bundle)
    guard = bundle['loan_to_income_guard_not_applied']

    out = []
    for i, (_, row) in enumerate(factors.iterrows()):
        feats, clean = features.iloc[i], clean_rows[i]
        items = []
        for factor, value in row.items():
            if value == 0.0:                 # the model gives this factor no weight
                continue
            item = {'factor': factor, 'label': label_of(factor),
                    'contribution': float(value)}
            if factor in CATEGORY_FACTORS:
                item['value'] = clean[factor]
                rare, into = CAT_FOLD_RARE.get(factor, ([], None))
                if clean[factor] in rare:
                    item['note'] = f"scored as '{into}', as in training"
            else:
                item['value'] = float(feats[factor])
                item['typical'] = float(typical[factor])
                if factor == GUARD_COLUMN and apply_guard and feats[factor] > guard:
                    item['note'] = (f'scored as {guard:.3f}, the highest value in the '
                                    f'training data')
                if factor == 'credit_history_months' and feats[factor] >= CREDIT_HISTORY_CAP:
                    item['note'] = f'capped at {CREDIT_HISTORY_CAP} months, as in training'
            items.append(item)

        raising = sorted((x for x in items if x['contribution'] > 0),
                         key=lambda x: -x['contribution'])
        lowering = sorted((x for x in items if x['contribution'] < 0),
                          key=lambda x: x['contribution'])
        out.append({
            'base_log_odds': base,
            'total_log_odds': base + float(row.sum()),
            'raised_risk': raising[:top],
            'lowered_risk': lowering[:top],
            'n_factors': len(items),
        })
    return out


# ---- self-check: is this still slice 7's explanation? -------------------

if __name__ == '__main__':
    import joblib
    from .economics import PROC, ROOT
    from .predict import predict_calibrated
    from .tolerance import max_abs_diff

    bundle = joblib.load(ROOT / 'models' / 'calibrated_logreg.joblib')
    X = pd.read_parquet(PROC / 'features_test.parquet')
    slice7 = pd.read_parquet(PROC / 'shap_logreg_test.parquet')
    cols = bundle['columns']

    assert np.array_equal(X['target'].to_numpy(), slice7['target'].to_numpy()), \
        'features_test and slice 7 contributions are not in the same row order'

    # every model feature must land in a factor that has a label
    for col in cols:
        label_of(factor_of(col))

    # 1. guard off, raw score: slice 7's saved contributions, value for value.
    #    1e-9 and not 0.0: slice 7 saved these from a notebook cell, and its own
    #    assertion against the shap library used 1e-8.
    phi_raw, base_raw = contributions(bundle, X, calibrated=False)
    diff = max_abs_diff(phi_raw[cols], slice7[cols])
    print(f'contributions vs slice 7        max difference {diff:.2e}  '
          f'over {phi_raw.shape[0]:,} rows x {phi_raw.shape[1]} features')
    assert diff < 1e-9, 'this is no longer the explanation slice 7 published'

    # 2. guard on, calibrated: they add up to the probability the service gives
    phi, base = contributions(bundle, X, apply_guard=True)
    p = predict_calibrated(bundle, X, apply_guard=True)
    total = base + phi.sum(axis=1).to_numpy()
    back = 1 / (1 + np.exp(-total))
    print(f'base + contributions -> probability   max difference '
          f'{np.abs(back - p).max():.2e}')
    assert np.abs(back - p).max() < 1e-12, 'the reasons do not add up to the probability'

    # 3. grouping into factors loses nothing
    factors = by_factor(phi)
    assert np.abs(factors.sum(axis=1).to_numpy() - phi.sum(axis=1).to_numpy()).max() < 1e-12
    print(f'{phi.shape[1]} features -> {factors.shape[1]} factors, same total')

    print('\nthe reasons are slice 7\'s explanation and add up to the served probability')
