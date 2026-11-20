"""The train/serve skew test (slice 9). The most important test in the slice.

THE QUESTION
A model is trained on features that a notebook built. In production the
features are built again, by other code, one application at a time. If the
two builds differ in anything - a median, the order of two steps, how a blank
is treated - the model is scored on numbers it was never trained on, and
nothing crashes. It just gets quietly worse. That is train/serve skew.

THE TEST
Take the raw rows, before any feature engineering. Turn each one into the
JSON a caller would send. Push it through the service. Then demand that what
comes out is what the training pipeline saved:

    the 71 features          ==  features_test.parquet / features_train.parquet
    the probability          ==  predictions_test_slice9.parquet
    the decision             ==  the published probability against the threshold

and over HTTP, not only by calling the function - JSON is one more place a
number can change.

"==" means a difference of exactly 0.0 on the machine that wrote the files.
src/tolerance.py explains the one setting for any other machine.

The test set is used here for reporting only: nothing is fitted, chosen or
tuned from what these tests see.
"""

import numpy as np
import pandas as pd
import pytest

from src.contract import validate
from src.decide import to_application
from src.economics import PROC
from src.predict import predict_calibrated
from src.tolerance import FLOAT_TOL, SINGLE_ROW_TOL, max_abs_diff


def _features(engine, applications):
    clean = [validate(a, engine.spec)[0] for a in applications]
    return engine.features(clean)


def test_rows_are_in_time_order(raw_rows):
    # every later comparison is by position, so this comes first
    assert raw_rows['train']['issue_d'].is_monotonic_increasing
    assert raw_rows['test']['issue_d'].is_monotonic_increasing
    assert raw_rows['train']['issue_d'].max() < raw_rows['test']['issue_d'].min()


def test_no_real_application_is_rejected(engine, test_applications):
    """The contract was written from the training data. If it turned away
    real test applications, it would be a contract for a different dataset."""
    results = engine.decide_many(test_applications)
    rejected = [r for r in results if 'rejected' in r]
    assert len(results) == 5_378
    assert not rejected, f'{len(rejected)} test applications rejected, e.g. {rejected[0]}'


@pytest.mark.parametrize('split', ['test', 'train'])
def test_service_rebuilds_the_saved_features(engine, raw_rows, split):
    raw = raw_rows[split]
    saved = pd.read_parquet(PROC / f'features_{split}.parquet')
    assert np.array_equal(raw['target'].to_numpy(), saved['target'].to_numpy()), \
        'raw rows and saved features are not in the same row order'

    built = _features(engine, [to_application(row) for _, row in raw.iterrows()])

    cols = engine.bundle['columns']
    assert len(cols) == 71
    worst = {c: max_abs_diff(built[c], saved[c]) for c in cols}
    col = max(worst, key=worst.get)
    assert worst[col] <= FLOAT_TOL, \
        f'{split}: feature {col} differs from the saved file by {worst[col]:.2e}'


def test_service_reproduces_the_published_probabilities(engine, test_applications, published):
    results = engine.decide_many(test_applications)
    p = np.array([r['probability_of_default'] for r in results])

    diff = max_abs_diff(p, published['p_logreg'])
    assert diff <= FLOAT_TOL, f'served probabilities differ from the published ones by {diff:.2e}'


def test_service_agrees_with_the_chain_on_the_saved_features(engine, test_applications):
    """Two roads to the same number: raw row -> service, and saved features ->
    src/predict.py. Both run here, on this machine, so this one is always exact."""
    saved = pd.read_parquet(PROC / 'features_test.parquet')
    direct = predict_calibrated(engine.bundle, saved, apply_guard=True)
    served = np.array([r['probability_of_default']
                       for r in engine.decide_many(test_applications)])
    assert max_abs_diff(served, direct) == 0.0


def test_decisions_match_the_published_ones(engine, test_applications, published):
    results = engine.decide_many(test_applications)
    served = np.array([r['decision'] == 'approve' for r in results])
    expected = published['p_logreg'].to_numpy() < engine.threshold

    assert (served == expected).all(), f'{int((served != expected).sum())} decisions differ'
    # slice 8 published 90.0% approval at 0.43 on these 5,378 loans
    assert served.sum() == 4_842
    assert round(served.mean(), 3) == 0.900


def test_the_guard_touches_exactly_the_rows_slice_7_measured(engine, test_applications, published):
    results = engine.decide_many(test_applications)
    guarded = [i for i, r in enumerate(results) if r['flags']['guard_applied']]
    moved = np.flatnonzero(published['p_logreg'].to_numpy()
                           != published['p_logreg_unguarded'].to_numpy())
    assert guarded == list(moved)
    assert len(guarded) == 2


def test_over_http_nothing_changes(client, engine, test_applications, published):
    """Every 20th test row through the real endpoint, one request each.

    Against the same application decided directly: exactly equal. JSON does
    not move a number. Against the published batch: equal to SINGLE_ROW_TOL,
    because one row and a batch do not add up in the same order (src/tolerance.py).
    """
    for i in range(0, len(test_applications), 20):
        response = client.post('/decision', json=test_applications[i])
        assert response.status_code == 200, response.text
        body = response.json()
        direct = engine.decide(test_applications[i])

        assert body['probability_of_default'] == direct['probability_of_default']
        assert body == direct
        assert abs(body['probability_of_default']
                   - published['p_logreg'].iloc[i]) <= SINGLE_ROW_TOL
        assert (body['decision'] == 'approve') == (published['p_logreg'].iloc[i] < engine.threshold)


def test_reasons_are_slice_7s_and_add_up(client, engine, test_applications, published):
    slice7 = PROC / 'shap_logreg_test.parquet'
    if not slice7.exists():
        pytest.skip('data/processed/shap_logreg_test.parquet is missing (slice 7)')
    slice7 = pd.read_parquet(slice7)
    slope = float(engine.bundle['calibrator'].coef_[0, 0])
    unmoved = (published['p_logreg'] == published['p_logreg_unguarded']).to_numpy()

    for i in range(0, len(test_applications), 50):
        body = client.post('/reasons', json=test_applications[i]).json()
        reasons = body['reasons']

        # they add up to the probability in the same response
        p_back = 1 / (1 + np.exp(-reasons['total_log_odds']))
        assert abs(p_back - body['probability_of_default']) < 1e-12

        # and each factor is slice 7's contribution, carried through the calibrator
        if unmoved[i]:
            for item in reasons['raised_risk'] + reasons['lowered_risk']:
                f = item['factor']
                cols = [c for c in engine.bundle['columns']
                        if c == f or c.startswith(f + '_') or c == 'log_' + f]
                expected = slope * float(slice7.loc[i, cols].sum())
                assert abs(item['contribution'] - expected) < 1e-9, (i, f)
