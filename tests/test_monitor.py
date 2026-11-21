"""The monitor and the request log (slice 10).

The first half needs only models/. The last test replays history and needs
data/processed, so it skips where the data is missing (CI, the Docker image).
"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import requestlog
from src.monitor import (Reference, frame_from_records, level, load_policy, make_bins, psi,
                         score_of, shares)


# ---- PSI -----------------------------------------------------------------

def test_psi_is_zero_for_the_same_values():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == 0.0


def test_psi_is_small_for_a_fresh_sample_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.01
    assert psi(ref, rng.normal(loc=1.0, size=20_000)) > 0.25      # one sd: "shifted"


def test_psi_matches_a_value_worked_by_hand():
    # two bins, 50/50 in the reference, 80/20 now:
    # (0.8-0.5)*ln(0.8/0.5) + (0.2-0.5)*ln(0.2/0.5) = 0.4159
    ref = np.array([0] * 50 + [1] * 50)
    cur = np.array([0] * 80 + [1] * 20)
    assert psi(ref, cur) == pytest.approx(0.415888, abs=1e-6)


def test_a_value_the_reference_never_had_gets_its_own_bin():
    ref = np.array([0] * 990 + [1] * 10)
    bins = make_bins(ref)
    assert bins['kind'] == 'discrete'
    s = shares(np.array([0, 1, 2, 2]), bins)
    assert list(s) == [0.25, 0.25, 0.5]                             # the last bin is "unseen"
    assert np.isfinite(psi(ref, np.array([2] * 100)))               # no division by zero


def test_bins_are_cut_on_the_reference_only():
    ref = np.arange(1000.0)
    bins = make_bins(ref)
    assert len(bins['edges']) == 9
    assert shares(ref, bins) == pytest.approx([0.1] * 10, abs=0.002)
    assert shares(ref + 10_000, bins)[-1] == 1.0                    # all beyond the last edge


def test_a_small_window_has_a_noise_floor_and_a_large_one_a_lower_one():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({'a': rng.normal(size=8000), 'b': rng.integers(0, 3, 8000)})
    reference = Reference(frame, ['a', 'b'])
    small, large = reference.noise_p99(200), reference.noise_p99(4000)
    assert (small > large).all() and (large > 0).all()
    # inside the noise nothing is ever raised, however the thresholds are set
    assert level(small['a'] * 0.99, warn=0.0, alert=0.0, noise=small['a']) == 'ok'
    assert level(0.3, warn=0.10, alert=0.25) == 'alert'
    assert level(0.15, warn=0.10, alert=0.25) == 'warn'


def test_the_policy_was_declared_with_round_numbers():
    policy = load_policy()
    assert (policy['psi']['warn'], policy['psi']['alert']) == (0.10, 0.25)
    assert policy['min_rows'] == 300


# ---- the request log -----------------------------------------------------

@pytest.fixture
def logged_client(tmp_path, monkeypatch):
    """The service with the request log switched on, writing to a temp file."""
    import service.app as app_module
    log = requestlog.RequestLog(tmp_path / 'requests.jsonl')
    monkeypatch.setattr(app_module, 'request_log', log)
    return TestClient(app_module.app), log.path


def test_the_log_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv(requestlog.ENV_VAR, raising=False)
    assert requestlog.from_environment() is None
    assert client.get('/health').json()['request_log'] is False


def test_every_request_is_logged_and_the_caller_sees_no_difference(logged_client, engine, typical):
    logged, path = logged_client
    bad = {**typical, 'annual_inc': -1}

    with_log = logged.post('/decision', json=typical)
    assert with_log.json() == engine.decide(typical)        # the answer with the log off
    assert 'reasons' not in with_log.json()                 # computed for the log, not returned
    assert logged.post('/reasons', json=typical).status_code == 200
    assert logged.post('/decision', json=bad).status_code == 422

    records = requestlog.read_log(path)
    assert len(records) == 3
    assert records[0]['application'] == typical
    assert records[0]['probability_of_default'] == with_log.json()['probability_of_default']
    assert records[0]['raised_risk'] and records[0]['model_version']
    assert records[2]['rejected'][0]['field'] == 'annual_inc'
    assert all(json.dumps(r) for r in records)


def test_the_monitor_reads_what_the_service_wrote(logged_client, engine, typical):
    logged, path = logged_client
    stretched = {**typical, 'term': 60, 'loan_amnt': 35_000, 'annual_inc': 30_000, 'dti': 38}
    for application in (typical, stretched, {**typical, 'loan_amnt': 40_000},
                        {**typical, 'emp_length': None}, {**typical, 'term': 48}):
        logged.post('/decision', json=application)

    frame = frame_from_records(engine, requestlog.read_log(path))
    assert list(frame['rejected']) == [False, False, False, False, True]
    assert list(frame['outside'][:4]) == [False, False, True, False]
    assert frame['outside_fields'][2] == ['loan_amnt']
    assert list(frame['imputed'][:4]) == [False, False, False, True]
    assert pd.isna(frame['main_reason'][0]) and isinstance(frame['main_reason'][1], str)

    # the features rebuilt from the log are the ones the service scored
    rebuilt = engine.probabilities(frame.loc[[0, 1, 2, 3], engine.bundle['columns']])
    assert np.abs(rebuilt - frame['p'][:4].to_numpy(dtype='float64')).max() < 1e-12


def test_a_failed_write_does_not_fail_the_request(tmp_path, typical, engine, capsys):
    log = requestlog.RequestLog(tmp_path / 'log.jsonl')
    log.path = tmp_path                                     # a directory: open() will fail
    log.write(typical, engine.decide(typical))
    log.write(typical, engine.decide(typical))
    assert capsys.readouterr().err.count('could not write') == 1     # said once, not per request


# ---- the replay ----------------------------------------------------------

def test_the_replay_keeps_the_reference_to_the_training_rows(engine, raw_rows):
    """The reference must be the 16,134 rows the weights saw, and a quarter
    drawn from inside it must not look like drift."""
    from src.features import transform
    cols = engine.bundle['columns']
    train = raw_rows['train']
    inner = train[train['issue_d'] < pd.Timestamp(engine.bundle['split_date'])]
    assert len(inner) == 16_134

    ref = transform(inner, engine.spec)[cols].reset_index(drop=True)
    ref['score'] = score_of(engine.probabilities(ref))
    reference = Reference(ref, cols + ['score'])

    window = ref.sample(1000, random_state=3)
    assert reference.psi(window)['score'] < reference.noise_p99(1000)['score'] * 1.5

    late = transform(raw_rows['test'], engine.spec)[cols]
    drift = reference.psi(late.assign(score=score_of(engine.probabilities(late))))
    assert drift.drop('score').idxmax() == 'application_type_Joint App'
