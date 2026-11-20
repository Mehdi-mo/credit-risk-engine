"""The service over HTTP (slice 9). Needs only models/, no data."""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_health_and_versions(client, engine):
    body = client.get('/health').json()
    assert body['status'] == 'ok'
    for key in ('model_version', 'feature_spec_version', 'economics_version', 'policy_version'):
        assert body[key] == engine.versions[key]
        assert '@' in body[key] and len(body[key].split('@')[1]) == 12


def test_decision_returns_what_slice_9_promised(client, engine, typical):
    body = client.post('/decision', json=typical).json()
    # the decision, the probability, the threshold it was compared against,
    # the model version and the economics version
    assert body['decision'] == 'approve'
    assert 0 < body['probability_of_default'] < 1
    assert body['threshold'] == 0.43
    assert body['model_version'] == engine.versions['model_version']
    assert body['economics_version'] == engine.versions['economics_version']
    # and the threshold never travels without its assumption
    assert 'margin' in body['threshold_assumption']
    assert (body['decision'] == 'approve') == (body['probability_of_default'] < body['threshold'])


def test_a_declined_application_is_told_why(client, typical):
    application = {**typical, 'term': 60, 'loan_amnt': 35_000, 'annual_inc': 30_000,
                   'dti': 38, 'fico_range_low': 660, 'fico_range_high': 664,
                   'home_ownership': 'RENT', 'purpose': 'small_business',
                   'acc_open_past_24mths': 12, 'inq_last_6mths': 4}
    body = client.post('/reasons', json=application).json()
    assert body['decision'] == 'decline'

    raised = body['reasons']['raised_risk']
    assert 1 <= len(raised) <= 4
    assert all(r['contribution'] > 0 for r in raised)
    assert [r['contribution'] for r in raised] == sorted((r['contribution'] for r in raised),
                                                        reverse=True)
    assert all(r['label'] and 'value' in r for r in raised)
    # the things this application was made bad at are the reasons it gets
    assert {'loan_to_income', 'term'} <= {r['factor'] for r in raised}

    total = body['reasons']['total_log_odds']
    assert abs(1 / (1 + math.exp(-total)) - body['probability_of_default']) < 1e-12


def test_decision_and_reasons_agree(client, typical):
    a = client.post('/decision', json=typical).json()
    b = client.post('/reasons', json=typical).json()
    assert a['probability_of_default'] == b['probability_of_default']
    assert a['decision'] == b['decision']
    assert 'reasons' not in a and 'reasons' in b


def test_a_broken_application_gets_422_and_every_reason(client, typical):
    response = client.post('/decision', json={**typical, 'annual_inc': -1, 'colour': 'blue'})
    assert response.status_code == 422
    fields = {p['field'] for p in response.json()['rejected']}
    assert fields == {'annual_inc', 'colour'}

    assert client.post('/decision', json=[1, 2, 3]).status_code == 422


def test_contract_endpoint_lists_every_field(client):
    body = client.get('/contract').json()
    from src.contract import FIELDS, OPTIONAL_FIELDS
    assert [f['field'] for f in body['fields']] == list(FIELDS) + list(OPTIONAL_FIELDS)
    assert len(body['blank_is_imputed_for']) == 10


def test_example_is_accepted_as_it_is(client):
    example = client.get('/example').json()
    assert client.post('/decision', json=example).status_code == 200


def test_version_states_the_assumption(client):
    body = client.get('/version').json()
    assert body['policy']['threshold'] == 0.43
    assert body['policy']['sensitivity']['margin_0.6x_to_1.4x'] == [0.24, 0.55]
    assert len(body['limits']) == 3


def test_dashboard_is_served(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_the_service_holds_no_copy_of_the_pipeline():
    """The rule of the slice, as a test. If any of these words shows up in
    service/, a step of the pipeline has been retyped there."""
    forbidden = ['decision_function', 'predict_proba', '.transform(', 'fillna',
                 'get_dummies', 'log1p', 'clip(', 'coef_', 'margin_pct', 'p_star']
    for path in (ROOT / 'service').glob('*.py'):
        source = path.read_text(encoding='utf-8')
        found = [word for word in forbidden if word in source]
        assert not found, f'{path.name} contains {found}'
    page = (ROOT / 'service' / 'dashboard.html').read_text(encoding='utf-8')
    assert '0.43' not in page, 'the dashboard hard-codes the threshold'
