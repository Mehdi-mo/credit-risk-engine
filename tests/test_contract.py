"""The input contract, one decision per test (slice 9).

docs/slice_09_input_contract.md says what happens to a missing field, an
impossible value, an unseen category and a negative income, and why. These
tests are that document, executable. They need only models/, no data.
"""

import math

import pandas as pd
import pytest

from src.contract import FIELDS, IMPUTABLE, ContractError, validate
from src.economics import price_loans
from src.features import INQ_FILL


def _problems(engine, application):
    with pytest.raises(ContractError) as err:
        validate(application, engine.spec)
    return {p['field']: p['problem'] for p in err.value.problems}


def test_the_typical_application_is_clean(engine, typical):
    result = engine.decide(typical)
    assert result['decision'] in ('approve', 'decline')
    assert result['flags'] == {'imputed': [], 'outside_training_range': [], 'folded': [],
                               'guard_applied': False, 'credit_history_capped': False}
    assert result['review_recommended'] is False


# ---- a missing field ----------------------------------------------------

@pytest.mark.parametrize('field', [f for f in FIELDS if f not in IMPUTABLE])
def test_missing_required_field_is_rejected(engine, typical, field):
    application = {k: v for k, v in typical.items() if k != field}
    assert field in _problems(engine, application)
    # and null is the same as absent
    assert field in _problems(engine, {**typical, field: None})


@pytest.mark.parametrize('field', IMPUTABLE)
def test_blank_imputable_field_is_filled_as_training_filled_it(engine, typical, field):
    blank = engine.decide({**typical, field: None})
    assert blank['flags']['imputed'] == [field]

    # training filled mths_since_recent_inq with the "longest ago" value and
    # every other blank with the train median. Sending that value explicitly
    # must give the same features - except the one flag that records the blank.
    fill = INQ_FILL if field == 'mths_since_recent_inq' else engine.spec['fill_values'][field]
    a = engine.features([validate({**typical, field: None}, engine.spec)[0]])
    b = engine.features([validate({**typical, field: fill}, engine.spec)[0]])
    cols = [c for c in engine.bundle['columns'] if c != 'mths_since_recent_inq_missing']
    assert (a[cols].to_numpy() == b[cols].to_numpy()).all()

    if field == 'mths_since_recent_inq':
        assert a['mths_since_recent_inq_missing'].iloc[0] == 1
        assert b['mths_since_recent_inq_missing'].iloc[0] == 0


def test_only_columns_that_were_blank_in_training_are_imputable(engine):
    # the rule, not the list: nothing is filled that training never filled
    assert set(IMPUTABLE) <= set(engine.spec['fill_values'])
    assert len(IMPUTABLE) == 10


# ---- an impossible value ------------------------------------------------

@pytest.mark.parametrize('field,value', [
    ('annual_inc', -40_000), ('annual_inc', 0),          # a negative or zero income
    ('loan_amnt', 0), ('loan_amnt', -5_000),
    ('dti', -1), ('open_acc', -2), ('revol_bal', -10),
    ('fico_range_low', 250), ('fico_range_high', 900),
    ('emp_length', 11), ('emp_length', 2.5),
    ('pct_tl_nvr_dlq', 101), ('term', 48), ('total_acc', 0),
])
def test_impossible_value_is_rejected(engine, typical, field, value):
    assert field in _problems(engine, {**typical, field: value})


@pytest.mark.parametrize('value', ['65000', True, float('nan'), float('inf'), [1], {}])
def test_a_number_must_be_a_number(engine, typical, value):
    application = {**typical, 'annual_inc': value}
    if isinstance(value, float) and math.isnan(value):
        # NaN is a blank, and income was never blank in training
        assert 'missing' in _problems(engine, application)['annual_inc']
    else:
        assert 'annual_inc' in _problems(engine, application)


def test_fields_that_contradict_each_other_are_rejected(engine, typical):
    assert 'fico_range_high' in _problems(engine, {**typical, 'fico_range_high': 600})
    assert 'earliest_cr_line' in _problems(engine, {**typical, 'earliest_cr_line': '2020-01-01'})
    assert 'open_acc' in _problems(engine, {**typical, 'open_acc': 40})
    assert 'issue_d' in _problems(engine, {**typical, 'issue_d': 'last Tuesday'})


def test_every_problem_is_reported_at_once(engine, typical):
    application = {**typical, 'annual_inc': -1, 'term': 48, 'purpose': 'educational'}
    assert set(_problems(engine, application)) == {'annual_inc', 'term', 'purpose'}


# ---- an unseen category, an unknown field -------------------------------

@pytest.mark.parametrize('field,value', [
    ('purpose', 'educational'), ('home_ownership', 'NONE'), ('home_ownership', 'rent'),
    ('verification_status', 'verified'), ('application_type', 'Group'), ('purpose', 7),
])
def test_unseen_category_is_rejected_not_scored_as_the_base_level(engine, typical, field, value):
    assert field in _problems(engine, {**typical, field: value})


def test_rare_category_is_folded_as_training_folded_it(engine, typical):
    folded = engine.decide({**typical, 'purpose': 'wedding'})
    other = engine.decide({**typical, 'purpose': 'other'})
    assert folded['flags']['folded'] == [{'field': 'purpose', 'value': 'wedding',
                                          'scored_as': 'other'}]
    assert folded['probability_of_default'] == other['probability_of_default']


def test_unknown_field_is_rejected(engine, typical):
    # 'emp_lenght' would otherwise be dropped and emp_length silently imputed
    application = {k: v for k, v in typical.items() if k != 'emp_length'}
    application['emp_lenght'] = 7
    assert _problems(engine, application) == {'emp_lenght': 'unknown field'}


# ---- possible, but outside what the model saw ---------------------------

def test_value_outside_training_range_is_scored_and_flagged(engine, typical):
    lo, hi = engine.spec['train_range']['fico_range_low']
    result = engine.decide({**typical, 'fico_range_low': 600, 'fico_range_high': 604})
    assert 600 < lo
    assert result['decision'] in ('approve', 'decline')
    assert {f['field'] for f in result['flags']['outside_training_range']} == \
        {'fico_range_low', 'fico_range_high'}
    assert result['review_recommended'] is True


def test_the_guard_caps_loan_to_income_and_says_so(engine, typical):
    # the slice 5 row: a loan 19.5 times the income
    wild = engine.decide({**typical, 'loan_amnt': 39_000, 'annual_inc': 2_000})
    # just inside the training range: an income a hair above loan / cap
    at_cap = engine.decide({**typical, 'loan_amnt': 39_000,
                            'annual_inc': 39_000 / engine.guard_value * 1.001})
    assert wild['flags']['guard_applied'] is True
    assert wild['review_recommended'] is True
    assert wild['probability_of_default'] < 1.0
    assert at_cap['flags']['guard_applied'] is False
    assert wild['probability_of_default'] > at_cap['probability_of_default'] * 0.999


# ---- the price ----------------------------------------------------------

def test_the_price_never_reaches_the_model(engine, typical):
    without = engine.decide(typical)
    cheap = engine.decide({**typical, 'int_rate': 6.0})
    dear = engine.decide({**typical, 'int_rate': 28.0})
    assert cheap['probability_of_default'] == dear['probability_of_default'] \
        == without['probability_of_default']
    assert cheap['decision'] == dear['decision'] == without['decision']
    assert 'break_even' not in without


def test_break_even_is_slice_8s_number(engine, typical):
    result = engine.decide({**typical, 'int_rate': 13.5})
    priced = price_loans(pd.DataFrame({'int_rate': [13.5], 'term': [typical['term']],
                                       'funded_amnt': [typical['loan_amnt']]}),
                         engine.economics).iloc[0]
    assert result['break_even']['break_even_probability'] == float(priced['p_star'])
