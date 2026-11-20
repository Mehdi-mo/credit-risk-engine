"""The input contract of the scoring service (slice 9).

The reasoning is in docs/slice_09_input_contract.md, which was written before
this file. This file is the contract as code: one table, one function.

A loan application is a flat set of fields with the same names as the columns
the model was built from. There is no translation layer, on purpose - a second
vocabulary is a second place to be wrong.

EVERY FIELD GETS ONE OF THREE ANSWERS

  REJECT            the application is not scored. Used when the value is
                    impossible, or when scoring it would mean inventing
                    something training never invented.
  IMPUTE AND FLAG   the blank is filled exactly as training filled it, and
                    the response says which fields were filled.
  SCORE AND FLAG    the value is possible but outside what the model saw.
                    It is scored, and the response says so.

Nothing is ever fixed silently. The one silent path this file closes: pandas
turns an unknown category into "no dummy set", which scores it as the base
level - purpose 'educational' would quietly become 'car'. Here it is rejected.

Nothing in this file is learned from data. The training ranges and the level
lists come from models/feature_spec.json, which src/feature_spec.py fits on
training rows only.
"""

import math
from datetime import date, datetime

from .columns import CAT_FOLD_RARE

# ---- the table ----------------------------------------------------------
# kind      what the value is
# lo, hi    HARD limits: outside them the value is impossible -> reject.
#           None means no hard limit on that side. These are facts about the
#           world (a count cannot be negative), not about the training data.
# blank     'reject' or 'impute'. 'impute' only where training itself had
#           blanks in that column and filled them - the service does to a
#           blank what training did to a blank, and nothing more.
# whole     the value must be a whole number

def _n(label, lo=0, hi=None, blank='reject', whole=False):
    return {'kind': 'number', 'label': label, 'lo': lo, 'hi': hi,
            'blank': blank, 'whole': whole}


def _count(label, lo=0, hi=None, blank='reject'):
    return _n(label, lo=lo, hi=hi, blank=blank, whole=True)


def _cat(label):
    return {'kind': 'category', 'label': label, 'blank': 'reject'}


def _date(label):
    return {'kind': 'date', 'label': label, 'blank': 'reject'}


FIELDS = {
    # ---- the loan ----
    'loan_amnt':        _n('Loan amount requested', lo=1),
    'term':             {'kind': 'choice', 'label': 'Term in months',
                         'choices': [36, 60], 'blank': 'reject'},
    'purpose':          _cat('Purpose of the loan'),
    'application_type': _cat('Individual or joint application'),
    'issue_d':          _date('Application date'),

    # ---- the applicant ----
    'annual_inc':          _n('Annual income', lo=1),
    'emp_length':          _count('Years in current job (10 = ten or more)',
                                  hi=10, blank='impute'),
    'home_ownership':      _cat('Home ownership'),
    'verification_status': _cat('Whether the income was verified'),
    'dti':                 _n('Debt payments as % of income', blank='impute'),

    # ---- the credit file: score and age ----
    'fico_range_low':   _count('Credit score, low end of range', lo=300, hi=850),
    'fico_range_high':  _count('Credit score, high end of range', lo=300, hi=850),
    'earliest_cr_line': _date('Date of the oldest credit account'),

    # ---- the credit file: accounts ----
    'open_acc':               _count('Open credit accounts'),
    'total_acc':              _count('Credit accounts ever held', lo=1),
    'acc_open_past_24mths':   _count('Accounts opened in the last 24 months'),
    'mort_acc':               _count('Mortgage accounts'),
    'num_actv_bc_tl':         _count('Active bankcard accounts'),
    'num_actv_rev_tl':        _count('Active revolving accounts'),
    'num_bc_sats':            _count('Bankcard accounts in good standing'),
    'num_bc_tl':              _count('Bankcard accounts ever held'),
    'num_il_tl':              _count('Instalment accounts ever held'),
    'num_op_rev_tl':          _count('Open revolving accounts'),
    'num_rev_accts':          _count('Revolving accounts ever held'),
    'num_rev_tl_bal_gt_0':    _count('Revolving accounts with a balance'),
    'num_sats':               _count('Accounts in good standing'),
    'num_tl_op_past_12m':     _count('Accounts opened in the last 12 months'),

    # ---- the credit file: balances and limits ----
    'revol_bal':                  _n('Revolving balance'),
    'revol_util':                 _n('Revolving credit used, %', blank='impute'),
    'tot_cur_bal':                _n('Total current balance, all accounts'),
    'avg_cur_bal':                _n('Average balance per account'),
    'total_rev_hi_lim':           _n('Total revolving credit limit'),
    'tot_hi_cred_lim':            _n('Total credit limit, all accounts'),
    'total_bal_ex_mort':          _n('Total balance excluding mortgage'),
    'total_bc_limit':             _n('Total bankcard limit'),
    'total_il_high_credit_limit': _n('Total instalment credit limit'),
    'bc_open_to_buy':             _n('Unused bankcard credit', blank='impute'),
    'bc_util':                    _n('Bankcard credit used, %', blank='impute'),
    'percent_bc_gt_75':           _n('% of bankcards above 75% of limit',
                                     hi=100, blank='impute'),

    # ---- the credit file: how recent things are, in months ----
    'mo_sin_old_il_acct':    _count('Months since oldest instalment account opened',
                                    blank='impute'),
    'mo_sin_old_rev_tl_op':  _count('Months since oldest revolving account opened'),
    'mo_sin_rcnt_rev_tl_op': _count('Months since newest revolving account opened'),
    'mo_sin_rcnt_tl':        _count('Months since newest account opened'),
    'mths_since_recent_inq': _count('Months since the last credit inquiry '
                                    '(blank = none on record)', blank='impute'),
    'inq_last_6mths':        _count('Credit inquiries in the last 6 months'),

    # ---- the credit file: trouble ----
    'delinq_2yrs':                _count('Delinquencies in the last 2 years'),
    'acc_now_delinq':             _count('Accounts delinquent right now'),
    'delinq_amnt':                _n('Amount currently past due'),
    'num_accts_ever_120_pd':      _count('Accounts ever 120+ days past due'),
    'num_tl_120dpd_2m':           _count('Accounts 120 days past due, last 2 months',
                                         blank='impute'),
    'num_tl_30dpd':               _count('Accounts 30 days past due right now'),
    'num_tl_90g_dpd_24m':         _count('Accounts 90+ days past due, last 24 months'),
    'pct_tl_nvr_dlq':             _n('% of accounts never delinquent',
                                     hi=100, blank='impute'),
    'collections_12_mths_ex_med': _count('Collections in the last 12 months'),
    'chargeoff_within_12_mths':   _count('Charge-offs in the last 12 months'),
    'tot_coll_amt':               _n('Total ever owed to collections'),
    'pub_rec':                    _count('Public records'),
    'pub_rec_bankruptcies':       _count('Bankruptcies on public record'),
    'tax_liens':                  _count('Tax liens'),
}

# Not part of the application, not seen by the model (slice 3 held the price
# out of the features on purpose). If the caller sends it, the response adds
# the break-even probability for that price - as information, not as the rule.
OPTIONAL_FIELDS = {
    'int_rate': _n('Interest rate offered, % per year (optional)', lo=0, hi=100),
}

IMPUTABLE = [name for name, f in FIELDS.items() if f['blank'] == 'impute']


class ContractError(ValueError):
    """The application broke the contract. .problems lists every reason."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__('; '.join(f"{p['field']}: {p['problem']}" for p in problems))


def allowed_categories(spec):
    """What each category field may say: the levels training saw, plus the
    rare levels training folded into one of them. Read from the spec and from
    columns.py, so this list cannot drift from the build."""
    allowed = {col: list(levels) for col, levels in spec['levels'].items()}
    for col, (rare, _into) in CAT_FOLD_RARE.items():
        allowed[col] = allowed[col] + [r for r in rare if r not in allowed[col]]
    return allowed


def _is_blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def validate(application, spec):
    """One application in. A clean row and its flags out, or ContractError.

    Every problem is collected before raising, so a caller fixes an
    application once instead of one field per attempt.
    """
    if not isinstance(application, dict):
        raise ContractError([{'field': '(body)', 'problem': 'must be a JSON object'}])

    problems, row = [], {}
    flags = {'imputed': [], 'outside_training_range': [], 'folded': []}
    allowed = allowed_categories(spec)
    known = {**FIELDS, **OPTIONAL_FIELDS}

    # an unknown name is rejected, not ignored: a misspelt optional field
    # would otherwise be imputed and nobody would ever find out
    for name in application:
        if name not in known:
            problems.append({'field': name, 'problem': 'unknown field'})

    for name, f in known.items():
        value = application.get(name)
        optional = name in OPTIONAL_FIELDS

        if _is_blank(value):
            if optional:
                continue
            if f['blank'] == 'impute':
                row[name] = None
                flags['imputed'].append(name)
            else:
                problems.append({'field': name, 'problem': 'missing, and this field '
                                 'was never blank in training, so there is no honest fill'})
            continue

        if f['kind'] in ('number', 'choice'):
            # bool is an int in Python; "5" is not a number. Neither is guessed at.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append({'field': name, 'problem': f'must be a number, got {value!r}'})
                continue
            value = float(value)
            if not math.isfinite(value):
                problems.append({'field': name, 'problem': 'must be a finite number'})
                continue

        if f['kind'] == 'choice':
            if value not in f['choices']:
                problems.append({'field': name,
                                 'problem': f"must be one of {f['choices']}, got {value:g}"})
                continue
            row[name] = value

        elif f['kind'] == 'number':
            if f['whole'] and value != int(value):
                problems.append({'field': name, 'problem': f'must be a whole number, got {value:g}'})
                continue
            if f['lo'] is not None and value < f['lo']:
                problems.append({'field': name, 'problem':
                                 f"must be at least {f['lo']:g}, got {value:g}"})
                continue
            if f['hi'] is not None and value > f['hi']:
                problems.append({'field': name, 'problem':
                                 f"must be at most {f['hi']:g}, got {value:g}"})
                continue
            row[name] = value

            # possible, but did the model ever see anything like it?
            if not optional:
                lo, hi = spec['train_range'][name]
                if value < lo or value > hi:
                    flags['outside_training_range'].append(
                        {'field': name, 'value': value, 'train_min': lo, 'train_max': hi})

        elif f['kind'] == 'category':
            if not isinstance(value, str) or value not in allowed[name]:
                problems.append({'field': name, 'problem':
                                 f'must be one of {allowed[name]}, got {value!r}'})
                continue
            row[name] = value
            if name in CAT_FOLD_RARE and value in CAT_FOLD_RARE[name][0]:
                flags['folded'].append({'field': name, 'value': value,
                                        'scored_as': CAT_FOLD_RARE[name][1]})

        elif f['kind'] == 'date':
            parsed = _parse_date(value)
            if parsed is None:
                problems.append({'field': name, 'problem':
                                 f'must be a date like 2016-09-01, got {value!r}'})
                continue
            row[name] = parsed

    # ---- checks that need two fields ------------------------------------
    def both(a, b):
        return row.get(a) is not None and row.get(b) is not None

    if both('fico_range_low', 'fico_range_high') and row['fico_range_high'] < row['fico_range_low']:
        problems.append({'field': 'fico_range_high',
                         'problem': 'is below fico_range_low'})
    if both('issue_d', 'earliest_cr_line') and row['earliest_cr_line'] > row['issue_d']:
        problems.append({'field': 'earliest_cr_line',
                         'problem': 'is after the application date, which would '
                                    'make the credit history negative'})
    if both('open_acc', 'total_acc') and row['open_acc'] > row['total_acc']:
        problems.append({'field': 'open_acc',
                         'problem': 'is larger than total_acc (accounts open now '
                                    'cannot outnumber accounts ever held)'})

    if problems:
        raise ContractError(problems)
    return row, flags


def describe(spec):
    """The contract as rows, for GET /contract and for the docs table."""
    allowed = allowed_categories(spec)
    rows = []
    for name, f in {**FIELDS, **OPTIONAL_FIELDS}.items():
        optional = name in OPTIONAL_FIELDS
        if f['kind'] == 'number':
            lo = '' if f['lo'] is None else f"{f['lo']:g}"
            hi = '' if f['hi'] is None else f"{f['hi']:g}"
            accepts = f"{'whole number' if f['whole'] else 'number'} {lo or '-inf'} .. {hi or 'no limit'}"
        elif f['kind'] == 'choice':
            accepts = ' or '.join(str(c) for c in f['choices'])
        elif f['kind'] == 'category':
            accepts = ', '.join(allowed[name])
        else:
            accepts = 'date, YYYY-MM-DD'
        choices = (f['choices'] if f['kind'] == 'choice'
                   else allowed[name] if f['kind'] == 'category' else None)
        rows.append({
            'field': name,
            'meaning': f['label'],
            'kind': f['kind'],
            'choices': choices,
            'accepts': accepts,
            'if_blank': 'left out' if optional else f['blank'],
            'seen_in_training': (None if optional or f['kind'] != 'number'
                                 else spec['train_range'][name]),
        })
    return rows
