"""From a loan application to a decision (slice 9).

This is the whole path, and every step of it is somebody else's function:

    src/contract.py    validate()            reject, impute-and-flag, score-and-flag
    src/features.py    transform()           the slice 3 build, replayed
    src/predict.py     predict_calibrated()  the slice 5 chain, guard on
    src/economics.py   price_loans()         the slice 8 break-even
    src/reasons.py     explain()             the slice 7 contributions

Nothing is computed here that is computed anywhere else. That is the rule
slice 9 exists to keep: the service in service/app.py imports this file, the
tests import this file, and neither holds a copy of anything above.

WHAT IS LOADED, AND FROM WHERE
    models/calibrated_logreg.joblib   scaler, model, calibrator        slice 5
    models/feature_spec.json          medians, levels, ranges          slice 9
    models/economics.json             margin, exposure, LGD            slice 8
    models/policy.json                the threshold and why            slice 9
Four files, all under models/, all in git. The service needs no data.

VERSIONS
Each file's version is the first 12 characters of the SHA-256 of its bytes.
A date or a hand-typed number can be forgotten; a hash cannot. If any byte of
the model changes, the version in every response changes with it.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .contract import FIELDS, ContractError, validate
from .economics import price_loans
from .features import CREDIT_HISTORY_CAP, transform
from .predict import GUARD_COLUMN, predict_calibrated
from .reasons import explain

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'

BUNDLE_FILE = 'calibrated_logreg.joblib'
SPEC_FILE = 'feature_spec.json'
ECONOMICS_FILE = 'economics.json'
POLICY_FILE = 'policy.json'


def _version(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _frame(clean_rows):
    """Validated applications as the frame the feature build expects."""
    df = pd.DataFrame(clean_rows, columns=list(FIELDS))
    for name, f in FIELDS.items():
        if f['kind'] in ('number', 'choice'):
            df[name] = df[name].astype('float64')      # None -> NaN
        elif f['kind'] == 'date':
            df[name] = pd.to_datetime(df[name])
    return df


class Engine:
    """Everything loaded once; decide() is then pure arithmetic."""

    def __init__(self, models_dir=MODELS):
        models_dir = Path(models_dir)
        self.bundle = joblib.load(models_dir / BUNDLE_FILE)
        self.spec = json.loads((models_dir / SPEC_FILE).read_text(encoding='utf-8'))
        self.economics = json.loads((models_dir / ECONOMICS_FILE).read_text(encoding='utf-8'))
        self.policy = json.loads((models_dir / POLICY_FILE).read_text(encoding='utf-8'))

        self.threshold = float(self.policy['threshold'])
        self.apply_guard = bool(self.policy['apply_loan_to_income_guard'])
        self.guard_value = float(self.bundle['loan_to_income_guard_not_applied'])

        self.versions = {
            'model_version': f"calibrated_logreg@{_version(models_dir / BUNDLE_FILE)}",
            'feature_spec_version': f"feature_spec@{_version(models_dir / SPEC_FILE)}",
            'economics_version': f"economics@{_version(models_dir / ECONOMICS_FILE)}",
            'policy_version': f"policy@{_version(models_dir / POLICY_FILE)}",
        }

        # the price must never reach the model: slice 3 held it out on purpose
        assert 'int_rate' not in self.bundle['columns']
        # and the build must produce what the model was trained on
        typical = self.features([validate(self.spec['typical_application'], self.spec)[0]])
        missing = set(self.bundle['columns']) - set(typical.columns)
        assert not missing, f'the feature build does not produce {sorted(missing)}'

    # ---- the steps ------------------------------------------------------

    def features(self, clean_rows):
        return transform(_frame(clean_rows), self.spec)

    def probabilities(self, features):
        return predict_calibrated(self.bundle, features, apply_guard=self.apply_guard)

    def _break_even(self, clean, p):
        """What this loan's own price says. Information, not the rule in force."""
        priced = price_loans(pd.DataFrame({'int_rate': [clean['int_rate']],
                                           'term': [clean['term']],
                                           'funded_amnt': [clean['loan_amnt']]}),
                             self.economics).iloc[0]
        return {
            'int_rate': clean['int_rate'],
            'margin_if_repaid': float(priced['margin_pct']),
            'loss_if_default': float(priced['loss_pct']),
            'break_even_probability': float(priced['p_star']),
            'per_loan_rule_would': 'approve' if p < priced['p_star'] else 'decline',
            'note': 'not the rule in force - see why_not_per_loan in GET /version',
        }

    # ---- the path -------------------------------------------------------

    def decide_many(self, applications, with_reasons=False):
        """Applications in, one result each, in order.

        A result is either a decision or {'rejected': [...]}: one bad
        application does not stop the others being scored.
        """
        results = [None] * len(applications)
        clean_rows, flag_rows, positions = [], [], []

        for i, application in enumerate(applications):
            try:
                clean, flags = validate(application, self.spec)
            except ContractError as err:
                results[i] = {'rejected': err.problems}
                continue
            clean_rows.append(clean)
            flag_rows.append(flags)
            positions.append(i)

        if not clean_rows:
            return results

        features = self.features(clean_rows)
        p = self.probabilities(features)
        reasons = (explain(self.bundle, features, clean_rows, self.apply_guard,
                           top=int(self.policy['reasons_returned']))
                   if with_reasons else None)

        lti = features[GUARD_COLUMN].to_numpy()
        history = features['credit_history_months'].to_numpy()

        for k, i in enumerate(positions):
            clean, flags = clean_rows[k], dict(flag_rows[k])
            flags['guard_applied'] = bool(self.apply_guard and lti[k] > self.guard_value)
            flags['credit_history_capped'] = bool(history[k] >= CREDIT_HISTORY_CAP)

            result = {
                # p < threshold, the same comparison src/profit.py uses
                'decision': 'approve' if p[k] < self.threshold else 'decline',
                'probability_of_default': float(p[k]),
                'threshold': self.threshold,
                'rule': self.policy['rule'],
                **self.versions,
                'flags': flags,
                # a person should look when the model was asked about something
                # it never saw. The decision is still returned; it is flagged.
                'review_recommended': bool(flags['outside_training_range']
                                           or flags['guard_applied']),
                'threshold_assumption': self.policy['assumption'],
            }
            if clean.get('int_rate') is not None:
                result['break_even'] = self._break_even(clean, p[k])
            if with_reasons:
                result['reasons'] = reasons[k]
                result['reasons_note'] = (
                    'Each factor is compared with a typical borrower in the training '
                    'data. A contribution is in log-odds; positive raises the '
                    'probability of default. It describes what the model weighed, '
                    'not what would happen if the factor changed.')
            results[i] = result

        return results

    def decide(self, application, with_reasons=False):
        return self.decide_many([application], with_reasons)[0]


def to_application(row):
    """A row of the raw table as the JSON a caller would send.

    Used by the skew test and nowhere in the scoring path: NaN becomes null,
    a timestamp becomes YYYY-MM-DD, and only the contract's fields are kept.
    """
    out = {}
    for name, f in FIELDS.items():
        value = row[name]
        if f['kind'] == 'date':
            out[name] = pd.Timestamp(value).date().isoformat()
        elif f['kind'] == 'category':
            out[name] = None if pd.isna(value) else str(value)
        else:
            out[name] = None if pd.isna(value) else float(value)
    return out
