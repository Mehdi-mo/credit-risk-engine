"""The request log (slice 10).

Slice 9 ended with: "No request log. It is the first thing slice 10 needs."
A monitor can only count what was written down, so every decision the service
makes is appended here as one line of JSON - the application as it arrived,
and what the service said about it.

WHAT IS LOGGED, AND WHY THE WHOLE APPLICATION
Drift is a change in what comes IN. To measure it later the monitor rebuilds
the model's features from the logged application, with the same functions the
service used (src/monitor.py calls src/decide.py - it does not rebuild them
its own way). So the application is stored as sent, not as features.

That makes this file personal data in any real use: it holds incomes and
credit files. Here the service is a demonstration and the log is OFF unless
CRE_REQUEST_LOG names a file. data/ is gitignored, and so is anything under it.

Only the standard library is used, so the service image needs nothing new.
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ENV_VAR = 'CRE_REQUEST_LOG'

VERSION_KEYS = ['model_version', 'feature_spec_version',
                'economics_version', 'policy_version']


def make_record(application, result, ts=None):
    """One log line. Small on purpose: everything else can be recomputed
    from the application and the versions."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    record = {'ts': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
              'application': application}

    if 'rejected' in result:
        record['rejected'] = result['rejected']
        return record

    record.update({
        'decision': result['decision'],
        'probability_of_default': result['probability_of_default'],
        'threshold': result['threshold'],
        'flags': result['flags'],
        'review_recommended': result['review_recommended'],
        **{k: result[k] for k in VERSION_KEYS},
    })
    reasons = result.get('reasons')
    if reasons:
        # the factor names only; the numbers can be recomputed from the application
        record['raised_risk'] = [r['factor'] for r in reasons['raised_risk']]
    return record


class RequestLog:
    """Appends one JSON line per decision. Safe to call from several threads."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._warned = False

    def write(self, application, result, ts=None):
        line = json.dumps(make_record(application, result, ts), default=str)
        try:
            with self._lock, open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError as err:
            # a full disk must not turn into a refused loan decision. But a
            # silent gap in the log is a monitoring failure, so say it once.
            if not self._warned:
                print(f'request log: could not write to {self.path}: {err}', file=sys.stderr)
                self._warned = True


def from_environment():
    """The log named by CRE_REQUEST_LOG, or None when logging is off."""
    path = os.environ.get(ENV_VAR)
    return RequestLog(path) if path else None


def read_log(path):
    """Every record in a log file, in the order it was written."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
