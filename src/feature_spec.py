"""Write down what the feature build learned, and prove the record is right (slice 9).

THE GAP THIS FILE CLOSES
The model bundle holds a scaler, a model and a calibrator. It does not hold
the feature build. The medians that fill blanks, the level list of each
category and the choice of which columns get a log were computed in notebook
03, used once, and never saved. So until slice 9 nothing outside that notebook
could turn a raw application into the 71 features the model wants - the
project had a model it could not serve.

This file refits those values with the same functions (src/features.py), from
training rows only, and saves them to models/feature_spec.json.

THE PROOF
Refitting is only safe if it lands on the same numbers. So this file then
rebuilds every row of features_train.parquet and features_test.parquet from
the raw table through transform() - the function the service calls - and
asserts the result equals what slice 3 saved. Not the shapes: the values.
Notebook 03's own check compared shapes and column names only.

WHAT IS LEARNED FROM WHICH ROWS
  fill values, levels, log columns   the 20,902 rows before 2016-09-01.
                                     This is slice 3's decision and is kept:
                                     changing it would change the features.
  train_range, typical_application   the 16,134 rows before 2016-01-01, the
                                     rows the model's weights were fitted on.
                                     "Outside what the model saw" has to mean
                                     those rows - it is also where the
                                     loan_to_income guard value comes from.
Nothing is learned from the test rows. They are rebuilt and compared, which
is reporting.

Run with:  python -m src.feature_spec
Needs data/processed/model_table.parquet and sample_50k.parquet (slice 2).
"""

import json

import joblib
import numpy as np
import pandas as pd

from .contract import FIELDS
from .economics import PROC, ROOT
from .features import (add_logs, fit_feature_spec, prepare_rows, restrict_years,
                       split_by_time, transform)
from .tolerance import FLOAT_TOL, max_abs_diff

SPEC_PATH = ROOT / 'models' / 'feature_spec.json'

# notebook 02's row filter, repeated so the dates line up by position
LABELLED = ['Fully Paid', 'Charged Off', 'Default']


def load_spec(path=SPEC_PATH):
    return json.loads(path.read_text(encoding='utf-8'))


def load_raw_rows():
    """The raw table slice 3 started from: 81 columns, the target, two dates."""
    table = pd.read_parquet(PROC / 'model_table.parquet')
    sample = pd.read_parquet(PROC / 'sample_50k.parquet')
    labelled = sample[sample['loan_status'].isin(LABELLED)].reset_index(drop=True)

    # position-matching is only valid if both sides are the same length
    assert len(labelled) == len(table), \
        f'{len(labelled)} labelled rows but {len(table)} in model_table'

    table['issue_d'] = pd.to_datetime(labelled['issue_d'], format='%b-%Y')
    table['earliest_cr_line'] = pd.to_datetime(labelled['earliest_cr_line'], format='%b-%Y')

    # sorted and split exactly as slice 3 did, so rows match the saved files
    train, test, cut = split_by_time(restrict_years(table))
    return train, test, cut


def compare(rebuilt, saved, name):
    """Every column of the saved file against the rebuild, value by value."""
    assert set(rebuilt.columns) == set(saved.columns), \
        f'{name}: column sets differ: {set(rebuilt.columns) ^ set(saved.columns)}'
    assert len(rebuilt) == len(saved), f'{name}: {len(rebuilt)} rows vs {len(saved)}'

    worst, worst_col = 0.0, None
    for col in saved.columns:
        a, b = rebuilt[col], saved[col]
        if pd.api.types.is_numeric_dtype(b):
            d = max_abs_diff(a, b)
            if d > worst:
                worst, worst_col = d, col
        else:
            assert (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all(), \
                f'{name}: column {col} differs'
    return worst, worst_col


if __name__ == '__main__':
    train_raw, test_raw, cut = load_raw_rows()
    assert train_raw['issue_d'].is_monotonic_increasing, 'train rows are not in issue_d order'
    assert test_raw['issue_d'].is_monotonic_increasing, 'test rows are not in issue_d order'
    assert len(train_raw) == 20_902 and len(test_raw) == 5_378, \
        f'{len(train_raw)} train and {len(test_raw)} test rows, expected 20,902 and 5,378'
    print(f'raw rows: {len(train_raw):,} train, {len(test_raw):,} test, cut {cut.date()}')

    # ---- fit, on train only ---------------------------------------------
    spec = fit_feature_spec(add_logs(prepare_rows(train_raw)))

    bundle = joblib.load(ROOT / 'models' / 'calibrated_logreg.joblib')
    inner = train_raw[train_raw['issue_d'] < pd.Timestamp(bundle['split_date'])]
    assert len(inner) == 16_134, f'{len(inner)} inner train rows, expected 16,134'

    numeric = [n for n, f in FIELDS.items() if f['kind'] == 'number']
    spec['train_range'] = {n: [float(inner[n].min()), float(inner[n].max())] for n in numeric}

    # a made-up applicant for the dashboard and the docs: the middle value of
    # every field. Not a real person - no row of the data is shipped.
    typical = {}
    for name, f in FIELDS.items():
        if f['kind'] == 'number':
            typical[name] = float(inner[name].median())
        elif f['kind'] in ('category', 'choice'):
            mode = inner[name].mode().iloc[0]
            typical[name] = float(mode) if f['kind'] == 'choice' else str(mode)
    typical['issue_d'] = '2016-01-01'
    months = int(prepare_rows(inner)['credit_history_months'].median())
    typical['earliest_cr_line'] = str(
        (pd.Timestamp('2016-01-01') - pd.DateOffset(months=months)).date())
    typical['fico_range_high'] = typical['fico_range_low'] + 4     # always 4 apart
    spec['typical_application'] = typical

    spec = {
        'what': 'everything the feature build learns, fitted on training rows only',
        'written_by': 'python -m src.feature_spec',
        'fitted_on': f'{len(train_raw):,} rows before {cut.date()} (fills, levels, logs); '
                     f'{len(inner):,} rows before {bundle["split_date"]} (ranges, typical)',
        **spec,
    }

    print(f"log columns : {spec['log_columns']}")
    print(f"fill values : {len(spec['fill_values'])} medians")
    for col, levels in spec['levels'].items():
        print(f'levels      : {col} -> {levels}')

    # ---- the proof: transform() against what slice 3 saved ---------------
    print(f'\n---- rebuild through transform() vs the saved files '
          f'(tolerance {FLOAT_TOL:g}) ----')
    for name, raw in (('train', train_raw), ('test', test_raw)):
        saved = pd.read_parquet(PROC / f'features_{name}.parquet')
        assert np.array_equal(raw['target'].to_numpy(), saved['target'].to_numpy()), \
            f'{name}: raw rows and saved features are not in the same row order'
        worst, col = compare(transform(raw, spec), saved, name)
        print(f'features_{name}.parquet   {len(saved):>6,} rows x {saved.shape[1]} columns   '
              f'max difference {worst:.2e}' + (f'  ({col})' if col else ''))
        assert worst <= FLOAT_TOL, \
            f'{name}: the rebuild differs from the saved file by {worst:.2e} in {col}'

    # ---- and the record survives being written and read back -------------
    # newline fixed so the file, and so its version hash, is the same on any OS
    SPEC_PATH.write_text(json.dumps(spec, indent=2), encoding='utf-8', newline='\n')
    reread = load_spec()
    saved = pd.read_parquet(PROC / 'features_test.parquet')
    worst, col = compare(transform(test_raw, reread), saved, 'test, spec read from disk')
    assert worst <= FLOAT_TOL, f'the spec does not survive a round trip through JSON ({col})'

    print(f'\nwrote {SPEC_PATH.relative_to(ROOT)}')
    print('the saved spec rebuilds both feature files')
