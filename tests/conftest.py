"""Shared pieces for the slice 9 tests.

The service itself needs only models/. These tests also need the processed
data from slices 2-8, which is gitignored - so a test that needs a file that
is not there is skipped with the reason, not failed.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.decide import Engine, to_application
from src.economics import PROC

NEEDED = ['model_table.parquet', 'sample_50k.parquet', 'features_train.parquet',
          'features_test.parquet', 'predictions_test_slice9.parquet']


@pytest.fixture(scope='session')
def engine():
    return Engine()


@pytest.fixture(scope='session')
def client():
    from service.app import app
    return TestClient(app)


@pytest.fixture(scope='session')
def typical(engine):
    return dict(engine.spec['typical_application'])


@pytest.fixture(scope='session')
def raw_rows():
    """The raw train and test rows slice 3 started from, in slice 3's order."""
    missing = [f for f in NEEDED if not (PROC / f).exists()]
    if missing:
        pytest.skip(f'data/processed is missing {missing} - rebuild with slices 2-8, '
                    f'then python -m src.score_test')
    from src.feature_spec import load_raw_rows
    train, test, _cut = load_raw_rows()
    return {'train': train, 'test': test}


@pytest.fixture(scope='session')
def test_applications(raw_rows):
    """Every test row as the JSON a caller would send. 5,378 of them."""
    return [to_application(row) for _, row in raw_rows['test'].iterrows()]


@pytest.fixture(scope='session')
def published():
    return pd.read_parquet(PROC / 'predictions_test_slice9.parquet')
