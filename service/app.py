"""The scoring service (slice 9).

This file is deliberately thin. It turns HTTP into a function call and a
result into JSON, and that is all. The contract, the feature build, the
scoring chain, the economics and the reasons live in src/ and are imported -
if any of them appeared here as well, that would be the bug this slice exists
to prevent. tests/test_service.py checks that the answers that come out over
HTTP are the same numbers src/ produces directly.

Run from the repository root:

    uvicorn service.app:app --port 8000

then open http://localhost:8000 for the dashboard, or /docs for the API.
"""

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse

from src import requestlog
from src.contract import IMPUTABLE, describe
from src.decide import Engine

HERE = Path(__file__).resolve().parent

engine = Engine()          # four files, loaded once, at start-up

# Off unless CRE_REQUEST_LOG names a file. src/monitor.py reads what it writes.
request_log = requestlog.from_environment()

app = FastAPI(
    title='Credit risk engine',
    version=engine.versions['model_version'],
    description='Scores a loan application: probability of default, the decision, '
                'and the reasons. A portfolio project on Lending Club data - not a '
                'lending system. See GET /version for what the threshold assumes.',
)

EXAMPLE = {'typical': {'summary': 'The middle value of every field in the training data',
                       'value': engine.spec['typical_application']}}


def _decide(application, with_reasons):
    # When the log is on, the reasons are worked out for every request, so the
    # monitor can count what the declines are made of. They only go back to
    # the caller who asked for them.
    result = engine.decide(application, with_reasons=with_reasons or request_log is not None)
    if request_log is not None:
        request_log.write(application, result)

    # 422: the request was understood and breaks the contract
    if 'rejected' in result:
        return JSONResponse(status_code=422, content=result)
    if not with_reasons:
        result = {k: v for k, v in result.items() if k not in ('reasons', 'reasons_note')}
    return result


@app.get('/health')
def health():
    return {'status': 'ok', 'request_log': request_log is not None, **engine.versions}


@app.get('/version')
def version():
    """Which model, which economics, which threshold - and what it assumes."""
    b = engine.bundle
    return {
        **engine.versions,
        'model': {
            'kind': f"L1 logistic regression, C={b['C']}, {b['n_features_used']} of "
                    f"{len(b['columns'])} features used, {b['calibration']}",
            'trained_on': b['trained_on'],
            'calibrated_on': b['calibrated_on'],
            'saved': b['saved'],
            'test_auc': b['test_auc'],
            'test_auc_ci95': b['test_auc_ci95'],
        },
        'policy': engine.policy,
        'economics_fitted_on': engine.economics['fitted_on'],
        'limits': [
            'Trained on loans Lending Club approved between 2013 and 2015. It has '
            'never seen an applicant Lending Club turned away.',
            'The threshold rests on a measured margin that is the weakest number in '
            'the project. Change the margin and the threshold moves between 0.24 and 0.55.',
            'An application outside the range of the training data is scored and '
            'flagged, with review_recommended set to true.',
        ],
    }


@app.get('/contract')
def contract():
    """Every field: what it means, what is accepted, what happens to a blank."""
    return {
        'unknown_field': 'reject',
        'impossible_value': 'reject',
        'unseen_category': 'reject',
        'blank': 'reject, except these, which are filled as training filled them and flagged',
        'blank_is_imputed_for': IMPUTABLE,
        'outside_training_range': 'score and flag',
        'fields': describe(engine.spec),
    }


@app.get('/example')
def example():
    """A made-up applicant: the middle value of every field. Not a real person."""
    return engine.spec['typical_application']


@app.post('/decision')
def decision(application: dict[str, Any] = Body(..., openapi_examples=EXAMPLE)):
    """The decision, the probability, the threshold it was compared against,
    and the versions of the model and the economics that produced it."""
    return _decide(application, with_reasons=False)


@app.post('/reasons')
def reasons(application: dict[str, Any] = Body(..., openapi_examples=EXAMPLE)):
    """The same decision, with why: the factors that raised this applicant's
    risk the most and the ones that lowered it, compared with a typical
    borrower. For a declined application these are the reasons for the decline."""
    return _decide(application, with_reasons=True)


@app.get('/', include_in_schema=False)
def dashboard():
    return FileResponse(HERE / 'dashboard.html')
