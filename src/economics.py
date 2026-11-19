"""The economics of a lending decision (slice 8), written down before use.

THE RULE THIS FILE EXISTS TO KEEP
Every number here is fitted on train rows only (2013-01 to 2015-12). Nothing
is learned from valid or test. The test set has now been used five times,
every time for reporting, never for choosing, and this file does not break
that: it is fitted, saved to models/economics.json, and only then applied.

WHAT IS MEASURED AND WHAT IS ASSUMED
Measured, from realised cash flows on train loans:
  margin_pct   what a repaid loan actually returned, per euro funded
  ead_pct      principal still outstanding when a loan went bad
  int_pct      interest collected before it went bad - not a loss
  LGD          the share of that exposure never recovered
Assumed:
  nothing in the central case. The sensitivity analysis in src/profit.py
  then moves LGD and the margin away from their measured values on purpose,
  because measured-on-this-book is not the same as true-in-general.

WHY int_rate IS IN HERE AND NOT IN THE MODEL
A lender setting the price knows the price. Slice 3 held int_rate out of the
features so the model would judge the applicant instead of copying Lending
Club's grading. Pricing the decision is a different question from scoring the
applicant, and the answer to it is allowed to use the price. That is a
decision, not an oversight.

WHAT THIS DATA CANNOT SUPPORT
Only funded loans exist. Every threshold here is evaluated inside the
population Lending Club already approved; nothing can be said about the
applicants it turned away. And slice 3's censoring dropped loans that had not
yet resolved, so the book is survivor-shaped. Relative comparisons between
thresholds are sound. Absolute profit is not.

THE SHAPE OF THE MODEL
A repaid loan returns   margin_pct x funded
A defaulted loan loses  (ead_pct x LGD - int_pct) x funded

The second line is an identity, not a fit: src/money.py asserts that
net_pct = int_pct - ead_pct x LGD holds to 1e-6 on every defaulted loan. It
is written in this form so LGD stays a single knob the sensitivity analysis
can turn, which a directly-measured net loss would not give.

margin_pct, ead_pct and int_pct are straight lines in int_rate and a
60-month flag. Cell means by grade x term were the alternative and were
rejected: G/36-month has 14 train loans and A/60-month has 60, so those
cells are noise, and the sensitivity surface needs something smooth.
Fit quality against the 14 cell means is printed by the self-check below.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
PARAMS_PATH = ROOT / 'models' / 'economics.json'

# the three fitted quantities, and what each is fitted on
FITS = {
    'margin_pct': ('target == 0', 'net_pct'),    # repaid loans
    'ead_pct':    ('target == 1', 'ead_pct'),    # defaulted loans
    'int_pct':    ('target == 1', 'int_pct'),    # defaulted loans
}

TRAIN_SPLIT = 'train'


def _design(df):
    """Intercept, price, and a 60-month flag. Both known at decision time."""
    return np.column_stack([
        np.ones(len(df)),
        df['int_rate'].to_numpy(dtype='float64'),
        (df['term'] == 60).to_numpy(dtype='float64'),
    ])


def fit_economics(money, verbose=True):
    """Fit the economic parameters on train rows. Returns a plain dict."""
    train = money[money['split'] == TRAIN_SPLIT]
    assert len(train) == 16_134, f'train has {len(train)} rows, expected 16,134'
    assert train['issue_d'].max() < pd.Timestamp('2016-01-01'), \
        'a non-train row reached the economics fit'

    params = {'n_train': int(len(train)),
              'fitted_on': 'split == train (2013-01 .. 2015-12)'}

    for name, (query, column) in FITS.items():
        sub = train.query(query)
        X, y = _design(sub), sub[column].to_numpy(dtype='float64')
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        params[name] = {
            'intercept': float(coef[0]),
            'per_rate_point': float(coef[1]),
            'is_60_month': float(coef[2]),
            'n': int(len(sub)),
            'r2': float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        }

    # LGD is flat and is kept flat on purpose: across all 14 grade x term
    # cells its standard deviation is 0.015, which is noise, not structure
    bad = train[train['target'] == 1]
    params['lgd'] = {
        'value': float(bad['lgd'].mean()),
        'sd_across_grade_term_cells': float(
            bad.groupby(['grade', 'term'], observed=True)['lgd'].mean().std()),
        'n': int(len(bad)),
    }

    if verbose:
        for name in FITS:
            p = params[name]
            print(f"{name:11s} = {p['intercept']:+.5f} "
                  f"{p['per_rate_point']:+.5f}*rate "
                  f"{p['is_60_month']:+.5f}*is60     "
                  f"R2 {p['r2']:.3f}  n {p['n']:,}")
        print(f"lgd         = {params['lgd']['value']:.4f}  flat  "
              f"(sd across grade x term cells "
              f"{params['lgd']['sd_across_grade_term_cells']:.4f})")

    return params


def save_params(params, path=PARAMS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))
    return path


def load_params(path=PARAMS_PATH):
    return json.loads(Path(path).read_text())


def _line(p, rate, is60):
    return p['intercept'] + p['per_rate_point'] * rate + p['is_60_month'] * is60


def price_loans(df, params, lgd=None, margin_multiplier=1.0):
    """Per-loan economics. Adds what a loan earns, what it costs, and its
    own break-even probability.

    lgd and margin_multiplier exist so the sensitivity analysis can move the
    assumption without refitting anything. Left alone, both reproduce the
    measured central case exactly.
    """
    out = df.copy()
    rate = out['int_rate'].to_numpy(dtype='float64')
    is60 = (out['term'] == 60).to_numpy(dtype='float64')
    lgd = params['lgd']['value'] if lgd is None else float(lgd)

    out['margin_pct'] = _line(params['margin_pct'], rate, is60) * margin_multiplier
    out['loss_pct'] = (_line(params['ead_pct'], rate, is60) * lgd
                       - _line(params['int_pct'], rate, is60))

    out['gain_eur'] = out['margin_pct'] * out['funded_amnt']
    out['loss_eur'] = out['loss_pct'] * out['funded_amnt']

    # approve while the chance of default is below the point where the
    # expected gain stops covering the expected loss:
    #   (1-p)*gain = p*loss   ->   p* = gain / (gain + loss)
    # per loan, because the price is per loan. A single global threshold is
    # the special case where every loan is priced the same, and they are not.
    total = out['margin_pct'] + out['loss_pct']
    out['p_star'] = np.where(total > 0, out['margin_pct'] / total, 0.0)

    return out


if __name__ == '__main__':
    money = pd.read_parquet(PROC / 'money_table.parquet')

    print('---- fitted on train only ----')
    params = fit_economics(money)
    path = save_params(params)
    print(f'\nwrote {path.relative_to(ROOT)}')

    # does a smooth line in rate and term reproduce what each cell actually did?
    train = money[money['split'] == 'train']
    priced = price_loans(train, params)
    rows = []
    for (grade, term), sub in priced.groupby(['grade', 'term'], observed=True):
        good, bad = sub[sub['target'] == 0], sub[sub['target'] == 1]
        if len(good) < 5 or len(bad) < 5:
            continue
        rows.append({
            'grade': grade, 'term': int(term), 'n': len(sub),
            'margin_actual': good['net_pct'].mean(),
            'margin_fitted': sub['margin_pct'].mean(),
            'loss_actual': -bad['net_pct'].mean(),
            'loss_fitted': sub['loss_pct'].mean(),
        })
    cells = pd.DataFrame(rows)
    cells['margin_err'] = cells['margin_fitted'] - cells['margin_actual']
    cells['loss_err'] = cells['loss_fitted'] - cells['loss_actual']

    print('\n---- fitted vs what each grade x term cell actually did (train) ----')
    print(cells.round(4).to_string(index=False))
    print(f"\nmean absolute cell error   margin {cells['margin_err'].abs().mean():.4f}"
          f"   loss {cells['loss_err'].abs().mean():.4f}")

    # the headline this produces, before a single curve is drawn
    test = price_loans(money[money['split'] == 'test'], params)
    print('\n---- break-even probability per grade (test rows, train parameters) ----')
    print(test.groupby('grade', observed=True)
              .agg(n=('p_star', 'size'), margin=('margin_pct', 'mean'),
                   loss=('loss_pct', 'mean'), break_even=('p_star', 'mean'))
              .round(4).to_string())
