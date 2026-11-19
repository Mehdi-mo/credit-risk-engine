"""The money table for the credit risk engine (slice 8).

Slice 2 removed the cash-flow columns from the modelling table, correctly:
recoveries and total_rec_prncp only exist after a loan has gone bad, so a
model that sees them is cheating. But they were never deleted from
sample_50k.parquet, which still holds all 151 raw columns.

That means loss given default and exposure at default can be MEASURED here,
not assumed. Using realised cash to price the decision is not leakage: it is
the same outcome information the target itself is built from, used for
evaluation and never as a feature.

int_rate and installment come back the same way. Using the price for the
economics while excluding it as a feature is a decision, not an oversight -
a lender setting the price knows the price. The model must not see it; the
profit calculation must.

HOW THE ROWS ARE MATCHED
Not by a column join. sample_50k filtered to the three resolved statuses and
reindexed is model_table row for row - 29,619 = 29,619, every shared column
agreeing positionally at 1.0000. Slice 3's window and split then reproduce
the three splits exactly. An 8-column join on (issue_d, zip, state,
loan_amnt, term, annual_inc, dti, revol_bal) was tried first and lost 4 rows,
because slice 2 nulled 11 dti values that the raw sample still has. The
positional path loses nothing and carries the raw `id` forward as a real key.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'

# ---- the lineage, as constants so a silent change fails loudly ----------
RESOLVED_STATUSES = ['Fully Paid', 'Charged Off', 'Default']
BAD_STATUSES      = ['Charged Off', 'Default']

YEAR_MIN, YEAR_MAX = 2013, 2017          # src/features.py restrict_years
VALID_START = pd.Timestamp('2016-01-01')  # inner cut: train_inner | valid
TEST_START  = pd.Timestamp('2016-09-01')  # outer cut, = issue_d.quantile(0.8)

N_RESOLVED = 29_619     # == len(model_table.parquet)
N_WINDOW   = 26_280     # after 2013-2017
N_TRAIN, N_VALID, N_TEST = 16_134, 4_768, 5_378

# ---- what to pull out of the raw sample --------------------------------
KEY_COLUMNS = ['id', 'issue_d', 'zip_code', 'addr_state', 'loan_status']

# known at the moment of the decision: a lender setting the price knows it
PRICE_COLUMNS = ['int_rate', 'installment', 'grade', 'sub_grade',
                 'loan_amnt', 'funded_amnt', 'term']

# only true after the fact: for the economics, never for a feature
CASH_COLUMNS = ['total_pymnt', 'total_rec_prncp', 'total_rec_int',
                'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'out_prncp', 'last_pymnt_d']

TO_NUMERIC = ['int_rate', 'installment', 'loan_amnt', 'funded_amnt',
              'total_pymnt', 'total_rec_prncp', 'total_rec_int',
              'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
              'out_prncp']


def _months_between(later, earlier):
    """Whole months from `earlier` to `later`. NaT in, NaN out."""
    return ((later.dt.year - earlier.dt.year) * 12
            + (later.dt.month - earlier.dt.month))


def build_money_table(proc_dir=PROC, verbose=True):
    """Price and realised cash per loan, split-labelled, in slice 3's row order.

    Decides no economics. Every number here is either copied from the raw
    sample or is arithmetic on it. The assumptions live in slice 8's write-up.
    """
    def say(msg):
        if verbose:
            print(msg)

    raw = pd.read_parquet(proc_dir / 'sample_50k.parquet',
                          columns=KEY_COLUMNS + PRICE_COLUMNS + CASH_COLUMNS)

    # ---- 1. the lineage anchor ------------------------------------------
    df = raw[raw['loan_status'].isin(RESOLVED_STATUSES)].reset_index(drop=True)
    assert len(df) == N_RESOLVED, (
        f'lineage broke: {len(df)} resolved rows, expected {N_RESOLVED}. '
        'This file is only valid while it is model_table row for row.')
    say(f'resolved statuses          {len(df):>7,}   (model_table lineage)')

    # ---- 2. text to numbers ---------------------------------------------
    # the raw sample stores everything as text: " 60 months", "20.99"
    df['issue_d']      = pd.to_datetime(df['issue_d'],      format='%b-%Y', errors='coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
    for col in TO_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    assert df['issue_d'].notna().all(), 'an issue_d failed to parse'
    df['target'] = df['loan_status'].isin(BAD_STATUSES).astype('int8')

    # ---- 3. slice 3's window --------------------------------------------
    years = df['issue_d'].dt.year
    df = df[(years >= YEAR_MIN) & (years <= YEAR_MAX)].reset_index(drop=True)
    assert len(df) == N_WINDOW, f'window gave {len(df)}, expected {N_WINDOW}'
    say(f'{YEAR_MIN}-{YEAR_MAX} window          {len(df):>7,}')

    # ---- 4. slice 3's sort ----------------------------------------------
    # mergesort is stable: loans sharing an issue_d keep their relative order,
    # which is the only reason this file can be matched by position at all
    df = df.sort_values('issue_d', kind='mergesort').reset_index(drop=True)
    assert df['issue_d'].is_monotonic_increasing, 'rows are not in issue_d order'

    # ---- 5. the three splits --------------------------------------------
    df['split'] = np.select(
        [df['issue_d'] < VALID_START, df['issue_d'] < TEST_START],
        ['train', 'valid'],
        default='test')

    counts = df['split'].value_counts()
    for name, expected in [('train', N_TRAIN), ('valid', N_VALID), ('test', N_TEST)]:
        got = int(counts.get(name, 0))
        assert got == expected, f'{name} has {got} rows, expected {expected}'
        bad = df.loc[df['split'] == name, 'target'].mean()
        say(f'  {name:<6} {expected:>7,}   bad rate {bad:.4f}')

    # ---- 6. realised cash, per loan -------------------------------------
    # total_pymnt ALREADY CONTAINS recoveries. Verified on all 26,280 rows:
    # total_pymnt == prncp + int + late + recoveries, exact but for one
    # rounding cent. Adding recoveries again would quietly understate every
    # default by whatever was clawed back - the first version of this file
    # did exactly that. The assert below is what caught it, so it stays.
    composed = (df['total_rec_prncp'] + df['total_rec_int']
                + df['total_rec_late_fee'] + df['recoveries'])
    off = (df['total_pymnt'] - composed).abs()
    assert (off > 0.02).sum() == 0, (
        f'{int((off > 0.02).sum())} rows where total_pymnt is not '
        'prncp + int + late + recoveries; the net cash formula assumes it is')

    # everything the lender got back, minus the fee charged on collections,
    # minus everything it put in
    df['net_cash'] = (df['total_pymnt'] - df['collection_recovery_fee']
                      - df['funded_amnt'])
    df['net_pct'] = df['net_cash'] / df['funded_amnt']

    # exposure at default: principal still outstanding when it went bad
    df['ead']     = df['funded_amnt'] - df['total_rec_prncp']
    df['ead_pct'] = df['ead'] / df['funded_amnt']

    # loss given default: the share of that exposure never recovered.
    # only meaningful where the loan actually defaulted and had exposure left
    bad_rows = df['target'].eq(1)
    exposed = bad_rows & df['ead'].gt(0)
    df['lgd'] = np.where(
        exposed,
        (df['ead'] - df['recoveries'] + df['collection_recovery_fee']) / df['ead'].where(df['ead'] != 0),
        np.nan)

    # interest collected before the loan went bad - the part of a default
    # that is not a loss. Late fees belong here: the lender kept them.
    df['int_pct'] = ((df['total_rec_int'] + df['total_rec_late_fee'])
                     / df['funded_amnt'])

    # how long the loan actually ran, which is not its term: these borrowers
    # prepay, so scheduled interest overstates what arrives
    df['life_m'] = _months_between(df['last_pymnt_d'], df['issue_d'])

    # the decomposition slice 8 prices with, checked rather than trusted:
    #     net_pct = int_pct - ead_pct * LGD
    # it is what makes LGD a knob the sensitivity analysis can turn
    identity = (df.loc[bad_rows, 'net_pct']
                - (df.loc[bad_rows, 'int_pct']
                   - df.loc[bad_rows, 'ead_pct'] * df.loc[bad_rows, 'lgd'])).abs()
    assert identity.max() < 1e-6, (
        f'loss decomposition is off by {identity.max():.6f}; '
        'net_pct = int_pct - ead_pct * LGD no longer holds')

    # ---- 7. sanity, reported not fixed ----------------------------------
    # a default that still made money is not a data fault: it ran most of its
    # term, paid most of its interest, then stopped. Those run 29 months on
    # average against 14.6 for the ones that lost money. The economics have to
    # survive that, which is exactly why loss is measured and not assumed.
    still_open = int((df['out_prncp'] > 0).sum())
    profitable = bad_rows & df['net_pct'].gt(0)
    say(f"\ncash not final (out_prncp > 0)      {still_open}")
    say(f"defaults that still made money     {int(profitable.sum())}"
        f"   (ran {df.loc[profitable, 'life_m'].mean():.1f} months vs "
        f"{df.loc[bad_rows & ~profitable, 'life_m'].mean():.1f})")
    say(f"net_pct above +1.0                 {int(df['net_pct'].gt(1).sum())}")
    say(f"life_m missing (no last_pymnt_d)   {int(df['life_m'].isna().sum())}")

    off_scale = int(((df['lgd'] < 0) | (df['lgd'] > 1)).sum())
    assert off_scale == 0, f'{off_scale} loans have an LGD outside [0, 1]'

    return df


def attach_predictions(money, proc_dir=PROC):
    """Add p_logreg and p_lgbm_tuned to the test rows, by position.

    Checked, not assumed: a reordering would show up later as a strange
    profit curve and send you hunting the wrong bug.
    """
    test = money[money['split'] == 'test'].reset_index(drop=True)
    published = pd.read_parquet(proc_dir / 'predictions_test_slice6.parquet')

    assert len(test) == len(published), (
        f'{len(test)} test rows vs {len(published)} published predictions')
    assert np.array_equal(test['issue_d'].to_numpy(),
                          published['issue_d'].to_numpy()), 'issue_d differs'
    assert np.array_equal(test['target'].to_numpy().astype(int),
                          published['target'].to_numpy().astype(int)), 'target differs'
    assert np.array_equal(test['zip_code'].to_numpy(),
                          published['zip_code'].to_numpy()), 'zip_code differs'

    test['p_logreg']     = published['p_logreg'].to_numpy()
    test['p_lgbm_tuned'] = published['p_lgbm_tuned'].to_numpy()
    return test


if __name__ == '__main__':
    money = build_money_table()
    out = PROC / 'money_table.parquet'
    money.to_parquet(out, index=False)
    print(f'\nwrote {out.relative_to(ROOT)}   {len(money):,} rows x {money.shape[1]} cols')

    test = attach_predictions(money)
    print(f'predictions attached to {len(test):,} test rows, row order verified')

    print('\n---- measured on TRAIN only, for slice 8 to decide on ----')
    tr   = money[money['split'] == 'train']
    good = tr[tr['target'] == 0]
    bad  = tr[tr['target'] == 1]
    print(f'good loans  n={len(good):,}   net {good["net_pct"].mean():+.4f} of funded'
          f'   ran {good["life_m"].mean():.1f} months')
    print(f'bad loans   n={len(bad):,}   net {bad["net_pct"].mean():+.4f} of funded'
          f'   ran {bad["life_m"].mean():.1f} months')
    print(f'EAD  {bad["ead_pct"].mean():.4f} of funded')
    print(f'LGD  {bad["lgd"].mean():.4f} of exposure')
