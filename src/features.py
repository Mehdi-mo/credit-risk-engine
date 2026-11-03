"""Feature engineering for the credit risk engine (slice 3).

Row-by-row steps compute each row from itself alone and are safe before
the split. Learned steps (medians, category levels) read a whole column
and must be fitted on train only.
"""

import numpy as np
import pandas as pd

from .columns import (CAT_ONE_HOT, CAT_FOLD_RARE, CAT_DROP,
                      DROP_TIME_DEPENDENT, DROP_DELINQ_FAMILY)

YEAR_MIN, YEAR_MAX   = 2013, 2017
CREDIT_HISTORY_CAP   = 600     # 50 years; 24 rows had 1950s credit lines
INQ_FILL             = 36      # blanks mean "no recent inquiry" = the safe end
LOG_CANDIDATES       = ['annual_inc', 'revol_bal', 'loan_amnt', 'tot_cur_bal']
SKEW_THRESHOLD       = 1.0


# ---- row-by-row: safe either side of the split --------------------------

def add_row_features(df):
    """Each row computed from itself alone. Nothing learned."""
    df = df.copy()

    # every "how long ago" is measured from issue_d, never from today
    df['credit_history_months'] = (
        (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12
        + (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
    ).clip(upper=CREDIT_HISTORY_CAP)

    # an amount only means something next to the capacity to carry it
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']

    safe_total = df['total_acc'].replace(0, np.nan)   # inf lies, NaN is honest
    df['open_to_total_acc'] = df['open_acc'] / safe_total

    # blank here sits on the same scale as "longest ago", so it is a feature
    df['mths_since_recent_inq_missing'] = df['mths_since_recent_inq'].isna().astype('int8')

    # two columns, one fact (always 4 apart)
    df['fico'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    return df


def add_logs(df, threshold=SKEW_THRESHOLD):
    """Squash long right tails so a linear model is not dragged by outliers."""
    df = df.copy()
    cols = [c for c in LOG_CANDIDATES if c in df.columns and df[c].skew() > threshold]
    return pd.concat([df, np.log1p(df[cols]).add_prefix('log_')], axis=1)


def drop_decided_columns(df):
    """Columns ruled out in slice 3, for reasons recorded in columns.py."""
    gone = (DROP_DELINQ_FAMILY + DROP_TIME_DEPENDENT + CAT_DROP
            + ['fico_range_low', 'fico_range_high'])
    return df.drop(columns=[c for c in gone if c in df.columns])


def fold_rare_levels(df):
    """Fold tiny category levels into an existing level - never into a new one."""
    df = df.copy()
    for col, (rare, into) in CAT_FOLD_RARE.items():
        df[col] = df[col].where(~df[col].isin(rare), into)
    return df


def restrict_years(df, year_min=YEAR_MIN, year_max=YEAR_MAX):
    years = df['issue_d'].dt.year
    return df[(years >= year_min) & (years <= year_max)].copy().reset_index(drop=True)


# ---- the split: everything below is fitted on train only ----------------

def split_by_time(df, quantile=0.8):
    """Split by date. Rows come back in time order, index reset.

    Masking keeps the right rows but not their order, and TimeSeriesSplit
    cuts by position - so the sort is not cosmetic.
    """
    # mergesort is stable: loans sharing an issue_d keep their relative order
    df = df.sort_values('issue_d', kind='mergesort').reset_index(drop=True)

    cut = df['issue_d'].quantile(quantile)
    train = df[df['issue_d'] <  cut].copy().reset_index(drop=True)
    test  = df[df['issue_d'] >= cut].copy().reset_index(drop=True)
    return train, test, cut


def fit_fill_values(train):
    """Medians learned from train. Test never gets its own."""
    num_cols = train.select_dtypes(include='number').columns.drop('target')
    return train[num_cols].median()


def apply_fills(frame, fill_values):
    frame = frame.copy()
    frame['mths_since_recent_inq'] = frame['mths_since_recent_inq'].fillna(INQ_FILL)
    cols = fill_values.index
    frame[cols] = frame[cols].fillna(fill_values)
    return frame


def encode_categories(train, test):
    """Level lists fixed by train, so both halves give identical columns."""
    train, test = train.copy(), test.copy()

    for col in CAT_ONE_HOT:
        cats = sorted(train[col].dropna().unique())
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col]  = pd.Categorical(test[col],  categories=cats)

    train = pd.get_dummies(train, columns=CAT_ONE_HOT, drop_first=True, dtype='int8')
    test  = pd.get_dummies(test,  columns=CAT_ONE_HOT, drop_first=True, dtype='int8')
    return train, test