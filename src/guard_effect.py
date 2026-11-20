"""What switching the loan_to_income guard on actually changed (slice 9).

Slice 7 measured the guard on the probabilities: 2 of 5,378 rows move. This
file measures it on everything downstream of the probabilities, because a
changed probability only matters if it changes a metric or a decision.

It runs src/profit.py twice - once on slice 6's unguarded probabilities, once
on slice 9's guarded ones - with writing switched off, and compares the two
results number by number. Nothing is chosen here. The threshold that comes
out of the guarded run is reported next to the one slice 8 published, and
if they differed that would be a finding to write down, not a setting to tune.

Run with:  python -m src.guard_effect      (after python -m src.score_test)
"""

import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

from .economics import PROC, ROOT
from .profit import PREDICTIONS, PREDICTIONS_SLICE6, main as profit_main

OUT_PATH = PROC / 'slice09_guard_effect.csv'

# the three rules slice 8 compared. The guard matters only if one of them
# decides a loan differently.
SLICE5_THRESHOLD = 0.20


def _quiet(predictions):
    """profit.main prints a full report. Here only its numbers are wanted."""
    with contextlib.redirect_stdout(io.StringIO()):
        return profit_main(predictions, write=False)


def _metrics(y, p):
    return {'auc': roc_auc_score(y, p),
            'average_precision': average_precision_score(y, p),
            'brier': brier_score_loss(y, p),
            'log_loss': log_loss(y, p),
            'max_probability': float(np.max(p))}


if __name__ == '__main__':
    preds = pd.read_parquet(PREDICTIONS)
    y = preds['target'].to_numpy()
    off, on = preds['p_logreg_unguarded'].to_numpy(), preds['p_logreg'].to_numpy()

    before, after = _quiet(PREDICTIONS_SLICE6), _quiet(PREDICTIONS)

    # the two runs must be the same loans in the same order, or nothing
    # below means anything
    assert np.array_equal(before['priced']['target'].to_numpy(),
                          after['priced']['target'].to_numpy())
    assert np.array_equal(before['priced']['p_logreg'].to_numpy(), off) or \
        np.abs(before['priced']['p_logreg'].to_numpy() - off).max() < 1e-12, \
        "slice 9's unguarded column is not slice 6's p_logreg"

    rows = []

    # ---- 1. the probabilities as probabilities ---------------------------
    print('---- 1. the metrics slice 5 published, guard off -> guard on ----')
    m_off, m_on = _metrics(y, off), _metrics(y, on)
    for name in m_off:
        print(f'  {name:18s} {m_off[name]:.6f} -> {m_on[name]:.6f}   '
              f'({m_on[name] - m_off[name]:+.6f})')
        rows.append({'what': name, 'guard_off': m_off[name], 'guard_on': m_on[name]})

    moved = np.flatnonzero(off != on)
    ll_off = -(y * np.log(np.clip(off, 1e-15, 1 - 1e-15))
               + (1 - y) * np.log(np.clip(1 - off, 1e-15, 1 - 1e-15)))
    share = ll_off[moved].sum() / ll_off.sum()
    print(f'  the {len(moved)} moved rows carried {share:.2%} of the whole test log loss')

    # ---- 2. the threshold ------------------------------------------------
    print('\n---- 2. the threshold curve A chooses ----')
    print(f"  guard off {before['chosen_t']:.2f}    guard on {after['chosen_t']:.2f}")
    rows.append({'what': 'threshold chosen on curve A',
                 'guard_off': before['chosen_t'], 'guard_on': after['chosen_t']})

    # ---- 3. the decisions ------------------------------------------------
    print('\n---- 3. does any loan get a different decision? ----')
    p_star = after['priced']['p_star'].to_numpy()
    rules = {
        f"slice 5's {SLICE5_THRESHOLD:.2f}": lambda p: p < SLICE5_THRESHOLD,
        f"global {after['chosen_t']:.2f}": lambda p: p < after['chosen_t'],
        'per-loan break-even': lambda p: p < p_star,
    }
    for name, rule in rules.items():
        changed = int((rule(off) != rule(on)).sum())
        print(f'  {name:22s} {changed} of {len(y):,} decisions change')
        rows.append({'what': f'decisions changed under {name}', 'guard_off': 0, 'guard_on': changed})

    for i in moved:
        print(f'  row {i}: p {off[i]:.6f} -> {on[i]:.6f}, break-even for this loan '
              f'{p_star[i]:.3f}, highest break-even in the book {p_star.max():.3f}')

    # ---- 4. the profit tables and curves ---------------------------------
    print('\n---- 4. slice 8\'s numbers ----')
    t_before, t_after = before['table'], after['table']
    same_rules = list(t_before.index) == list(t_after.index)
    table_diff = (np.abs(t_after.to_numpy(dtype='float64') - t_before.to_numpy(dtype='float64')).max()
                  if same_rules else float('nan'))
    delta = (t_after.astype('float64') - t_before.astype('float64')).abs()
    n_cells = int((delta.to_numpy() > 1e-9).sum())
    where = delta.stack().idxmax()
    print(f'  approval-rule table: {n_cells} of {delta.size} cells change. The largest is '
          f'{table_diff:,.0f} in "{where[0]}" / {where[1]}')
    rows.append({'what': 'largest change in the approval-rule table', 'guard_off': 0, 'guard_on': table_diff})

    c_before, c_after = before['curve'], after['curve']
    cols = [c for c in c_before.columns if c != 'threshold']
    changed_t = c_before.loc[
        (np.abs(c_after[cols].to_numpy(dtype='float64')
                - c_before[cols].to_numpy(dtype='float64')) > 1e-9).any(axis=1), 'threshold']
    if len(changed_t):
        print(f'  profit curve: changes only at thresholds {changed_t.min():.2f} to '
              f'{changed_t.max():.2f} ({len(changed_t)} of {len(c_before)} points)')
    else:
        print('  profit curve: no point changes')
    rows.append({'what': 'lowest threshold where the profit curve changes',
                 'guard_off': float('nan'),
                 'guard_on': float(changed_t.min()) if len(changed_t) else float('nan')})

    s_diff = np.abs(after['surface']['best_threshold'].to_numpy()
                    - before['surface']['best_threshold'].to_numpy()).max()
    f_diff = np.abs(after['frontier']['gap'].to_numpy() - before['frontier']['gap'].to_numpy())
    print(f'  sensitivity surface: largest change in a best threshold {s_diff:.6g}')
    print(f'  fairness frontier:   largest change in the thin-file gap '
          f'{np.nanmax(f_diff) * 100:.4f} points, and none below threshold '
          f"{after['frontier'].loc[f_diff > 1e-12, 'threshold'].min():.2f}"
          if (f_diff > 1e-12).any() else
          '  fairness frontier:   no point changes')
    rows.append({'what': 'largest change in a sensitivity-surface threshold', 'guard_off': 0, 'guard_on': s_diff})
    rows.append({'what': 'largest change in the thin-file gap (points)', 'guard_off': 0,
                 'guard_on': float(np.nanmax(f_diff) * 100)})

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f'\nwrote {OUT_PATH.relative_to(ROOT)}')
