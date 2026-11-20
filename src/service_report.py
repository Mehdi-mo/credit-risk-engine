"""What the service does with the test book (slice 9). Reporting only.

Every one of the 5,378 test applications goes through the same path a caller
uses - contract, feature build, chain, reasons - and this file counts what
came out: how many were flagged and for what, and what the declines were made
of. Nothing is fitted, chosen or tuned from what is counted here.

Run with:  python -m src.service_report      (after python -m src.score_test)
"""

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .decide import Engine, to_application
from .economics import PROC, ROOT
from .feature_spec import load_raw_rows
from .profit import ACCENT, LIGHT, MUTED, _save, _style
from .reasons import label_of

OUT_PATH = PROC / 'slice09_service_on_test.csv'
BAD = '#A6402E'


if __name__ == '__main__':
    engine = Engine()
    _train, test, _cut = load_raw_rows()
    assert test['issue_d'].is_monotonic_increasing, 'test rows are not in issue_d order'
    assert len(test) == 5_378

    results = engine.decide_many([to_application(r) for _, r in test.iterrows()],
                                 with_reasons=True)
    assert not any('rejected' in r for r in results), 'a real application was rejected'

    y = test['target'].to_numpy()
    approve = np.array([r['decision'] == 'approve' for r in results])
    review = np.array([r['review_recommended'] for r in results])
    n = len(results)

    print(f'threshold {engine.threshold}   {engine.versions["model_version"]}')
    print(f'\n---- decisions on {n:,} test applications ----')
    print(f'  approved {approve.sum():,} ({approve.mean():.1%}), bad rate among them '
          f'{y[approve].mean():.2%}')
    print(f'  declined {(~approve).sum():,} ({(~approve).mean():.1%}), bad rate among them '
          f'{y[~approve].mean():.2%}')

    # ---- flags -----------------------------------------------------------
    flags = {
        'rejected by the contract': 0,
        'a blank was imputed': sum(bool(r['flags']['imputed']) for r in results),
        'a rare category was folded': sum(bool(r['flags']['folded']) for r in results),
        'credit history capped': sum(r['flags']['credit_history_capped'] for r in results),
        'a value outside the training range': sum(bool(r['flags']['outside_training_range'])
                                                  for r in results),
        'loan_to_income guard applied': sum(r['flags']['guard_applied'] for r in results),
        'review recommended': int(review.sum()),
    }
    print('\n---- what the contract did ----')
    for name, count in flags.items():
        print(f'  {name:36s} {count:>5,}   {count / n:6.2%}')

    outside = Counter(f['field'] for r in results for f in r['flags']['outside_training_range'])
    imputed = Counter(f for r in results for f in r['flags']['imputed'])
    print('\n  outside the training range, by field:')
    for field, count in outside.most_common(6):
        lo, hi = engine.spec['train_range'][field]
        print(f'    {field:24s} {count:>4}   training saw {lo:g} to {hi:g}, '
              f'test goes {test[field].min():g} to {test[field].max():g}')
    print('  imputed, by field: ' + ', '.join(f'{f} {c}' for f, c in imputed.most_common()))

    # do the flagged applications behave differently? reported, not acted on
    print(f'\n  review recommended: {review.sum()} applications, approval '
          f'{approve[review].mean():.1%}, actual bad rate {y[review].mean():.1%}')
    print(f'  everything else   : {(~review).sum():,} applications, approval '
          f'{approve[~review].mean():.1%}, actual bad rate {y[~review].mean():.1%}')

    # ---- what the declines are made of -----------------------------------
    declined = [r for r, a in zip(results, approve) if not a]
    first = Counter(r['reasons']['raised_risk'][0]['factor'] for r in declined)
    anywhere = Counter(x['factor'] for r in declined for x in r['reasons']['raised_risk'])

    print(f'\n---- the {len(declined)} declines: which factor is the main reason ----')
    for factor, count in first.most_common(8):
        print(f'  {label_of(factor):46s} main reason {count:>4} ({count / len(declined):5.1%})   '
              f'in the top 4: {anywhere[factor] / len(declined):5.1%}')

    rows = ([{'what': k, 'count': v, 'share': v / n} for k, v in flags.items()]
            + [{'what': f'outside training range: {f}', 'count': c, 'share': c / n}
               for f, c in outside.most_common()]
            + [{'what': f'main decline reason: {f}', 'count': c, 'share': c / len(declined)}
               for f, c in first.most_common()])
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    # ---- figure ----------------------------------------------------------
    top = anywhere.most_common(8)[::-1]
    labels = [label_of(f) for f, _ in top]
    in_top4 = np.array([c for _, c in top]) / len(declined) * 100
    main = np.array([first.get(f, 0) for f, _ in top]) / len(declined) * 100

    fig, ax = plt.subplots(figsize=(10, 5.2))
    pos = np.arange(len(top))
    ax.barh(pos, in_top4, color=LIGHT, height=0.62, label='among the four reasons returned')
    ax.barh(pos, main, color=BAD, height=0.62, label='the main reason')
    ax.set_yticks(pos, labels)
    for p, a, b in zip(pos, in_top4, main):
        ax.text(a + 1, p, f'{a:.0f}%', va='center', color=MUTED, fontsize=9)
    _style(ax, xlabel=f'share of the {len(declined)} declined test applications')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 105)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower right')
    path = _save(fig, 'slice09_decline_reasons.png',
                 'Two factors explain most declines: the term and the size of the loan')

    print(f'\nwrote {OUT_PATH.relative_to(ROOT)}')
    print(f'wrote {path.relative_to(ROOT)}')
