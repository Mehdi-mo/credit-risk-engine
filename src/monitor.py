"""Is the model still looking at the world it was trained on? (slice 10)

A model does not break with an error. The applicants change, the lender
changes its rules, and the same code keeps returning confident numbers about
a population it never saw. This file is the instrument that would notice.

WHAT IT READS
The request log - one record per decision, as src/requestlog.py writes it.
To measure drift it rebuilds the model's features from each logged
application through src/decide.py, the same path the service used. There is
no second copy of the feature build here either.

WHAT IT MEASURES, IN THE ORDER IT BECOMES AVAILABLE

  The day a request arrives - no outcome needed
    1. Counters slice 9 built: the share of requests rejected, flagged as
       outside the training range (and which field), imputed, guarded, and
       what the declines are made of.
    2. PSI of every model feature against the training rows.
    3. PSI of the score, and the approval rate.
    4. WHY the average score moved. The model is linear, so the shift in the
       mean log-odds is exactly the sum over factors of (weight x shift in
       that factor). It is arithmetic (src/reasons.py), not an estimate.

  Months or years later - once loans have repaid or defaulted
    5. AUC and calibration per quarter.
    6. Realised margin and loss against what models/economics.json predicts.

PSI, IN ONE PARAGRAPH
Cut the training values of a feature into 10 equal-sized bins. Count what
share of the new window falls in each bin. PSI = sum over bins of
(new share - old share) x ln(new share / old share). It is 0 when the two
distributions match and grows as they pull apart. The usual reading is
below 0.10 stable, 0.10 to 0.25 worth a look, above 0.25 shifted.

THE REPLAY
There is no live traffic, so `python -m src.monitor` replays history: every
application from 2013 to 2017 is pushed through the service's own path with
its application date as the timestamp, exactly as if it had arrived then.
The model was fitted on applications before 2016-01, so the eight quarters
of 2016 and 2017 are what production would have looked like.

THE RULES IT KEEPS
- Every threshold is in models/monitor_policy.json, which was written before
  this file was run on anything.
- The reference is the training rows only. The noise floor is drawn from
  the training rows only.
- 2016-09 onwards is the test set. It is read here for reporting, for the
  seventh time. Nothing is fitted, chosen or tuned from it, and an alert
  raised here changes nothing in models/.

Run with:  python -m src.monitor            (the replay, tables and figures)
           python -m src.monitor LOGFILE    (a real request log)
"""

import json
import sys

import numpy as np
import pandas as pd

from .contract import ContractError, validate
from .decide import ROOT, Engine
from .reasons import by_factor, contributions, label_of
from .requestlog import make_record, read_log

POLICY_PATH = ROOT / 'models' / 'monitor_policy.json'

N_BINS = 10
DISCRETE_MAX = 10        # this many distinct values or fewer: one bin per value
FLOOR = 1e-4             # an empty bin would make ln() infinite
NOISE_DRAWS = 200
SEED = 42


def load_policy(path=POLICY_PATH):
    return json.loads(path.read_text(encoding='utf-8'))


# ---- PSI ---------------------------------------------------------------

def make_bins(reference):
    """How to cut one feature, decided by the reference values alone."""
    reference = np.asarray(reference, dtype='float64')
    values = np.unique(reference)
    if len(values) <= DISCRETE_MAX:
        return {'kind': 'discrete', 'values': values}
    # quantile edges; np.unique because a spiky feature repeats an edge
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return {'kind': 'quantile', 'edges': inner}


def shares(x, bins):
    """The share of x in each bin. A value no bin was made for gets its own."""
    x = np.asarray(x, dtype='float64')
    if bins['kind'] == 'discrete':
        counts = np.array([(x == v).sum() for v in bins['values']], dtype='float64')
        counts = np.append(counts, len(x) - counts.sum())      # never seen in reference
    else:
        counts = np.bincount(np.searchsorted(bins['edges'], x, side='right'),
                             minlength=len(bins['edges']) + 1).astype('float64')
    return counts / max(len(x), 1)


def psi_from_shares(ref, cur):
    ref, cur = np.maximum(ref, FLOOR), np.maximum(cur, FLOOR)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi(reference, current, bins=None):
    """Population stability index of `current` against `reference`."""
    if bins is None:
        bins = make_bins(reference)
    return psi_from_shares(shares(reference, bins), shares(current, bins))


class Reference:
    """The training rows, cut into bins once."""

    def __init__(self, frame, columns):
        self.columns = list(columns)
        self.frame = frame[self.columns].reset_index(drop=True)
        self.bins = {c: make_bins(self.frame[c]) for c in self.columns}
        self.shares = {c: shares(self.frame[c], self.bins[c]) for c in self.columns}
        self._noise = {}

    def psi(self, window):
        return pd.Series({c: psi_from_shares(self.shares[c], shares(window[c], self.bins[c]))
                          for c in self.columns})

    def noise_p99(self, n):
        """The PSI that sampling alone gives a window of n rows: the 99th
        percentile over random samples of n training rows. Rounded to the
        nearest 50 rows so nearby window sizes share one answer."""
        n = int(min(max(50, round(n / 50) * 50), len(self.frame)))
        if n not in self._noise:
            rng = np.random.default_rng(SEED + n)
            draws = [self.psi(self.frame.iloc[rng.choice(len(self.frame), n, replace=False)])
                     for _ in range(NOISE_DRAWS)]
            self._noise[n] = pd.DataFrame(draws).quantile(0.99)
        return self._noise[n]


def level(value, warn, alert, noise=0.0):
    """'ok', 'warn' or 'alert' - and never above 'ok' inside the noise."""
    if value <= noise:
        return 'ok'
    return 'alert' if value >= alert else 'warn' if value >= warn else 'ok'


# ---- from log records to one row per request ----------------------------

def frame_from_records(engine, records):
    """Log records in, one row per request out: what the service said, and
    the model's features rebuilt from the logged application."""
    rows, clean_rows, positions = [], [], []
    for i, r in enumerate(records):
        row = {'ts': pd.Timestamp(r['ts']), 'rejected': 'rejected' in r}
        if not row['rejected']:
            flags = r['flags']
            raised = r.get('raised_risk') or [None]
            row.update({
                'p': r['probability_of_default'],
                'approve': r['decision'] == 'approve',
                'review': r['review_recommended'],
                'guard': flags['guard_applied'],
                'imputed': bool(flags['imputed']),
                'folded': bool(flags['folded']),
                'outside': bool(flags['outside_training_range']),
                'outside_fields': [f['field'] for f in flags['outside_training_range']],
                'main_reason': raised[0] if r['decision'] == 'decline' else None,
            })
            try:
                clean_rows.append(validate(r['application'], engine.spec)[0])
                positions.append(i)
            except ContractError:
                # scored when it was logged, refused by today's contract: the
                # contract changed in between. Count it; do not guess features.
                row['contract_changed'] = True
        rows.append(row)

    frame = pd.DataFrame(rows)
    if clean_rows:
        feats = engine.features(clean_rows)[engine.bundle['columns']]
        feats.index = positions
        frame = frame.join(feats)
    return frame


def score_of(p):
    """Log-odds of the probability: the scale the model adds things up on."""
    p = np.clip(np.asarray(p, dtype='float64'), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# ---- the report for one window ------------------------------------------

def window_report(engine, reference, ref_frame, window, policy):
    """Everything that can be said about one window without an outcome."""
    cols = engine.bundle['columns']
    scored = window[~window['rejected']]
    n = len(scored)
    noise = reference.noise_p99(n)
    p_rule, s_rule = policy['psi'], policy['share_of_requests']

    drift = reference.psi(scored.assign(score=score_of(scored['p'])))
    weights = pd.Series(np.abs(engine.bundle['model'].coef_[0]), index=cols)

    out = {
        'n': len(window),
        'rejected': window['rejected'].mean(),
        'outside': scored['outside'].mean(),
        'imputed': scored['imputed'].mean(),
        'guard': scored['guard'].mean(),
        'review': scored['review'].mean(),
        'approval': scored['approve'].mean(),
        'mean_p': scored['p'].mean(),
        'psi_score': drift['score'],
        'psi_score_noise': noise['score'],
        'psi_max_feature': drift[cols].idxmax(),
        'psi_max': drift[cols].max(),
        # only features the model gives a weight to can move a decision
        'psi_max_weighted_feature': drift[cols][weights > 0].idxmax(),
        'psi_max_weighted': drift[cols][weights > 0].max(),
        'n_features_warn': int(sum(level(drift[c], p_rule['warn'], p_rule['alert'], noise[c]) != 'ok'
                                   for c in cols)),
        'n_features_alert': int(sum(level(drift[c], p_rule['warn'], p_rule['alert'], noise[c]) == 'alert'
                                    for c in cols)),
    }
    out['level_score'] = level(out['psi_score'], p_rule['warn'], p_rule['alert'], noise['score'])
    out['level_outside'] = level(out['outside'], **s_rule['outside_training_range'])
    out['level_rejected'] = level(out['rejected'], **s_rule['rejected'])

    # why the average score moved: exact for a linear model
    phi_ref, _ = contributions(engine.bundle, ref_frame, apply_guard=engine.apply_guard)
    phi_win, _ = contributions(engine.bundle, scored, apply_guard=engine.apply_guard)
    shift = by_factor(phi_win).mean() - by_factor(phi_ref).mean()
    out['score_shift'] = float(shift.sum())

    return out, drift, shift.sort_values(key=np.abs, ascending=False)


# ---- outcomes, when they exist ------------------------------------------

def auc_interval(y, p, draws=500, seed=SEED):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def outcome_report(y, p, policy):
    """AUC and calibration for one window whose outcomes are known."""
    y, p = np.asarray(y, dtype='float64'), np.asarray(p, dtype='float64')
    rule = policy['needs_outcomes']
    auc, lo, hi = auc_interval(y, p)
    gap = p.mean() - y.mean()
    # could the observed rate be the predicted one, with this many loans?
    se = np.sqrt(p.mean() * (1 - p.mean()) / len(y))
    outside = abs(gap) > 1.96 * se
    cal = rule['calibration_gap']
    return {
        'n': len(y), 'bad_rate': y.mean(), 'mean_p': p.mean(), 'gap': gap,
        'level_calibration': level(abs(gap), cal['warn'], cal['alert']) if outside else 'ok',
        'auc': auc, 'auc_lo': lo, 'auc_hi': hi,
        'level_auc': 'warn' if hi < rule['auc']['reference'] else 'ok',
        'brier': float(np.mean((p - y) ** 2)),
    }


# ---- a real log ----------------------------------------------------------

def report_log(path):
    """The no-outcome report for a request log written by the service."""
    engine, policy = Engine(), load_policy()
    frame = frame_from_records(engine, read_log(path))
    print(f'{len(frame)} requests from {frame["ts"].min()} to {frame["ts"].max()}')
    print(f'rejected {frame["rejected"].mean():.1%}')
    scored = frame[~frame['rejected']]
    if len(scored) == 0:
        return
    print(f'approved {scored["approve"].mean():.1%}   outside the training range '
          f'{scored["outside"].mean():.1%}   imputed {scored["imputed"].mean():.1%}   '
          f'guard {scored["guard"].mean():.1%}')
    if len(scored) < policy['min_rows']:
        print(f'{len(scored)} scored requests: fewer than the {policy["min_rows"]} the policy '
              f'asks for before a PSI means anything. Counters only.')
        return
    from .feature_spec import load_raw_rows
    from .features import transform
    train, _test, _cut = load_raw_rows()
    inner = train[train['issue_d'] < pd.Timestamp(engine.bundle['split_date'])]
    ref_frame = transform(inner, engine.spec)[engine.bundle['columns']].reset_index(drop=True)
    ref_frame['score'] = score_of(engine.probabilities(ref_frame))
    reference = Reference(ref_frame, engine.bundle['columns'] + ['score'])
    out, drift, shift = window_report(engine, reference, ref_frame, frame, policy)
    print(f"score PSI {out['psi_score']:.3f} ({out['level_score']})   "
          f"largest feature PSI {out['psi_max']:.3f} ({out['psi_max_feature']})")
    print('what moved the average score:')
    for factor, value in shift.head(5).items():
        print(f'  {label_of(factor):46s} {value:+.3f}')


# ---- the replay ----------------------------------------------------------

def replay():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .decide import to_application
    from .economics import PROC, price_loans
    from .feature_spec import load_raw_rows
    from .profit import ACCENT, LIGHT, MUTED, _save, _style

    BAD = '#A6402E'
    engine, policy = Engine(), load_policy()
    cols = engine.bundle['columns']
    split = pd.Timestamp(engine.bundle['split_date'])

    train, test, cut = load_raw_rows()
    raw = pd.concat([train, test], ignore_index=True)
    assert raw['issue_d'].is_monotonic_increasing, 'rows are not in issue_d order'
    assert len(raw) == 26_280

    # every application through the service's own path, stamped with its date
    applications = [to_application(r) for _, r in raw.iterrows()]
    results = engine.decide_many(applications, with_reasons=True)
    records = [make_record(a, r, ts) for a, r, ts in zip(applications, results, raw['issue_d'])]
    frame = frame_from_records(engine, records)
    assert not frame['rejected'].any(), 'a historical application was rejected'
    frame['quarter'] = frame['ts'].dt.to_period('Q').astype(str)
    frame['target'] = raw['target'].to_numpy()

    # the reference: the rows the weights were fitted on, and nothing later
    is_ref = frame['ts'] < split
    assert is_ref.sum() == 16_134
    ref_frame = frame.loc[is_ref, cols].reset_index(drop=True)
    ref_frame['score'] = score_of(frame.loc[is_ref, 'p'])
    reference = Reference(ref_frame, cols + ['score'])

    print(f'reference: {is_ref.sum():,} applications before {split.date()}')
    print(f'replayed:  {len(frame):,} applications, {frame["quarter"].nunique()} quarters\n')

    # ---- 1. no outcome needed --------------------------------------------
    rows, drifts, shifts = [], {}, {}
    for q, window in frame.groupby('quarter'):
        out, drift, shift = window_report(engine, reference, ref_frame, window, policy)
        out.update(quarter=q, in_training=bool((window['ts'] < split).all()))
        rows.append(out)
        drifts[q], shifts[q] = drift, shift
    table = pd.DataFrame(rows).set_index('quarter')
    live = table[~table['in_training']]

    print('---- 1. what a monitor sees the day the requests arrive ----')
    print('  quarter      n  approve  outside  imputed  score PSI (noise)  level   '
          'features warn/alert   mean score shift')
    for q, r in table.iterrows():
        mark = ' ' if r['in_training'] else '>'
        print(f" {mark}{q}  {r['n']:>5,}   {r['approval']:6.1%}  {r['outside']:6.1%}   "
              f"{r['imputed']:6.1%}   {r['psi_score']:6.3f}  ({r['psi_score_noise']:.3f})   "
              f"{r['level_score']:<5}   {r['n_features_warn']:>2} / {r['n_features_alert']:<2}"
              f"             {r['score_shift']:+.3f}")
    print('  > = after the training window. Levels are never raised inside it.')

    last = live.index[-1]
    print(f'\n---- 2. the features that drifted most by {last} (PSI, weight in the model) ----')
    weights = pd.Series(engine.bundle['model'].coef_[0], index=cols)
    noise = reference.noise_p99(live.loc[last, 'n'])
    for c in drifts[last][cols].sort_values(ascending=False).head(8).index:
        print(f'  {c:28s} PSI {drifts[last][c]:.3f}   noise {noise[c]:.3f}   '
              f'weight {weights[c]:+.3f}{"   (the model ignores it)" if weights[c] == 0 else ""}')

    print(f'\n---- 3. why the average score moved by {last} (log-odds, exact) ----')
    print(f"  total shift {live.loc[last, 'score_shift']:+.3f}")
    for factor, value in shifts[last].head(6).items():
        print(f'  {label_of(factor):46s} {value:+.3f}')

    print('\n---- 4. outside the training range, by field and quarter ----')
    exploded = frame.loc[~is_ref, ['quarter', 'outside_fields']].explode('outside_fields').dropna()
    by_field = (exploded.groupby(['outside_fields', 'quarter']).size().unstack(fill_value=0)
                .reindex(columns=live.index, fill_value=0))
    by_field = by_field.loc[by_field.sum(axis=1).sort_values(ascending=False).index]
    print(by_field.head(5).to_string())

    # ---- 2. outcomes, known only much later ------------------------------
    print('\n---- 5. once outcomes are known: AUC and calibration by quarter ----')
    out_rows = []
    for q, window in frame[~is_ref].groupby('quarter'):
        r = outcome_report(window['target'], window['p'], policy)
        r['quarter'] = q
        out_rows.append(r)
    outcomes = pd.DataFrame(out_rows).set_index('quarter')
    print('  quarter      n   predicted  observed    gap   level    AUC   95% interval     level')
    for q, r in outcomes.iterrows():
        print(f"  {q}  {r['n']:>5,}     {r['mean_p']:6.1%}    {r['bad_rate']:6.1%}  "
              f"{r['gap']:+6.1%}   {r['level_calibration']:<5}  {r['auc']:.3f}  "
              f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]   {r['level_auc']}")
    print('  The calibrator was fitted on 2016-01 to 2016-08, so the first three quarters\n'
          '  are partly its own training data.')

    # ---- 2b. an alerted feature that is a yes/no flag gets split out ------
    alerted = [c for c in cols
               if weights[c] != 0 and reference.bins[c]['kind'] == 'discrete'
               and level(drifts[last][c], policy['psi']['warn'], policy['psi']['alert'],
                         noise[c]) == 'alert']
    segments = []
    for c in alerted:
        print(f'\n---- 5b. {c}: the feature that raised an alert, split by value ----')
        print(f'  in the training rows: {int((ref_frame[c] != 0).sum())} of {len(ref_frame):,} '
              f'applications had it set')
        for value, g in frame[~is_ref].groupby(c):
            se = np.sqrt(g['target'].mean() * (1 - g['target'].mean()) / len(g))
            print(f"  {c} = {value:g}   {len(g):>5,} live applications   predicted "
                  f"{g['p'].mean():6.1%}   observed {g['target'].mean():6.1%} (+/- {1.96 * se:.1%})"
                  f"   declined {1 - g['approve'].mean():6.1%}")
            segments.append({'feature': c, 'value': value, 'n': len(g),
                             'predicted': g['p'].mean(), 'observed': g['target'].mean(),
                             'declined': 1 - g['approve'].mean()})
    pd.DataFrame(segments).to_csv(PROC / 'slice10_alerted_segments.csv', index=False)

    # ---- 3. the money ----------------------------------------------------
    money = pd.read_parquet(PROC / 'money_table.parquet')
    assert np.array_equal(money['target'].to_numpy(), raw['target'].to_numpy()) and \
        np.array_equal(money['issue_d'].to_numpy(), raw['issue_d'].to_numpy()), \
        'money table and the replay are not in the same row order'
    priced = price_loans(money, engine.economics)
    money = money.assign(quarter=frame['quarter'].to_numpy(),
                         margin_pred=priced['margin_pct'].to_numpy(),
                         loss_pred=priced['loss_pct'].to_numpy())
    good, bad = money[money['target'] == 0], money[money['target'] == 1]
    margin = pd.DataFrame({
        'repaid': good.groupby('quarter').size(),
        'margin_predicted': good.groupby('quarter')['margin_pred'].mean(),
        'margin_realised': good.groupby('quarter')['net_pct'].mean(),
        'months_to_repay': good.groupby('quarter')['life_m'].mean(),
        'loss_predicted': bad.groupby('quarter')['loss_pred'].mean(),
        'loss_realised': -bad.groupby('quarter')['net_pct'].mean(),
    })
    margin['margin_ratio'] = margin['margin_realised'] / margin['margin_predicted']
    print('\n---- 6. the margin, the weakest number in the project, by quarter ----')
    print('  quarter   repaid   margin predicted  realised  ratio   months to repay   '
          'loss predicted  realised')
    for q, r in margin.iterrows():
        mark = ' ' if q < '2016' else '>'
        print(f" {mark}{q}  {int(r['repaid']):>6,}        {r['margin_predicted']:6.1%}     "
              f"{r['margin_realised']:6.1%}   {r['margin_ratio']:4.2f}        "
              f"{r['months_to_repay']:5.1f}            {r['loss_predicted']:6.1%}    "
              f"{r['loss_realised']:6.1%}")
    print('  A repaid loan in a late quarter is one that repaid FAST - it had to, to be in\n'
          '  this snapshot at all. So the falling ratio is censoring first, drift second.')

    # ---- 4. how late the truth arrives -----------------------------------
    # Fully paid: known the month of the last payment. A default: Lending Club
    # charges off about five months after the last payment, so add five.
    known = money['last_pymnt_d'].fillna(money['issue_d']) + \
        pd.to_timedelta(np.where(money['target'] == 1, 5 * 30.44, 0), unit='D')
    money = money.assign(known=known, p=frame['p'].to_numpy())
    print('\n---- 7. how late the truth arrives: what a monitor would have seen ----')
    print('  months after the quarter ended   outcomes known   bad rate seen   (final)')
    mat_rows = []
    for q in live.index:
        w = money[money['quarter'] == q]
        end = pd.Period(q, freq='Q').end_time.normalize()
        for months in (6, 12, 18, 24):
            seen = w[w['known'] <= end + pd.DateOffset(months=months)]
            mat_rows.append({'quarter': q, 'months_after': months,
                             'known_share': len(seen) / len(w),
                             'bad_rate_seen': seen['target'].mean() if len(seen) else np.nan,
                             'bad_rate_final': w['target'].mean()})
    maturity = pd.DataFrame(mat_rows)
    for months, g in maturity.groupby('months_after'):
        print(f"  {months:>2}                                    {g['known_share'].mean():6.1%}"
              f"          {g['bad_rate_seen'].mean():6.1%}      ({g['bad_rate_final'].mean():.1%})")
    print('  Averages over the eight live quarters. "Final" is itself only the loans that had\n'
          '  resolved by the snapshot; loans still running are not in the data at all.')

    # ---- 5. what the policy would have done ------------------------------
    alert_score = (live['level_score'] == 'alert')
    alert_cal = (outcomes['level_calibration'] == 'alert').reindex(live.index, fill_value=False)
    either = (alert_score | alert_cal).to_numpy()
    review = [bool(either[i] and either[i - 1]) for i in range(1, len(either))]
    print('\n---- 8. the declared policy, applied ----')
    print(f"  score PSI:      warn in {(live['level_score'] == 'warn').sum()} of {len(live)} live "
          f"quarters, alert in {alert_score.sum()}")
    print(f"  outside range:  warn in {(live['level_outside'] == 'warn').sum()}, "
          f"alert in {(live['level_outside'] == 'alert').sum()}")
    print(f"  calibration:    warn in {(outcomes['level_calibration'] == 'warn').sum()}, "
          f"alert in {alert_cal.sum()}   (needs outcomes)")
    print(f"  AUC:            warn in {(outcomes['level_auc'] == 'warn').sum()}   (needs outcomes)")
    print(f"  retrain review: {'triggered in ' + str(live.index[1:][review][0]) if any(review) else 'never triggered'}")

    # ---- write -----------------------------------------------------------
    table.to_csv(PROC / 'slice10_drift_by_quarter.csv')
    outcomes.to_csv(PROC / 'slice10_outcomes_by_quarter.csv')
    margin.to_csv(PROC / 'slice10_margin_by_quarter.csv')
    maturity.to_csv(PROC / 'slice10_label_maturity.csv', index=False)
    pd.DataFrame(drifts).T.to_csv(PROC / 'slice10_psi_all_features.csv')
    pd.DataFrame(shifts).T.to_csv(PROC / 'slice10_score_shift_by_factor.csv')

    # ---- figure 1: drift over time ---------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    x = np.arange(len(table))
    first_live = int((~table['in_training']).to_numpy().argmax())

    ax = axes[0]
    ax.plot(x, table['psi_score'], color=ACCENT, lw=2, marker='o', ms=4, label='PSI of the score')
    ax.plot(x, table['psi_max_weighted'], color=BAD, lw=1.6, marker='o', ms=3,
            label='largest PSI of a feature the model uses')
    ax.plot(x, table['psi_score_noise'], color=MUTED, lw=1, ls=':', label='what sampling alone gives')
    for y, name in ((policy['psi']['warn'], 'warn'), (policy['psi']['alert'], 'alert')):
        ax.axhline(y, color=MUTED, lw=0.8, ls='--')
        ax.text(0.1, y + 0.006, name, color=MUTED, fontsize=8.5)
    ax.axvspan(-0.5, first_live - 0.5, color=LIGHT, alpha=0.35, lw=0)
    ax.text(0.1, ax.get_ylim()[1] * 0.9, 'training window', color=MUTED, fontsize=9)
    _style(ax, ylabel='PSI against the training rows')
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper center')

    ax = axes[1]
    ax.plot(x, table['approval'] * 100, color=ACCENT, lw=2, marker='o', ms=4, label='approved, %')
    ax.plot(x, table['outside'] * 100, color=BAD, lw=1.6, marker='o', ms=3,
            label='a value outside the training range, %')
    ax.axvspan(-0.5, first_live - 0.5, color=LIGHT, alpha=0.35, lw=0)
    ax.set_xticks(x, table.index, rotation=60, ha='right', fontsize=8.5)
    _style(ax, ylabel='% of applications')
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='center left')
    fig_drift = _save(fig, 'slice10_drift.png',
                      'What a monitor sees without waiting for a single outcome')

    # ---- figure 2: outcomes ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xo = np.arange(len(outcomes))
    ax = axes[0]
    ax.plot(xo, outcomes['mean_p'] * 100, color=ACCENT, lw=2, marker='o', ms=4, label='predicted')
    ax.plot(xo, outcomes['bad_rate'] * 100, color=BAD, lw=2, marker='o', ms=4, label='observed')
    ax.set_xticks(xo, outcomes.index, rotation=60, ha='right', fontsize=8.5)
    _style(ax, ylabel='bad rate, %')
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax.set_title('Calibration', fontsize=10.5, color=MUTED, loc='left')

    ax = axes[1]
    ax.errorbar(xo, outcomes['auc'], yerr=[outcomes['auc'] - outcomes['auc_lo'],
                                           outcomes['auc_hi'] - outcomes['auc']],
                color=ACCENT, lw=1.6, marker='o', ms=4, capsize=3)
    ax.axhline(policy['needs_outcomes']['auc']['reference'], color=MUTED, lw=0.8, ls='--')
    ax.text(0.02, policy['needs_outcomes']['auc']['reference'] + 0.004,
            'cross-validated on training', color=MUTED, fontsize=8.5)
    ax.set_xticks(xo, outcomes.index, rotation=60, ha='right', fontsize=8.5)
    _style(ax, ylabel='AUC, with 95% interval')
    ax.set_title('Ranking', fontsize=10.5, color=MUTED, loc='left')
    fig_out = _save(fig, 'slice10_outcomes.png',
                    'What only shows up once the loans have run their course')

    print(f'\nwrote seven tables to data/processed/slice10_*.csv')
    print(f'wrote {fig_drift.relative_to(ROOT)} and {fig_out.relative_to(ROOT)}')
    return dict(table=table, outcomes=outcomes, margin=margin, maturity=maturity,
                drifts=drifts, shifts=shifts, frame=frame)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        report_log(sys.argv[1])
    else:
        replay()
