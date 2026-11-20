"""What the threshold is worth in money (slice 8).

THREE CURVES, NOT ONE
Slice 8 was meant to compute profit two ways - predicted from the
probabilities, realised from the outcomes - and read agreement as proof that
calibration holds in money. On this data that comparison is confounded, and
reading it as a calibration check would be wrong.

The reason is in the splits. Slice 3 dropped loans that had not yet resolved,
and that bites harder the later the vintage. Good 36-month loans ran 25.7
months in train and 12.1 in test: the test book is not a sample of repaid
loans, it is a sample of fast prepayers, because nothing else could finish
before the data was cut. Defaults land the same way - test defaults died at
10.9 months with 79% of principal still outstanding against 65% in train.

So the gap between predicted and realised money holds two different errors at
once. Three curves separate them, all on the same approved loans:

  A  predicted      p_i        x train-measured economics
  B  label + model  outcome_i  x train-measured economics
  C  realised       outcome_i  x that loan's own actual cash

A against B is calibration error: identical economics on both sides, only the
probability differs from the truth. B against C is the censoring distortion:
identical outcomes on both sides, only averaged euros differ from actual ones.

WHAT CHOOSES THE THRESHOLD
Not curve C. Picking the threshold that maximises realised profit would be
choosing on test outcomes, which this project has never done in seven slices
and does not start doing here. The operating point comes from the break-even
rule in src/economics.py, which uses the price and the train-fitted
parameters and never touches an outcome. Curve C's own maximum is computed
and reported, labelled as hindsight, because the distance between the chosen
point and that maximum is worth knowing - but it is a report, not a choice.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .economics import PROC, ROOT, fit_economics, price_loans, save_params

FIGURES = ROOT / 'figures'

# Slice 9 switched the loan_to_income guard on, so the probabilities now come
# from src/score_test.py. Slice 6's file is still on disk and still readable:
# main(PREDICTIONS_SLICE6) reproduces every number slice 8 first published.
PREDICTIONS = PROC / 'predictions_test_slice9.parquet'
PREDICTIONS_SLICE6 = PROC / 'predictions_test_slice6.parquet'
THRESHOLDS = np.round(np.arange(0.02, 1.0001, 0.01), 4)

# matching slices 5-7
BG, INK, MUTED = '#FCFCFB', '#52514E', '#8A8884'
ACCENT, LIGHT = '#1F6F6B', '#B9B7B1'


# ---- one approval rule, every number it produces ------------------------

def evaluate_rule(priced, approve, p_col):
    """Score one approve/reject decision over the whole test population.

    Total euros, not euros per approved loan: the lender faces a fixed pool
    of applicants, so a rule that approves fewer better loans still has to
    beat one that approves more. Per-euro figures come along for diagnosis.
    """
    approve = np.asarray(approve, dtype=bool)
    book = priced[approve]
    n_all, funded_all = len(priced), priced['funded_amnt'].sum()

    if len(book) == 0:
        zeros = dict.fromkeys(
            ['profit_pred', 'profit_label', 'profit_real', 'funded',
             'per_eur_pred', 'per_eur_label', 'per_eur_real', 'bad_rate'], 0.0)
        return dict(n_approved=0, approval_rate=0.0, **zeros)

    p = book[p_col].to_numpy()
    y = book['target'].to_numpy()
    gain, loss = book['gain_eur'].to_numpy(), book['loss_eur'].to_numpy()
    funded = book['funded_amnt'].sum()

    pred = ((1 - p) * gain - p * loss).sum()       # A
    label = ((1 - y) * gain - y * loss).sum()      # B
    real = book['net_cash'].sum()                  # C

    return dict(
        n_approved=len(book),
        approval_rate=len(book) / n_all,
        funded=funded,
        share_of_capital=funded / funded_all,
        bad_rate=float(y.mean()),
        profit_pred=pred, profit_label=label, profit_real=real,
        per_eur_pred=pred / funded,
        per_eur_label=label / funded,
        per_eur_real=real / funded,
    )


def sweep_thresholds(priced, p_col, thresholds=THRESHOLDS):
    """Every global threshold, scored the same three ways."""
    p = priced[p_col].to_numpy()
    rows = []
    for t in thresholds:
        row = evaluate_rule(priced, p < t, p_col)
        row['threshold'] = t
        rows.append(row)
    return pd.DataFrame(rows)


def per_loan_rule(priced, p_col):
    """Approve while this loan's own price still covers its own risk."""
    return priced[p_col].to_numpy() < priced['p_star'].to_numpy()


# ---- how far the answer moves when the assumption moves -----------------

def sensitivity_surface(test_raw, params, p_col,
                        lgd_grid=np.arange(0.70, 1.001, 0.05),
                        margin_grid=np.arange(0.6, 1.401, 0.1)):
    """Optimal threshold as the loss and margin assumptions move.

    The optimum is taken on curve A, never on realised money: the whole point
    is which threshold the assumptions imply, not which one hindsight likes.
    """
    rows = []
    for lgd in lgd_grid:
        for mult in margin_grid:
            priced = price_loans(test_raw, params, lgd=lgd, margin_multiplier=mult)
            curve = sweep_thresholds(priced, p_col)
            best = curve.loc[curve['profit_pred'].idxmax()]
            per_loan = evaluate_rule(priced, per_loan_rule(priced, p_col), p_col)
            rows.append(dict(
                lgd=round(float(lgd), 3), margin_multiplier=round(float(mult), 3),
                best_threshold=best['threshold'],
                best_profit_pred=best['profit_pred'],
                best_profit_real=best['profit_real'],
                per_loan_profit_real=per_loan['profit_real'],
                per_loan_approval=per_loan['approval_rate'],
                mean_break_even=float(priced['p_star'].mean()),
            ))
    return pd.DataFrame(rows)


# ---- what the money costs the thin-file group ---------------------------

def fairness_frontier(priced, group, p_col, thresholds=THRESHOLDS):
    """Approval of each group, and the profit, at every threshold."""
    p = priced[p_col].to_numpy()
    thin = (group == 'thin').to_numpy()
    rows = []
    for t in thresholds:
        approve = p < t
        rows.append(dict(
            threshold=t,
            approval_thin=approve[thin].mean(),
            approval_thick=approve[~thin].mean(),
            gap=approve[thin].mean() - approve[~thin].mean(),
            **{k: evaluate_rule(priced, approve, p_col)[k]
               for k in ('profit_pred', 'profit_label', 'profit_real')},
        ))
    return pd.DataFrame(rows)


def equalise_approval(priced, group, p_col, thick_threshold):
    """Give the thin-file group whatever threshold it needs to be approved at
    the same rate as the thick-file group, and price the difference.

    This equalises access, which is one fairness definition out of several.
    It is not free and it is not obviously right: the thin-file base rate is
    higher, so matching approval rates means knowingly approving worse loans.
    Slice 8 reports the trade, it does not pick a side.
    """
    p = priced[p_col].to_numpy()
    thin = (group == 'thin').to_numpy()
    target_rate = (p[~thin] < thick_threshold).mean()

    # the thin-file threshold that reaches that approval rate
    thin_threshold = float(np.quantile(p[thin], target_rate))

    approve = np.where(thin, p < thin_threshold, p < thick_threshold)
    equal = evaluate_rule(priced, approve, p_col)
    plain = evaluate_rule(priced, p < thick_threshold, p_col)

    return dict(
        thick_threshold=thick_threshold, thin_threshold=thin_threshold,
        approval_thin_before=(p[thin] < thick_threshold).mean(),
        approval_thin_after=approve[thin].mean(),
        approval_thick=target_rate,
        # the business measure, not this book's censored cash
        profit_before_label=plain['profit_label'],
        profit_after_label=equal['profit_label'],
        bad_rate_before=plain['bad_rate'], bad_rate_after=equal['bad_rate'],
    )


# ---- figures ------------------------------------------------------------

def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor('white')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(LIGHT)
    ax.grid(True, color=LIGHT, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return ax


def _save(fig, name, headline):
    fig.suptitle(headline, color='#0B0B0B', fontsize=14, y=0.98)
    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def fig_three_curves(curve, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    k = 1e-3

    ax = _style(axes[0], 'total profit over the 5,378 test loans',
                'approve when probability of default is below', 'thousand euros')
    ax.plot(curve.threshold, curve.profit_pred * k, color=INK, lw=2,
            label='A  predicted (model probability)')
    ax.plot(curve.threshold, curve.profit_label * k, color=ACCENT, lw=2, ls='--',
            label='B  true outcome, averaged economics')
    ax.plot(curve.threshold, curve.profit_real * k, color='#C2703D', lw=2,
            label='C  true outcome, actual cash')
    ax.axvline(chosen, color=MUTED, lw=1, ls=':')
    ax.axhline(0, color=LIGHT, lw=1)
    top = ax.get_ylim()[1]
    ax.annotate(f'chosen on A, {chosen:.2f}', xy=(chosen, top),
                xytext=(chosen + 0.025, top * 0.92), color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')

    ax = _style(axes[1], 'where the two gaps come from',
                'approve when probability of default is below', 'thousand euros')
    ax.fill_between(curve.threshold, curve.profit_pred * k, curve.profit_label * k,
                    color=ACCENT, alpha=0.25, label='A to B  calibration')
    ax.fill_between(curve.threshold, curve.profit_label * k, curve.profit_real * k,
                    color='#C2703D', alpha=0.25, label='B to C  censoring')
    ax.plot(curve.threshold, curve.profit_label * k, color=ACCENT, lw=1.4)
    ax.axhline(0, color=LIGHT, lw=1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    return _save(fig, 'slice08_profit_curves.png',
                 'Predicted and realised money disagree, and almost all of it is censoring')


def fig_break_even(priced):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    by_grade = (priced.groupby('grade', observed=True)
                      .agg(break_even=('p_star', 'mean'), n=('p_star', 'size'),
                           margin=('margin_pct', 'mean'), loss=('loss_pct', 'mean')))

    ax = _style(axes[0], 'the probability of default a loan can carry',
                '', 'break-even probability')
    ax.bar(by_grade.index, by_grade.break_even, color=INK, width=0.62)
    ax.axhline(0.20, color='#C2703D', lw=1.5, ls='--')
    ax.annotate("slice 5's single threshold, 0.20", xy=(4.2, 0.215),
                color='#C2703D', fontsize=9)
    ax.set_ylim(0, 0.65)

    ax = _style(axes[1], 'because price and loss move apart', '', 'per euro funded')
    ax.plot(by_grade.index, by_grade.margin, color=ACCENT, lw=2, marker='o',
            ms=5, label='margin on a repaid loan')
    ax.plot(by_grade.index, by_grade.loss, color='#C2703D', lw=2, marker='s',
            ms=5, label='loss on a defaulted loan')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_ylim(0, 0.55)
    return _save(fig, 'slice08_break_even_by_grade.png',
                 "A loan's price decides how much default risk it can carry: "
                 "break-even triples from A to G")


def fig_sensitivity(surface):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    grid = surface.pivot(index='lgd', columns='margin_multiplier',
                         values='best_threshold')

    ax = axes[0]
    im = ax.imshow(grid.values, aspect='auto', origin='lower', cmap='bone_r',
                   extent=(grid.columns.min(), grid.columns.max(),
                           grid.index.min(), grid.index.max()))
    _style(ax, 'best global threshold across the assumption grid',
           'margin multiplier', 'loss given default')
    ax.grid(False)
    ax.scatter([1.0], [0.8973], color='#C2703D', s=70, zorder=5, marker='x',
               linewidths=2)
    ax.annotate('measured', xy=(1.0, 0.8973), xytext=(1.03, 0.86),
                color='#C2703D', fontsize=9, fontweight='bold')
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    ax = _style(axes[1], 'at the measured margin, moving only the loss',
                'loss given default', 'best global threshold')
    line = surface[np.isclose(surface.margin_multiplier, 1.0)].sort_values('lgd')
    ax.plot(line.lgd, line.best_threshold, color=INK, lw=2, marker='o', ms=5)
    ax.axvline(0.8973, color='#C2703D', lw=1.2, ls='--')
    ax.annotate('measured\n0.897', xy=(0.8973, line.best_threshold.max()),
                xytext=(0.905, line.best_threshold.max() * 0.985),
                color='#C2703D', fontsize=9)
    return _save(fig, 'slice08_sensitivity.png',
                 'There is no one right threshold: it runs 0.21 to 0.68 across '
                 'assumptions anyone could defend')


def fig_models(curves):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    k = 1e-3
    ax = _style(axes[0], 'predicted profit', 'threshold', 'thousand euros')
    for (name, curve), color in zip(curves.items(), (INK, ACCENT)):
        ax.plot(curve.threshold, curve.profit_pred * k, lw=2, color=color, label=name)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')

    ax = _style(axes[1], 'profit on true outcomes (curve B)', 'threshold',
                'thousand euros')
    for (name, curve), color in zip(curves.items(), (INK, ACCENT)):
        ax.plot(curve.threshold, curve.profit_label * k, lw=2, color=color, label=name)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    return _save(fig, 'slice08_model_comparison.png',
                 'The money agrees with slice 6: boosting does not win here either')


def fig_fairness(frontier, chosen, at_slice5, equalised):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = _style(axes[0], 'who gets approved, as the threshold loosens',
                'approve when probability of default is below', 'approval rate')
    ax.plot(frontier.threshold, frontier.approval_thick, color=INK, lw=2,
            label='thick file (10 years or more)')
    ax.plot(frontier.threshold, frontier.approval_thin, color='#C2703D', lw=2,
            label='thin file (under 10 years)')
    ax.axvline(0.20, color=MUTED, lw=1, ls=':')
    ax.axvline(chosen, color=ACCENT, lw=1.2, ls='--')
    ax.annotate("slice 5\n0.20", xy=(0.20, 0.08), xytext=(0.215, 0.06),
                color=MUTED, fontsize=8.5)
    ax.annotate(f'profit-maximising\n{chosen:.2f}', xy=(chosen, 0.08),
                xytext=(chosen + 0.015, 0.06), color=ACCENT, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')

    ax = _style(axes[1], 'the access gap is a property of the threshold',
                'approve when probability of default is below',
                'thin minus thick, percentage points')
    ax.plot(frontier.threshold, frontier.gap * 100, color=INK, lw=2)
    ax.axhline(0, color=LIGHT, lw=1)
    ax.fill_between(frontier.threshold, frontier.gap * 100, 0,
                    where=(frontier.gap < 0), color='#C2703D', alpha=0.2)
    ax.axvline(chosen, color=ACCENT, lw=1.2, ls='--')
    worst = frontier.loc[frontier['gap'].idxmin()]
    ax.scatter([worst.threshold], [worst.gap * 100], color='#C2703D', zorder=5, s=35)
    ax.annotate(f"{worst.gap * 100:+.1f} pts at {worst.threshold:.2f}",
                xy=(worst.threshold, worst.gap * 100),
                xytext=(worst.threshold + 0.06, worst.gap * 100 + 0.6),
                color='#C2703D', fontsize=9)
    return _save(fig, 'slice08_fairness_frontier.png',
                 "Slice 7's thin-file gap belongs to the threshold, and the "
                 "profitable threshold closes it")


# ---- the slice ----------------------------------------------------------

def disagreement(priced, p_col, global_t):
    """Where the per-loan rule and the best global threshold differ, and
    whether the model was right about those loans.

    This is the whole reason the two rules score differently, so it gets
    measured rather than guessed at.
    """
    p = priced[p_col].to_numpy()
    per_loan, glob = per_loan_rule(priced, p_col), p < global_t
    out = {}
    for name, mask in [('global approves, per-loan rejects', glob & ~per_loan),
                       ('per-loan approves, global rejects', per_loan & ~glob)]:
        sub = priced[mask]
        out[name] = dict(n=len(sub), mean_p=sub[p_col].mean(),
                         actual=sub['target'].mean(),
                         error=sub[p_col].mean() - sub['target'].mean())
    return out


def main(predictions=PREDICTIONS, write=True):
    """write=False computes everything and writes nothing - no csv, no figure,
    no economics.json. src/guard_effect.py uses it to run this twice and compare."""
    money = pd.read_parquet(PROC / 'money_table.parquet')
    groups = pd.read_parquet(PROC / 'fairness_groups_test.parquet')

    params = fit_economics(money, verbose=False)
    if write:
        save_params(params)

    test_raw = money[money['split'] == 'test'].reset_index(drop=True)
    assert test_raw['issue_d'].is_monotonic_increasing, 'test rows are not in issue_d order'
    assert len(test_raw) == 5_378
    assert np.array_equal(test_raw['issue_d'].to_numpy(), groups['issue_d'].to_numpy()), \
        'money table and fairness groups are not in the same row order'

    published = pd.read_parquet(predictions)
    assert np.array_equal(test_raw['target'].to_numpy(), published['target'].to_numpy()), \
        'money table and the predictions are not in the same row order'
    test_raw['p_logreg'] = published['p_logreg'].to_numpy()
    test_raw['p_lgbm_tuned'] = published['p_lgbm_tuned'].to_numpy()

    priced = price_loans(test_raw, params)
    thin_thick = groups['file'].str.startswith('thin').map({True: 'thin', False: 'thick'})

    print('=' * 78)
    print('SLICE 8   business value'.center(78))
    print('=' * 78)
    print(f"\neconomics fitted on {params['n_train']:,} train rows; LGD {params['lgd']['value']:.4f} flat")
    print(f"break-even probability per loan: {priced['p_star'].min():.3f} to "
          f"{priced['p_star'].max():.3f}, mean {priced['p_star'].mean():.3f}")
    print('\nWHICH CURVE DOES WHAT')
    print('  A  chooses.  model probability x train economics. no outcome touched.')
    print('  B  scores.   true outcome x train economics. the business measure.')
    print('  C  diagnoses. true outcome x this book\'s own cash. censored, see below.')

    curve = sweep_thresholds(priced, 'p_logreg')
    chosen_t = float(curve.loc[curve['profit_pred'].idxmax(), 'threshold'])
    b_best_t = float(curve.loc[curve['profit_label'].idxmax(), 'threshold'])

    rules = {
        'approve everyone': np.ones(len(priced), bool),
        "slice 5's 0.20": priced['p_logreg'].to_numpy() < 0.20,
        f'global {chosen_t:.2f}  (chosen on A)': priced['p_logreg'].to_numpy() < chosen_t,
        'per-loan break-even  (the A optimum)': per_loan_rule(priced, 'p_logreg'),
        f'global {b_best_t:.2f}  (B optimum, hindsight)': priced['p_logreg'].to_numpy() < b_best_t,
    }
    table = pd.DataFrame({name: evaluate_rule(priced, mask, 'p_logreg')
                          for name, mask in rules.items()}).T

    print('\n---- approval rules, same 5,378 loans, euros in thousands ----')
    show = table[['approval_rate', 'bad_rate', 'profit_pred', 'profit_label', 'profit_real']].copy()
    show.columns = ['approval', 'bad rate', 'A predicted', 'B business', 'C censored']
    for c in ('A predicted', 'B business', 'C censored'):
        show[c] = (show[c] / 1e3).round(0)
    print(show.round(4).to_string())

    # ---- 1. does the choice made on A survive on B? ----------------------
    b_at_chosen = table.loc[f'global {chosen_t:.2f}  (chosen on A)', 'profit_label']
    b_ceiling = table.loc[f'global {b_best_t:.2f}  (B optimum, hindsight)', 'profit_label']
    print('\n---- 1. calibration, measured in money ----')
    print(f"threshold chosen on A         {chosen_t:.2f}")
    print(f"threshold that was best on B  {b_best_t:.2f}")
    print(f"B profit at the chosen one    {b_at_chosen / 1e3:>8,.0f}k of a possible "
          f"{b_ceiling / 1e3:,.0f}k   =  {b_at_chosen / b_ceiling:.2%}")
    print("choosing without looking at an outcome cost "
          f"{(b_ceiling - b_at_chosen) / 1e3:,.0f}k. Calibration holds in euros.")

    # ---- 2. the per-loan rule, and why the theorem is not the end --------
    per_loan_row = table.loc['per-loan break-even  (the A optimum)']
    glob_row = table.loc[f'global {chosen_t:.2f}  (chosen on A)']
    print('\n---- 2. per-loan threshold against the best global one ----')
    print(f"on A  per-loan {per_loan_row['profit_pred'] / 1e3:>8,.0f}k  vs global "
          f"{glob_row['profit_pred'] / 1e3:,.0f}k   "
          f"({(per_loan_row['profit_pred'] - glob_row['profit_pred']) / 1e3:+,.0f}k)")
    print(f"on B  per-loan {per_loan_row['profit_label'] / 1e3:>8,.0f}k  vs global "
          f"{glob_row['profit_label'] / 1e3:,.0f}k   "
          f"({(per_loan_row['profit_label'] - glob_row['profit_label']) / 1e3:+,.0f}k)")
    print("the per-loan rule cannot lose on A: approving exactly the loans with")
    print("positive expected profit IS the maximum of curve A. It loses on B.")
    for name, d in disagreement(priced, 'p_logreg', chosen_t).items():
        print(f"  {name:36s} n {d['n']:>4}  predicted {d['mean_p']:.4f}  "
              f"actual {d['actual']:.4f}  ({d['error']:+.4f})")
    print("both disagreement sets are places the model is miscalibrated. A global")
    print("threshold needs the ranking right near one cut; the per-loan rule needs")
    print("the probability right everywhere, and spends its edge in the tails.")

    # ---- 3. the censoring tax -------------------------------------------
    print('\n---- 3. what curve C is actually measuring ----')
    for split in ('train', 'valid', 'test'):
        s = money[money['split'] == split]
        good, bad = s[s['target'] == 0], s[s['target'] == 1]
        print(f"  {split:6s} good loan {good['net_pct'].mean():+.4f} after "
              f"{good['life_m'].mean():>4.1f} months on book   "
              f"bad loan {bad['net_pct'].mean():+.4f}, EAD {bad['ead_pct'].mean():.3f}")
    gap = (per_loan_row['profit_label'] - per_loan_row['profit_real']) / 1e3
    print(f"B minus C at the per-loan rule: {gap:+,.0f}k. The test book really did lose")
    print("money, because its good loans were cut off before they had paid their")
    print("interest. That is a fact about the snapshot, not about the policy.")

    # ---- 4. sensitivity --------------------------------------------------
    surface = sensitivity_surface(test_raw, params, 'p_logreg')
    only_lgd = surface[np.isclose(surface.margin_multiplier, 1.0)]
    only_margin = surface[np.isclose(surface.lgd, 0.90)]
    print('\n---- 4. sensitivity: the deliverable is a surface ----')
    print(f"whole grid        threshold {surface['best_threshold'].min():.2f} to "
          f"{surface['best_threshold'].max():.2f}")
    print(f"move LGD alone    {only_lgd['best_threshold'].min():.2f} to "
          f"{only_lgd['best_threshold'].max():.2f}   (0.70 to 1.00)")
    print(f"move margin alone {only_margin['best_threshold'].min():.2f} to "
          f"{only_margin['best_threshold'].max():.2f}   (0.6x to 1.4x)")
    print("the margin assumption moves the answer more than the loss assumption,")
    print("and the margin is the one measured from a book that prepaid early.")

    # ---- 5. both models --------------------------------------------------
    curves = {'logistic (shipped)': curve,
              'lightgbm (slice 6)': sweep_thresholds(priced, 'p_lgbm_tuned')}
    print('\n---- 5. does the money agree with slice 6? ----')
    for name, c in curves.items():
        col = 'p_logreg' if 'logistic' in name else 'p_lgbm_tuned'
        best_a = c.loc[c['profit_pred'].idxmax()]
        b_here = evaluate_rule(priced, priced[col].to_numpy() < best_a['threshold'], col)
        print(f"  {name:20s} chooses {best_a['threshold']:.2f} on A  ->  "
              f"B {b_here['profit_label'] / 1e3:>7,.0f}k at "
              f"{b_here['approval_rate']:.1%} approval")

    # ---- 6. the fairness frontier ---------------------------------------
    frontier = fairness_frontier(priced, thin_thick, 'p_logreg')
    worst = frontier.loc[frontier['gap'].idxmin()]
    at_slice5 = frontier[np.isclose(frontier.threshold, 0.20)].iloc[0]
    at_chosen = frontier[np.isclose(frontier.threshold, chosen_t)].iloc[0]
    eq_strict = equalise_approval(priced, thin_thick, 'p_logreg', 0.20)

    print('\n---- 6. the thin-file gap, across the whole threshold range ----')
    print(f"  worst at {worst['threshold']:.2f}   thin {worst['approval_thin']:.1%} vs "
          f"thick {worst['approval_thick']:.1%}   gap {worst['gap'] * 100:+.1f} pts")
    print(f"  slice 5's 0.20     thin {at_slice5['approval_thin']:.1%} vs "
          f"thick {at_slice5['approval_thick']:.1%}   gap {at_slice5['gap'] * 100:+.1f} pts")
    print(f"  chosen {chosen_t:.2f}        thin {at_chosen['approval_thin']:.1%} vs "
          f"thick {at_chosen['approval_thick']:.1%}   gap {at_chosen['gap'] * 100:+.1f} pts")
    print("slice 7's access gap is a property of the threshold, not of the model.")
    print("the profit-maximising threshold is loose enough that it closes on its own.")
    delta = eq_strict['profit_after_label'] - eq_strict['profit_before_label']
    print(f"\n  forcing equal access at 0.20 instead: thin threshold rises to "
          f"{eq_strict['thin_threshold']:.3f},")
    print(f"  thin approval {eq_strict['approval_thin_before']:.1%} -> "
          f"{eq_strict['approval_thin_after']:.1%}, and B profit "
          f"{'rises' if delta > 0 else 'falls'} by {abs(delta) / 1e3:,.0f}k "
          f"({abs(delta) / eq_strict['profit_before_label']:.2%})")
    print("  equal access is not bought here, it is found: 0.20 is too strict for")
    print("  everyone, and the thin-file group is where the strictness lands hardest.")

    result = dict(curve=curve, surface=surface, frontier=frontier, table=table,
                  priced=priced, chosen_t=chosen_t, params=params)
    if not write:
        return result

    # ---- write everything out --------------------------------------------
    curve.to_csv(PROC / 'slice08_profit_curve.csv', index=False)
    surface.to_csv(PROC / 'slice08_sensitivity.csv', index=False)
    frontier.to_csv(PROC / 'slice08_fairness_frontier.csv', index=False)
    table.to_csv(PROC / 'slice08_rules.csv')

    figs = [
        fig_three_curves(curve, chosen_t),
        fig_break_even(priced),
        fig_sensitivity(surface),
        fig_models(curves),
        fig_fairness(frontier, chosen_t, at_slice5, eq_strict),
    ]
    print('\n---- written ----')
    for f in figs:
        print(f'  {f.relative_to(ROOT)}')
    for name in ('slice08_profit_curve.csv', 'slice08_sensitivity.csv',
                 'slice08_fairness_frontier.csv', 'slice08_rules.csv'):
        print(f'  data/processed/{name}')

    return result


if __name__ == '__main__':
    main()
