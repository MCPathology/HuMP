"""
Per-center zero/few-shot data-efficiency line plots.
x = proportion of target-center training data used for adaptation
    (0 = zero-shot, 1.0 = full fine-tune); y = C-Index (%).
Shaded band = +/- std over 5 folds.  One PNG per center.

    python scripts/plot_fewshot_curve.py --out_dir fewshot_curve
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TCGA_LIHC = 65.86   # percent (internal reference)
N = {'C1': 500, 'C2': 228, 'C3': 120}
RATIOS = [0.0, 0.1, 0.25, 0.5, 1.0]
# per-fold C-Index (%); 1.0 = fine-tune folds
DATA = {
    'C1': {0.0:  [53.3, 62.6, 56.2, 61.2, 50.2],
           0.1:  [56.9, 64.6, 60.7, 61.8, 57.8],
           0.25: [66.0, 58.8, 66.3, 63.0, 58.8],
           0.5:  [73.3, 62.5, 64.1, 67.2, 60.5],
           1.0:  [62.6, 73.8, 66.1, 72.0, 71.5]},
    'C2': {0.0:  [46.6, 59.7, 55.9, 52.6, 55.9],
           0.1:  [64.7, 52.0, 56.1, 64.2, 63.3],
           0.25: [67.0, 60.3, 58.0, 68.2, 62.6],
           0.5:  [68.0, 60.9, 55.1, 62.9, 79.0],
           1.0:  [76.2, 65.1, 64.8, 62.3, 72.3]},
    'C3': {0.0:  [63.2, 49.7, 47.0, 57.0, 39.7],
           0.1:  [40.9, 60.4, 66.1, 68.9, 61.8],
           0.25: [57.3, 68.1, 60.7, 74.2, 70.2],
           0.5:  [57.4, 64.9, 66.7, 71.5, 64.2],
           1.0:  [71.1, 62.0, 60.0, 72.1, 84.1]},
}
# <<< EDIT COLORS HERE >>>
COL = {'C1': '#7E9BC9', 'C2': '#7E9BC9', 'C3': '#7E9BC9'}
C_REF = '#333333'
XLAB = ['0%', '10%', '25%', '50%', '100%']


def one_center(ctr, out_path, dpi):
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 13,
                         'xtick.labelsize': 10.5, 'ytick.labelsize': 11,
                         'pdf.fonttype': 42})

    xs = np.arange(len(RATIOS))
    mu = np.array([np.mean(DATA[ctr][r]) for r in RATIOS])
    sd = np.array([np.std(DATA[ctr][r]) for r in RATIOS])
    col = COL[ctr]

    fig, ax = plt.subplots(figsize=(3.9, 3.3), dpi=dpi)
    ax.bar(xs, mu, width=0.62, color=col, edgecolor='#333', linewidth=0.7,
           yerr=sd, capsize=4, error_kw={'elinewidth': 1.2, 'capthick': 1.2,
                                         'ecolor': '#333'}, zorder=3)

    # TCGA-LIHC internal reference, on top
    ax.axhline(TCGA_LIHC, color=C_REF, ls=(0, (5, 3)), lw=1.5, zorder=10)

    ax.set_xticks(xs); ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.6, len(RATIOS) - 0.4)
    ax.set_xlabel('Proportion of training data')
    ax.set_ylabel('C-Index (%)')
    ax.set_ylim(40, 82)
    ax.set_title(f'{ctr} (n={N[ctr]})', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.18, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='fewshot_curve')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ctr in DATA:
        one_center(ctr, os.path.join(args.out_dir, f'fewshot_{ctr}.png'), args.dpi)


if __name__ == '__main__':
    main()
