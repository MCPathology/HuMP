"""
Per-center horizontal violin plots of AHM 5-fold C-Index (scratch vs
fine-tune), styled after the reference figure (horizontal violins, inner
box, mean label above each, soft palette).  One PNG per center so they can
be arranged in a row.

    python scripts/plot_ahm_violin.py --out_dir ahm_violins
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N = {'C1': 500, 'C2': 228, 'C3': 120}
FOLDS = {   # percent scale (50-100)
    'C1': {'scratch': [62.0, 71.3, 62.3, 71.0, 69.3],
           'finetune': [62.6, 73.8, 66.1, 72.0, 71.5]},
    'C2': {'scratch': [79.7, 64.5, 61.4, 62.7, 67.1],
           'finetune': [76.2, 65.1, 64.8, 62.3, 72.3]},
    'C3': {'scratch': [71.1, 59.8, 57.8, 67.6, 78.6],
           'finetune': [71.1, 62.0, 60.0, 72.1, 84.1]},
}
# colors sampled from the reference figure
C_FT = '#E78A6C'       # coral/salmon  (fine-tune, highlight)
C_SCRATCH = '#56BBA9'  # teal/green    (from scratch)


def one_center(ctr, out_path, dpi):
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'pdf.fonttype': 42})

    ft = np.array(FOLDS[ctr]['finetune'])
    sc = np.array(FOLDS[ctr]['scratch'])
    data = [ft, sc]                  # top = finetune, bottom = scratch
    colors = [C_FT, C_SCRATCH]
    ypos = [2, 1]

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=dpi)
    vp = ax.violinplot(data, positions=ypos, vert=False, widths=0.85,
                       showmeans=False, showextrema=False, showmedians=False,
                       bw_method=0.6)
    for body, col in zip(vp['bodies'], colors):
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(1.0); body.set_linewidth(0)

    # inner mini box + mean marker + mean label
    for vals, yp, col in zip(data, ypos, colors):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = vals.min(), vals.max()
        ax.hlines(yp, lo, hi, color='#555', lw=1.0, zorder=3)           # whisker
        ax.add_patch(plt.Rectangle((q1, yp - 0.06), q3 - q1, 0.12,
                                   facecolor='#444', edgecolor='none', zorder=4))
        ax.scatter(med, yp, marker='D', s=14, color='white',
                   edgecolor='#222', linewidth=0.5, zorder=5)
        ax.text((lo + hi) / 2, yp + 0.45, f'{vals.mean():.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_yticks(ypos)
    ax.set_yticklabels(['Fine-tune', 'Scratch'])
    ax.set_xlim(50, 100)
    ax.set_xticks([50, 75, 100])
    ax.set_xlabel('C-Index')
    ax.set_ylim(0.4, 2.9)
    ax.set_title(f'{ctr} (n={N[ctr]})', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='ahm_violins')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ctr in FOLDS:
        one_center(ctr, os.path.join(args.out_dir, f'violin_{ctr}.png'), args.dpi)


if __name__ == '__main__':
    main()
