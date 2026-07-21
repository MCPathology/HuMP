"""
Per-center box plot of AHM C-Index: Scratch / Fine-tune (5-fold) and
Zero-shot LOCO (5-seed).  One figure per center (C1/C2/C3) so they can be
arranged in a row, with KM curves in the row below.

    python scripts/plot_ahm_box.py --out_dir ahm_box
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TCGA_LIHC = 65.86   # percent
N = {'C1': 500, 'C2': 228, 'C3': 120}
# percent scale; scratch/finetune = 5-fold, zeroshot = 5-seed LOCO
FOLDS = {
    'C1': {'scratch':  [62.0, 71.3, 62.3, 71.0, 69.3],
           'finetune': [62.6, 73.8, 66.1, 72.0, 71.5],
           'zeroshot': [65.2, 65.2, 65.4, 64.7, 64.9]},
    'C2': {'scratch':  [79.7, 64.5, 61.4, 62.7, 67.1],
           'finetune': [76.2, 65.1, 64.8, 62.3, 72.3],
           'zeroshot': [72.0, 71.7, 71.5, 72.3, 71.8]},
    'C3': {'scratch':  [71.1, 59.8, 57.8, 67.6, 78.6],
           'finetune': [71.1, 62.0, 60.0, 72.1, 84.1],
           'zeroshot': [67.2, 66.8, 66.4, 67.0, 67.5]},
}
# <<< EDIT COLORS HERE >>>
C_SCRATCH = '#52ADAD'   # teal
C_FT      = '#DB5C56'   # red
C_ZS      = '#7E9BC9'   # blue   (zero-shot)
C_REF     = '#333333'   # TCGA reference line

ORDER = [('scratch', 'Scratch', C_SCRATCH),
         ('finetune', 'Fine-tune', C_FT),
         ('zeroshot', 'Zero-shot', C_ZS)]


def one_center(ctr, out_path, dpi):
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(3.9, 3.3), dpi=dpi)
    pos = list(range(1, len(ORDER) + 1))
    labels = [lab for _, lab, _ in ORDER]

    # 5-fold protocols (scratch/finetune) -> boxes;  5-seed zero-shot ->
    # point + error bar (its variance is intrinsically tiny, a box degenerates
    # to a flat line).
    box_pos, box_data, box_col = [], [], []
    for p, (key, _, c) in zip(pos, ORDER):
        vals = np.asarray(FOLDS[ctr][key])
        if key == 'zeroshot':
            ax.errorbar(p, vals.mean(), yerr=vals.std(), fmt='D', ms=8,
                        color=c, ecolor=c, elinewidth=1.6, capsize=4,
                        capthick=1.6, zorder=4)
            ax.scatter(np.full(len(vals), p), vals, s=10, color=c,
                       edgecolor='#333', linewidth=0.3, zorder=5)
        else:
            box_pos.append(p); box_data.append(list(vals)); box_col.append(c)

    bp = ax.boxplot(box_data, positions=box_pos, widths=0.32,
                    patch_artist=True, showfliers=False, zorder=2)
    for b, c in zip(bp['boxes'], box_col):
        b.set(facecolor=c, edgecolor='#444', linewidth=1.0)
    for w in bp['whiskers']: w.set(color='#444', linewidth=1.0)
    for cap in bp['caps']:   cap.set(color='#444', linewidth=1.0)
    for m in bp['medians']:  m.set(color='#222', linewidth=1.4)
    for p, vals, c in zip(box_pos, box_data, box_col):
        jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.10
        ax.scatter(np.array([p] * len(vals)) + jit, vals, s=14, color=c,
                   edgecolor='#333', linewidth=0.3, zorder=3)

    # TCGA-LIHC reference line ON TOP (label added by user)
    ax.axhline(TCGA_LIHC, color=C_REF, ls=(0, (5, 3)), lw=1.6, zorder=10)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlim(0.4, len(ORDER) + 0.6)
    ax.set_ylabel('C-Index (%)')
    ax.set_ylim(55, 88)
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
    ap.add_argument('--out_dir', default='ahm_box')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ctr in FOLDS:
        one_center(ctr, os.path.join(args.out_dir, f'box_{ctr}.png'), args.dpi)


if __name__ == '__main__':
    main()
