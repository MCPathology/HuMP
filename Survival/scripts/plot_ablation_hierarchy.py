"""
Dual-y-axis grouped bar chart for Table 8 (embedding space x hierarchy).
C-Index (left axis) and t-AUC (right axis) each get their own zoomed scale
so both metrics' variations are clearly visible.  HuMP highlighted.

    python scripts/plot_ablation_hierarchy.py --out fig_ablation_hier.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (space, hierarchy, C-Index, t-AUC, is_hump)
ROWS = [
    ('Euclidean',  'None',        68.5, 51.5, False),
    ('Euclidean',  'Directional', 66.3, 51.6, False),
    ('Hyperbolic', 'None',        68.8, 49.3, False),
    ('Hyperbolic', 'Symmetric',   69.5, 51.7, False),
    ('Hyperbolic', 'Reverse',     69.1, 51.9, False),
    ('Hyperbolic', 'Directional', 71.2, 52.8, True),   # HuMP
]
C_CIDX = '#9BBBBE'   # muted teal  (C-Index)
C_TAUC = '#A3ABCF'   # muted blue  (t-AUC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='fig_ablation_hier.png')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    # ---- Times New Roman (use STIX, a TNR-metric-compatible serif) ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',          # Times-like math
        'axes.titlesize': 19, 'axes.labelsize': 18,
        'xtick.labelsize': 15, 'ytick.labelsize': 15,
        'legend.fontsize': 16, 'pdf.fonttype': 42,
    })

    n = len(ROWS)
    x = np.arange(n)
    w = 0.38
    cidx = [r[2] for r in ROWS]
    tauc = [r[3] for r in ROWS]
    hump = [r[4] for r in ROWS]

    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=args.dpi)
    ax2 = ax.twinx()

    b1 = ax.bar(x - w / 2, cidx, w, color=C_CIDX, edgecolor='#333',
                linewidth=0.6, label='C-Index', zorder=3)
    b2 = ax2.bar(x + w / 2, tauc, w, color=C_TAUC, edgecolor='#333',
                 linewidth=0.6, label='t-AUC', zorder=3)

    # zoomed independent ranges so both metrics' variation is visible
    ax.set_ylim(64, 73)
    ax2.set_ylim(47, 55)

    # Euclidean vs Hyperbolic divider + group labels
    ax.axvline(1.5, color='#444', ls='--', lw=2.2, zorder=5)
    ax.text(0.5, 72.2, 'Euclidean', ha='center', fontsize=17, style='italic')
    ax.text(3.5, 72.2, 'Hyperbolic', ha='center', fontsize=17, style='italic')

    labels = [r[1] + ('\n(HuMP)' if r[4] else '') for r in ROWS]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Hierarchical alignment')
    ax.set_ylabel('C-Index (%)', color='#4f7b7e')
    ax2.set_ylabel('t-AUC (%)', color='#5b639a')
    ax.tick_params(axis='y', labelcolor='#4f7b7e')
    ax2.tick_params(axis='y', labelcolor='#5b639a')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, lw=0.5)

    # combined legend ABOVE the axes (avoids overlap with bars / group labels)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
