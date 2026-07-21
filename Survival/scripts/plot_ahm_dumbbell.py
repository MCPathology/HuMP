"""
Dumbbell (connected dot) plot of AHM per-center 5-fold C-Index:
scratch vs. LIHC-pretrained fine-tune, with the TCGA-LIHC internal
result as a vertical reference line.

Non-bar visualization: each center is one row; an arrow connects the
from-scratch point to the fine-tuned point, error bars show +/- std.

    python scripts/plot_ahm_dumbbell.py --out fig_ahm_dumbbell.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- results (edit here if numbers change) ----
TCGA_LIHC = 0.6586
DATA = {            # center: (n, scratch_mean, scratch_std, ft_mean, ft_std)
    'C1': (500, 0.6719, 0.0416, 0.6920, 0.0417),
    'C2': (228, 0.6709, 0.0659, 0.6814, 0.0520),
    'C3': (120, 0.6698, 0.0849, 0.7704, 0.0963),
}

C_SCRATCH = '#A6BFD4'   # slate blue (from-scratch)
C_FT      = '#C97A7A'   # dusty rose (fine-tune, highlight)
C_REF     = '#888888'   # reference line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='fig_ahm_dumbbell.png')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 12,
        'legend.fontsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    centers = list(DATA.keys())
    y = np.arange(len(centers))[::-1]          # C1 on top

    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=args.dpi)

    # TCGA-LIHC reference line
    ax.axvline(TCGA_LIHC, color=C_REF, ls='--', lw=1.2, alpha=0.8, zorder=0)
    ax.text(TCGA_LIHC, len(centers) - 0.35, f'TCGA-LIHC\n{TCGA_LIHC:.3f}',
            color=C_REF, fontsize=9, ha='center', va='bottom')

    for yi, c in zip(y, centers):
        n, sm, ss, fm, fs = DATA[c]
        # connecting arrow scratch -> finetune
        ax.annotate('', xy=(fm, yi), xytext=(sm, yi),
                    arrowprops=dict(arrowstyle='-|>', color='#bbbbbb',
                                    lw=1.4, shrinkA=4, shrinkB=4))
        # scratch point + error bar
        ax.errorbar(sm, yi, xerr=ss, fmt='o', ms=9, color=C_SCRATCH,
                    ecolor=C_SCRATCH, elinewidth=1.3, capsize=3,
                    mfc='white', mec=C_SCRATCH, mew=1.8, zorder=3)
        # finetune point + error bar
        ax.errorbar(fm, yi, xerr=fs, fmt='o', ms=9, color=C_FT,
                    ecolor=C_FT, elinewidth=1.3, capsize=3, zorder=4)
        # delta annotation
        d = fm - sm
        ax.text(max(fm, sm) + max(fs, ss) + 0.012, yi,
                f'+{d:.3f}', va='center', ha='left',
                fontsize=9.5, color='#7a4a4a')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{c}\n(n={DATA[c][0]})' for c in centers])
    ax.set_xlabel('C-Index (5-fold mean $\\pm$ std)')
    ax.set_xlim(0.55, 0.92)
    ax.set_ylim(-0.6, len(centers) - 0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, lw=0.5)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', mfc='white', mec=C_SCRATCH, mew=1.8,
               color='w', ms=9, label='From scratch (AHM)'),
        Line2D([0], [0], marker='o', color='w', mfc=C_FT, ms=9,
               label='Fine-tune (LIHC$\\rightarrow$AHM)'),
        Line2D([0], [0], color=C_REF, ls='--', lw=1.2,
               label='TCGA-LIHC internal'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True,
              framealpha=0.9, edgecolor='#dddddd', fontsize=9)

    ax.set_title('Per-center external validation on AHM', pad=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
