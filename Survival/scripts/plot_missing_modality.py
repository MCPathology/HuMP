"""
Figure: missing-modality robustness — 1 row × 5 subplots for double-column paper.

Subplot layout:
    (1)  P  missing   — 5 methods × {C-Index, t-AUC}  bar chart
    (2)  O  missing   — same
    (3)  C  missing   — same
    (4)  HuMP double-missing (PC / PO / OC) — bar chart
    (5)  External validation — placeholder (data not yet available)

Conventions:
    * The 3 values in each method's `[a, b, c]` array correspond to
      P-missing / O-missing / C-missing in that order.
    * DisPro has `0.0` placeholders for P-missing (experiment not run);
      those bars are rendered with hatching and an annotation 'N/A'.
    * HuMP is highlighted with a distinct color across all subplots.

Usage:
    python scripts/plot_missing_modality.py \
        --out fig_missing_modality.pdf

Designed for IEEE double-column ~6.8" wide.  Tweak `--figsize` for other
journals.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Data (positional order in each array:  [P-missing, O-missing, C-missing])
# ---------------------------------------------------------------------------
PERFORMANCE = {
    'M$^2$Surv': {'C-Index': np.array([65.0, 66.9, 64.5]),
                  't-AUC':   np.array([49.6, 46.7, 27.6])},
    'DisPro':    {'C-Index': np.array([0.0, 65.9, 62.1]),
                  't-AUC':   np.array([0.0, 47.5, 47.6])},
    'LDCVAE':    {'C-Index': np.array([64.9, 65.4, 63.3]),
                  't-AUC':   np.array([49.7, 46.1, 48.1])},
    'M$^3$Surv': {'C-Index': np.array([65.4, 67.5, 66.4]),
                  't-AUC':   np.array([50.6, 48.4, 48.3])},
    'HuMP':      {'C-Index': np.array([67.7, 70.1, 68.5]),
                  't-AUC':   np.array([50.7, 50.6, 50.3])},
}

DOUBLE_MISSING_HUMP = {
    # Full-modality reference (no missing) — drawn first as the upper bound.
    'Full':       {'C-Index': 71.2, 't-AUC': 52.8},
    'PC missing': {'C-Index': 61.2, 't-AUC': 51.5},
    'PO missing': {'C-Index': 63.7, 't-AUC': 49.3},
    'OC missing': {'C-Index': 63.9, 't-AUC': 52.0},
}

SCENARIO_TITLES = ['Pathology missing', 'Omics missing', 'Clinical missing']

# ---------------------------------------------------------------------------
# Soft pastel palette aligned with the existing paper colours
#   anchor 1: #CCDBE7  (pale blue)
#   anchor 2: #DFD9E8  (pale lavender)
# All five methods sit in the same low-saturation / high-value range so the
# figure reads as a single visual family.  HuMP uses the warmest tone so it
# stands out without breaking the pastel scheme.
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    # Same pastel family as the previous version, but pulled ~20% darker so
    # the bars carry more weight on a white page while staying in the same
    # low-saturation visual range as the paper's anchor colours.
    'M$^2$Surv': '#A6BFD4',   # slate blue          (derived from #CCDBE7)
    'DisPro':    '#B5AECC',   # dusty lavender      (derived from #DFD9E8)
    'LDCVAE':    '#ADC4A7',   # sage green
    'M$^3$Surv': '#CFB99E',   # warm tan
    'HuMP':      '#C99696',   # dusty rose          (HuMP highlight)
}
# t-AUC bars are drawn at slightly lower opacity for the same hue so the
# C-Index / t-AUC distinction stays visible without harsh contrast.
METRIC_ALPHA = {'C-Index': 0.95, 't-AUC': 0.55}


# ---------------------------------------------------------------------------
# Subplot drawers
# ---------------------------------------------------------------------------
def _draw_single_missing(ax, scenario_idx, title):
    """One subplot for a single-modality-missing scenario.

    Groups: methods on x-axis.  Two bars per group: C-Index then t-AUC.
    For the P-missing subplot (scenario_idx == 0) DisPro is dropped, because
    no P-missing result is available for it.
    """
    # P-missing skips DisPro entirely; O/C-missing keep all 5 methods.
    if scenario_idx == 0:
        methods = [m for m in PERFORMANCE.keys() if m != 'DisPro']
    else:
        methods = list(PERFORMANCE.keys())
    n_methods = len(methods)
    bar_w = 0.36
    x = np.arange(n_methods)

    for i, m in enumerate(methods):
        c = PERFORMANCE[m]['C-Index'][scenario_idx]
        t = PERFORMANCE[m]['t-AUC'][scenario_idx]
        color = METHOD_COLORS[m]
        ax.bar(x[i] - bar_w / 2, c, width=bar_w,
               color=color, alpha=METRIC_ALPHA['C-Index'],
               edgecolor='#444', linewidth=0.4)
        ax.bar(x[i] + bar_w / 2, t, width=bar_w,
               color=color, alpha=METRIC_ALPHA['t-AUC'],
               edgecolor='#444', linewidth=0.4, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=12)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_title(title, fontsize=14, pad=6)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # Rotated method labels with ha='right' anchor on the right edge of the
    # text, which visually pulls them to the LEFT of their tick.  Shift each
    # label slightly to the right so it sits under its own bar group.
    dx = 12 / 72.0   # 12 pt ≈ 0.17 inch — bump to taste
    offset = mtransforms.ScaledTranslation(dx, 0, ax.figure.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


def _draw_double_missing(ax):
    """One subplot for HuMP's double-missing scenarios, with the full-modality
    reference bar drawn first so the reader sees the absolute drop.

    Groups: Full / PC / PO / OC on x-axis.  Two bars per group: C-Index, t-AUC.
    None-valued entries (e.g. Full when not yet measured) become a hatched
    'TBD' placeholder bar at a low fixed height.
    """
    scenarios = list(DOUBLE_MISSING_HUMP.keys())
    n_scen = len(scenarios)
    bar_w = 0.36
    x = np.arange(n_scen)

    color = METHOD_COLORS['HuMP']
    tbd_h = 6  # height used purely to show a placeholder slot

    for i, s in enumerate(scenarios):
        c = DOUBLE_MISSING_HUMP[s]['C-Index']
        t = DOUBLE_MISSING_HUMP[s]['t-AUC']

        if c is None and t is None:
            # placeholder bars — light gray, dashed edge, 'TBD' annotation
            ax.bar(x[i] - bar_w / 2, tbd_h, width=bar_w,
                   color='white', edgecolor='#888', linewidth=0.7,
                   linestyle='--')
            ax.bar(x[i] + bar_w / 2, tbd_h, width=bar_w,
                   color='white', edgecolor='#888', linewidth=0.7,
                   linestyle='--', hatch='//')
            ax.text(x[i], tbd_h + 1.0, 'TBD',
                    ha='center', va='bottom',
                    fontsize=11, color='#888', style='italic')
            continue

        ax.bar(x[i] - bar_w / 2, c, width=bar_w,
               color=color, alpha=METRIC_ALPHA['C-Index'],
               edgecolor='#444', linewidth=0.4)
        ax.bar(x[i] + bar_w / 2, t, width=bar_w,
               color=color, alpha=METRIC_ALPHA['t-AUC'],
               edgecolor='#444', linewidth=0.4, hatch='//')
        ax.text(x[i] - bar_w / 2, c + 1.0, f'{c:.1f}',
                ha='center', va='bottom', fontsize=10)
        ax.text(x[i] + bar_w / 2, t + 1.0, f'{t:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Human-readable x-axis labels: 'Full', 'P. & C. missing', etc.
    label_map = {
        'Full':       'Full',
        'PC missing': 'P. & C.\nmissing',
        'PO missing': 'P. & O.\nmissing',
        'OC missing': 'O. & C.\nmissing',
    }
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(s, s) for s in scenarios],
                       fontsize=12)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_title('Double-missing scenarios', fontsize=14, pad=6)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)


def _draw_external_placeholder(ax):
    """Placeholder subplot for external validation (data pending)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.55,
            'External\nvalidation',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color='#444')
    ax.text(0.5, 0.30,
            '(results pending)',
            ha='center', va='center', fontsize=12, style='italic',
            color='#888')
    # dashed border to signal "placeholder"
    for s in ax.spines.values():
        s.set_linestyle((0, (3, 3)))
        s.set_color('#bbb')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('External validation', fontsize=14, pad=6, color='#444')


# ---------------------------------------------------------------------------
# Top-level figure assembly + legend
# ---------------------------------------------------------------------------
def _draw_legend(fig):
    """Shared legend below the figure (methods + metric encoding)."""
    method_handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor='black',
              linewidth=0.5, label=m)
        for m in PERFORMANCE.keys()
    ]
    metric_handles = [
        Patch(facecolor='lightgray', edgecolor='black',
              linewidth=0.5, label='C-Index'),
        Patch(facecolor='lightgray', edgecolor='black',
              linewidth=0.5, hatch='//', alpha=0.55, label='t-AUC'),
    ]
    fig.legend(handles=method_handles + metric_handles,
               loc='lower center', bbox_to_anchor=(0.5, -0.06),
               ncol=len(method_handles) + len(metric_handles),
               fontsize=12, frameon=False,
               handletextpad=0.6, columnspacing=1.2)


def make_figure(out_path, figsize=(13.5, 2.9), dpi=300):
    """Build the 1-row × 5-col figure and save to `out_path`."""
    fig, axes = plt.subplots(
        1, 5, figsize=figsize, dpi=dpi,
        gridspec_kw={'width_ratios': [1.0, 1.0, 1.0, 1.0, 0.7]})

    # subplots 1–3: single-missing scenarios
    for i, title in enumerate(SCENARIO_TITLES):
        _draw_single_missing(axes[i], scenario_idx=i, title=title)
        if i == 0:
            axes[i].set_ylabel('Score (%)', fontsize=8)

    # subplot 4: HuMP double-missing
    _draw_double_missing(axes[3])

    # subplot 5: external validation placeholder
    _draw_external_placeholder(axes[4])

    _draw_legend(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Figure saved -> {out_path}")
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='fig_missing_modality.png',
                    help='Output figure path (.png / .pdf / .svg).')
    ap.add_argument('--figsize', type=float, nargs=2, default=[13.5, 2.9],
                    help='Figure size in inches (W H).')
    ap.add_argument('--dpi', type=int, default=800,
                    help='Output DPI; high-DPI PNG for print-grade clarity.')
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # ----------------------------------------------------------------------
    # SciencePlots IEEE style (`pip install SciencePlots`).
    # `['science', 'ieee']`   sets serif Times-style fonts, ~7-8pt axis labels,
    #                         IEEE-narrow column figure proportions, and
    #                         clean spines compatible with IEEE/InfFusion.
    # `'no-latex'`            avoids requiring a system LaTeX toolchain.
    # Falls back to a hand-tuned IEEE-ish rcParams if SciencePlots is missing.
    # ----------------------------------------------------------------------
    try:
        import scienceplots  # noqa: F401  registers the styles with matplotlib
        plt.style.use(['science', 'ieee', 'no-latex'])
        # bumped up across the board so 5-subplot row stays readable
        plt.rcParams.update({
            'axes.titlesize':  14,
            'axes.labelsize':  13,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'pdf.fonttype':  42,
            'ps.fonttype':   42,
        })
        print("[style] Using SciencePlots 'science+ieee' (no-latex).")
    except ImportError:
        print("[style] SciencePlots not installed (`pip install SciencePlots`); "
              "falling back to IEEE-like serif rcParams.")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif':  ['Times New Roman', 'DejaVu Serif',
                            'Liberation Serif'],
            'mathtext.fontset': 'stix',
            'axes.titlesize':  14,
            'axes.labelsize':  13,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.linewidth':  0.7,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 0.7,
            'ytick.major.width': 0.7,
            'pdf.fonttype':  42,
            'ps.fonttype':   42,
        })

    make_figure(out_path, figsize=tuple(args.figsize), dpi=args.dpi)


if __name__ == '__main__':
    main()
