"""
Diagnostic visualization of P / O / C token distributions for ONE patient.

Reads a single-patient .npz produced by
    scripts/extract_hgs_features.py --single_patient N --out X.npz
and renders a 1x3 panel with PCA-2D scatter of each modality's token bag.

For omics, sub-modality boundaries (genomic / transomic / protein) are
overlaid so the multi-sub-encoder structure is visible at a glance.

Usage:
    python scripts/plot_modality_tokens.py \
        --npz feats/brca_fold2_patient0.npz \
        --out fig_modality_tokens.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'P_main': '#406A8E',     # blue
    'O_main': '#6E956A',     # green (overall O cluster mean for ref)
    'O_g':    '#9C7AB0',     # purple — genomic sub-bag
    'O_t':    '#C99696',     # rose   — transomic sub-bag
    'O_p':    '#CFB99E',     # tan    — protein sub-bag
    'C_main': '#9A6B3F',     # brown
}


def pca_2d(X):
    """Return PCA-2D projection of X[N, D] and the centred mean."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z


def draw_ellipse(ax, pts, color, ls='-', alpha=0.55):
    from matplotlib.patches import Ellipse
    if len(pts) < 5:
        return
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    for s, ls_ in zip([1.0, 2.0], ['-', '--']):
        w = 2 * s * np.sqrt(max(eigvals[0], 1e-12))
        h = 2 * s * np.sqrt(max(eigvals[1], 1e-12))
        el = Ellipse(xy=mu, width=w, height=h, angle=angle,
                     fc='none', ec=color, lw=0.8,
                     linestyle=ls_, alpha=alpha)
        ax.add_patch(el)


def panel_modality(ax, tokens, color, title, ellipse=True,
                   sub_groups=None):
    """Plot one modality's tokens, optionally colour-coded by sub-groups.

    sub_groups : list of (label, slice, color) tuples for omics colouring.
    """
    Z = pca_2d(tokens)
    if sub_groups is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=10, color=color, alpha=0.65,
                   edgecolor='none', label=f'N={len(Z)}')
        if ellipse:
            draw_ellipse(ax, Z, color)
    else:
        for label, sl, c in sub_groups:
            sub = Z[sl]
            if len(sub) == 0:
                continue
            ax.scatter(sub[:, 0], sub[:, 1], s=10, color=c, alpha=0.7,
                       edgecolor='none', label=f'{label} (N={len(sub)})')
            if ellipse and len(sub) >= 5:
                draw_ellipse(ax, sub, c, alpha=0.35)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(title, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.legend(loc='best', frameon=True, framealpha=0.85,
              edgecolor='#dddddd', fontsize=9,
              handlelength=1.0, borderpad=0.35, labelspacing=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True,
                    help='Single-patient npz from extract_hgs_features.py')
    ap.add_argument('--out', default='fig_modality_tokens.png')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--figsize', type=float, nargs=2, default=[14.0, 4.4])
    args = ap.parse_args()

    # ---- SciencePlots style if available -------------------------------
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({
        'axes.titlesize':  13,
        'axes.labelsize':  12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth':  0.7,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
    })

    # ---- load --------------------------------------------------------
    d = np.load(args.npz, allow_pickle=True)
    if 'mode' in d.files and str(d['mode']) != 'single_patient':
        raise SystemExit(f"npz mode={d['mode']!r} is not single_patient")
    P = d['P'].astype(np.float32)      # [N_patch, 256]
    O = d['O'].astype(np.float32)      # [N_omic,  256]
    C = d['C'].astype(np.float32)      # [N_c,     256]
    pid = str(d['patient_id']) if 'patient_id' in d.files else 'unknown'
    cohort = str(d['cohort']) if 'cohort' in d.files else 'unknown'
    o_split = d['o_split'] if 'o_split' in d.files else None
    if o_split is not None:
        n_g, n_t, n_p = [int(x) for x in o_split]
    else:
        n_g, n_t, n_p = None, None, None

    print(f"[load]  cohort={cohort}  patient={pid}")
    print(f"        P:{P.shape}  O:{O.shape}  C:{C.shape}")
    if n_g is not None:
        print(f"        O breakdown: genomic={n_g}  transomic={n_t}  protein={n_p}")

    # ---- L2-normalize so PCA only reflects direction structure ------
    def _normalize(X):
        n = np.linalg.norm(X, axis=-1, keepdims=True)
        n = np.clip(n, 1e-12, None)
        return X / n
    P = _normalize(P)
    O = _normalize(O)
    C = _normalize(C)

    # ---- build figure -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=tuple(args.figsize), dpi=args.dpi)

    panel_modality(axes[0], P, COLORS['P_main'],
                   title=f'(a)  Pathology  ($N={P.shape[0]}$ patch tokens)')

    o_sub = None
    if n_g is not None and (n_g + n_t + n_p) == O.shape[0]:
        o_sub = [
            ('genomic',   slice(0,                n_g),                 COLORS['O_g']),
            ('transomic', slice(n_g,              n_g + n_t),           COLORS['O_t']),
            ('protein',   slice(n_g + n_t,        n_g + n_t + n_p),     COLORS['O_p']),
        ]
    panel_modality(axes[1], O, COLORS['O_main'],
                   title=f'(b)  Omics  ($N={O.shape[0]}$ tokens)',
                   sub_groups=o_sub)

    panel_modality(axes[2], C, COLORS['C_main'],
                   title=f'(c)  Clinical  ($N={C.shape[0]}$ attribute tokens)')

    fig.suptitle(f'{cohort.upper()} — patient {pid}',
                 fontsize=14, y=1.02)
    plt.tight_layout(w_pad=2.5)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.1)
    print(f"[done]  figure saved -> {args.out}")


if __name__ == '__main__':
    main()
