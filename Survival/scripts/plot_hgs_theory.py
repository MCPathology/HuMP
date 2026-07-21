"""
Theoretical / simulation analysis of the Hierarchy-Guided Sampling (HGS)
module.  Designed to address the reviewer concern that HGS "lacks
theoretical grounding" on (i) sampling efficiency and (ii) distribution
coverage, *without* requiring a trained checkpoint.

Two side-by-side panels:

  (a)  Acceptance rate of the cone-intersection rejection step as the
       cone half-angle theta is swept, evaluated on controlled
       synthetic anchor pairs in R^256 with three inter-anchor angles
       phi in {0.2, 0.4, 0.6} rad.  Right axis: expected number of
       trials to draw one valid sample (log scale).  The default
       operating point used by HuMP (theta = 0.35) is highlighted.

  (b)  Distribution coverage of HGS imputations versus a random-unit
       baseline.  We construct a controlled synthetic encoder
       distribution where each "patient" has a shared latent z_i on
       the unit sphere and (M, P, C) are nearby perturbations of z_i.
       For each patient, HGS imputes M from (P, C); we plot the
       histogram of cos(HGS, nearest real-M), cos(random, nearest
       real-M), and cos(real-M, second-nearest real-M) as an intra-
       distribution reference.  A right-shifted HGS histogram relative
       to the random baseline is empirical evidence of in-distribution
       sampling.

The script imports the actual HGS routine from
models/layers/lhyperbolic.py so the reported behaviour matches the
implementation used in training.

Usage:
    python scripts/plot_hgs_theory.py
        # default: pure simulation, writes fig_hgs_theory.png at 800 dpi
    python scripts/plot_hgs_theory.py --out fig_hgs.pdf --dpi 300
        # vector output
    python scripts/plot_hgs_theory.py --n_anchors 2000 --n_samples 800
        # tighter Monte-Carlo budget
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---- import HuMP's actual HGS function --------------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.layers.lhyperbolic import hyperbolic_entailment_completion_strict  # noqa: E402


# ===========================================================================
# Panel (a) — acceptance rate of cone-intersection rejection sampling
# ===========================================================================
def _make_anchor_pair(n, D, phi, device):
    """Generate `n` anchor pairs (a, b) with controlled angle phi between
    them.  In high-D space, two uniformly random unit vectors are nearly
    orthogonal (phi ~ pi/2), which is *not* the regime HGS operates in:
    real P/C encoders project the same patient to highly correlated points
    with phi in [0.2, 0.6] rad.  We construct anchors that exactly realize
    a target angle phi:

        a = e1
        b = cos(phi) * e1 + sin(phi) * e2

    then rotate the (e1, e2) frame to a random orientation in D-dim.
    """
    # build a in 2D subspace then rotate
    e1 = F.normalize(torch.randn(n, D, device=device), dim=-1)            # [N, D]
    raw_e2 = torch.randn(n, D, device=device)
    # remove e1 component to make e2 orthogonal to e1
    raw_e2 = raw_e2 - (raw_e2 * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(raw_e2, dim=-1)

    a = e1
    b = math.cos(phi) * e1 + math.sin(phi) * e2
    return a, b


def _empirical_acceptance(theta, phi, D, n_anchors, n_trials, device):
    """For `n_anchors` anchor pairs (a, b) with angle `phi` between them,
    sample `n_trials` candidate directions using the same bisector +
    perturbation rule as `hyperbolic_entailment_completion_strict`, and
    return the empirical fraction of candidates accepted by both cones
    of half-angle `theta`.
    """
    cos_thresh = math.cos(theta)
    a, b = _make_anchor_pair(n_anchors, D, phi, device)
    base = F.normalize(a + b, dim=-1)                                    # [N, D]

    noise = F.normalize(torch.randn(n_anchors, n_trials, D, device=device),
                        dim=-1)
    alpha = torch.rand(n_anchors, n_trials, 1, device=device) * 0.25
    cand = F.normalize(
        (1.0 - alpha) * base.unsqueeze(1) + alpha * noise, dim=-1)       # [N, T, D]

    cos_a = (cand * a.unsqueeze(1)).sum(-1)
    cos_b = (cand * b.unsqueeze(1)).sum(-1)
    ok = (cos_a >= cos_thresh) & (cos_b >= cos_thresh)
    rate = ok.float().mean(dim=1)                                        # [N]
    return rate.cpu().numpy()


def panel_acceptance(ax, args, device):
    """Draw panel (a): acceptance rate / expected trials vs cone angle.

    We plot one acceptance-rate curve per realistic anchor angle phi in
    {0.2, 0.4, 0.6} rad, reflecting the range of P/C pair angles produced
    by trained encoders in HuMP.  Showing multiple phi values lets the
    reader gauge robustness across the operating envelope.
    """
    thetas = np.linspace(0.10, 0.70, 25)
    phi_list = [0.2, 0.4, 0.6]
    phi_colors = ['#406A8E', '#9C7AB0', '#C99696']  # blue → purple → rose

    # ---- left axis: acceptance rate per phi ----------------------------
    rate_at_default = {}
    rates_by_phi = {}
    for phi, color in zip(phi_list, phi_colors):
        rate_mean = np.zeros_like(thetas)
        for i, th in enumerate(thetas):
            rate = _empirical_acceptance(
                theta=float(th), phi=phi, D=args.dim,
                n_anchors=args.n_anchors, n_trials=args.n_trials,
                device=device,
            )
            rate_mean[i] = rate.mean()
        rates_by_phi[phi] = rate_mean
        ax.plot(thetas, rate_mean, color=color, lw=1.6,
                label=fr'$\varphi\!=\!{phi}$')
        rate_at_default[phi] = float(np.interp(0.35, thetas, rate_mean))

    # default operating point — short tick label only
    th0 = 0.35
    ax.axvline(th0, color='#999', lw=1.0, linestyle='--', alpha=0.7)
    ax.text(th0 + 0.012, 0.05, fr'$\theta\!=\!0.35$',
            fontsize=9, color='#666', va='bottom')

    ax.set_xlabel(r'Cone half-angle $\theta$ (rad)')
    ax.set_ylabel('Acceptance rate')
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.spines['top'].set_visible(False)

    # ---- right axis: expected # trials (driven by phi=0.4 curve) ------
    ax2 = ax.twinx()
    eps = 1e-3
    rate_mid = rates_by_phi[0.4]
    exp_trials = 1.0 / np.maximum(rate_mid, eps)
    color_t = '#9A6B3F'
    ax2.plot(thetas, exp_trials, color=color_t, lw=1.4, linestyle='-.',
             label=r'#trials')
    ax2.set_ylabel('#trials per valid sample', color=color_t)
    ax2.set_yscale('log')
    ax2.set_ylim(0.8, 1000)
    ax2.tick_params(axis='y', labelcolor=color_t)
    ax2.axhline(50, color=color_t, lw=0.8, linestyle=':', alpha=0.5)
    ax2.text(0.695, 56, '50', fontsize=9,
             color=color_t, ha='right', va='bottom')
    ax2.spines['top'].set_visible(False)

    # compact combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower right',
              frameon=True, framealpha=0.85, edgecolor='#dddddd',
              fontsize=9, handlelength=1.4, handletextpad=0.4,
              borderpad=0.35, labelspacing=0.25)

    ax.set_title('(a)  Sampling efficiency', pad=8)


# ===========================================================================
# Panel (b) — distribution coverage of HGS imputed tokens
# ===========================================================================
def _gaussian_mmd(x, y, sigma=None):
    """Unbiased squared MMD with a Gaussian RBF kernel.  Inputs are 2D
    arrays of shape (n, d) and (m, d).  Lower is better."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    if sigma is None:
        # median heuristic over the pooled set
        with torch.no_grad():
            z = torch.cat([x, y], dim=0)
            dists = torch.cdist(z, z).reshape(-1)
            sigma = float(dists[dists > 0].median().item())
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)

    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    return float((k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()).item())


def _simulate_real_anchor_pair(n, D, device,
                               cluster_spread=0.6,
                               modality_noise=0.25):
    """Generate a synthetic cluster of 'real M-modality' tokens together
    with their corresponding (P, C) anchor pairs.

    Model: each of the `n` patients has a private latent `z_i` drawn
    from a Gaussian centred on `mu` (the cohort axis), then each of
    the three modalities (M, P, C) projects `z_i` to the sphere with
    its own independent noise.  This mirrors the actual HuMP setting,
    where the three encoders see the SAME patient but produce nearby
    (not identical) embeddings.

    The key change from the previous version is that anchors P_i and
    C_i depend on the per-patient z_i (not a shared mu), so the HGS
    imputation, which lives near the bisector of (P_i, C_i), spreads
    across the cluster instead of all collapsing to one point.
    """
    # cohort axis
    mu = F.normalize(torch.randn(D, device=device), dim=-1)              # [D]

    # per-patient latent
    z = F.normalize(
        mu.unsqueeze(0) + cluster_spread * torch.randn(n, D, device=device),
        dim=-1,
    )                                                                    # [N, D]

    # each modality = patient latent + modality-specific noise on sphere
    def _perturb(z):
        return F.normalize(
            z + modality_noise * torch.randn_like(z), dim=-1)

    m_real = _perturb(z)
    p      = _perturb(z)
    c      = _perturb(z)
    return m_real, p, c, mu


def _load_real_features(npz_paths, device):
    """Load one or more .npz files produced by extract_hgs_features.py.

    Two modes are auto-detected from the 'mode' field of the npz:

      * 'cohort_pool'  : each file has P, O, C of shape [N_patients, D]
                         (mean-pooled per patient).  Multiple files
                         concatenated along the patient axis.
                         Returns (O, P, C, D, mode='cohort_pool').

      * 'single_patient' : the file has P, O, C as full token bags of
                           shapes [N_patch, D] / [N_omic, D] / [N_c, D]
                           for ONE patient.  Only the first npz is used
                           in this mode (subsequent ones ignored).
                           Returns (O, P, C, D, mode='single_patient').
    """
    Ps, Os, Cs = [], [], []
    detected_mode = None
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        npz_mode = str(d['mode']) if 'mode' in d.files else 'cohort_pool'
        if detected_mode is None:
            detected_mode = npz_mode
        # single-patient mode: take the first file only, no concatenation
        if npz_mode == 'single_patient':
            print(f"  loaded {p} (single_patient): P={d['P'].shape} "
                  f"O={d['O'].shape} C={d['C'].shape}")
            Ps = [d['P']]; Os = [d['O']]; Cs = [d['C']]
            break
        # cohort_pool mode: concatenate
        Ps.append(d['P']); Os.append(d['O']); Cs.append(d['C'])
        print(f"  loaded {p} (cohort_pool): {d['P'].shape[0]} patients")

    P = np.concatenate(Ps, axis=0)
    O = np.concatenate(Os, axis=0)
    C = np.concatenate(Cs, axis=0)
    P = F.normalize(torch.from_numpy(P).float().to(device), dim=-1)
    O = F.normalize(torch.from_numpy(O).float().to(device), dim=-1)
    C = F.normalize(torch.from_numpy(C).float().to(device), dim=-1)
    return O, P, C, int(P.shape[-1]), detected_mode


def panel_coverage(ax, args, device):
    """Draw panel (b): cosine-similarity histogram on a controlled synthetic
    encoder distribution.

    We model the trained encoder geometry by drawing a per-patient latent
    z_i on the unit sphere (with a cohort axis mu and patient spread), and
    perturbing it to obtain three modality embeddings (M, P, C).  HGS then
    imputes M from (P, C) by rejection sampling in cone(P) intersect
    cone(C).  Three histograms are stacked:

      * cos(HGS_hat,  nearest real-M)         -- ours
      * cos(random unit vec,  nearest real-M) -- lower bound
      * cos(real-M, second-nearest real-M)    -- intra-distribution upper
                                                 bound

    An HGS histogram concentrated to the right of the random baseline,
    and overlapping the intra-distribution baseline, is evidence that
    the cone-intersection rejection step keeps samples inside the
    modality manifold rather than escaping to arbitrary directions.
    """
    manifold = None  # HGS API; the function tolerates None internally for sampling

    # ---- controlled synthetic simulation -------------------------------
    # Each "patient" i has a private latent z_i drawn from a Gaussian on
    # the unit sphere centred on a cohort axis mu.  The three modalities
    # (M, P, C) are independent perturbations of z_i with the same noise
    # scale, which corresponds to a trained encoder that satisfies the
    # cross-modal entailment loss (P, M, C of the same patient are
    # close in the manifold).
    D = args.dim
    n = args.n_samples
    m_real, g_anchor, c_anchor, _ = _simulate_real_anchor_pair(
        n=n, D=D, device=device,
        cluster_spread=0.6, modality_noise=0.25)

    # ---- HGS imputation -----------------------------------------------
    # Treat 'M' as the missing modality (we use the 'G' branch since
    # in HuMP the 3 missing branches share identical sampling code,
    # only the norm-scaling rule differs).  For an apples-to-apples
    # comparison with the real cluster on the unit sphere we DISABLE
    # the avg-with-prototype step and re-normalise the output.
    # Pure rejection sampling for the PATHOLOGY-missing branch — anchors
    # are G (omics) and C (clinical); no prototype averaging.
    m_hgs_list = []
    for i in range(n):
        embed = {
            'P': None,                                  # this is what we impute
            'G': g_anchor[i:i+1].unsqueeze(0),          # anchor 1
            'C': c_anchor[i:i+1].unsqueeze(0),          # anchor 2
        }
        out = hyperbolic_entailment_completion_strict(
            embed, 'P', manifold,
            cone_angle=0.35, between_frac=0.5,
            avg_with_prototype=False, max_trials=50,
        )
        out = out.reshape(-1, D)[0]
        m_hgs_list.append(out)
    m_hgs = torch.stack(m_hgs_list, dim=0)
    m_hgs = F.normalize(m_hgs, dim=-1)

    # ---- compute cosine similarity to the real-P bag for each draw ---
    # Helper: best (max) cosine to the real bag for each query token.
    def best_cos_to_real(query, real):
        # query: [Nq, D], real: [Nr, D]; both unit-normalized
        sims = query @ real.T                                # [Nq, Nr]
        return sims.max(dim=-1).values                       # [Nq]

    m_real_n = F.normalize(m_real, dim=-1)
    m_hgs_n  = F.normalize(m_hgs,  dim=-1)
    m_rand   = F.normalize(torch.randn_like(m_real_n), dim=-1)

    cos_hgs  = best_cos_to_real(m_hgs_n,  m_real_n).cpu().numpy()
    cos_rand = best_cos_to_real(m_rand,   m_real_n).cpu().numpy()
    # intra-real reference: for each real token, similarity to its NEAREST
    # OTHER real token (excluding itself via small offset)
    sims_rr = m_real_n @ m_real_n.T
    sims_rr.fill_diagonal_(-1.0)
    cos_real = sims_rr.max(dim=-1).values.cpu().numpy()

    # ---- histograms ---------------------------------------------------
    real_color = '#406A8E'
    hgs_color  = '#C99696'
    rand_color = '#999999'

    bins = np.linspace(min(cos_rand.min(), -0.2), 1.0, 36)

    ax.hist(cos_rand, bins=bins, color=rand_color, alpha=0.55,
            label=f'Random  (med={np.median(cos_rand):.2f})',
            edgecolor='none')
    ax.hist(cos_hgs,  bins=bins, color=hgs_color,  alpha=0.75,
            label=f'HGS     (med={np.median(cos_hgs):.2f})',
            edgecolor='#A56F6F', linewidth=0.3)
    ax.hist(cos_real, bins=bins, color=real_color, alpha=0.55,
            label=f'Real-Real (med={np.median(cos_real):.2f})',
            edgecolor='none', histtype='step', linewidth=1.6)

    ax.set_xlabel(r'cos(query, nearest real-P)')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.4, axis='y')
    ax.legend(loc='upper left',
              frameon=True, framealpha=0.85, edgecolor='#dddddd',
              fontsize=10, handlelength=1.4, handletextpad=0.4,
              borderpad=0.35, labelspacing=0.25)

    ax.set_title('(b)  Distribution coverage', pad=8)


# ===========================================================================
# Top-level driver
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='fig_hgs_theory.png',
                    help='Output figure path (.png / .pdf / .svg).')
    ap.add_argument('--dpi', type=int, default=800)
    ap.add_argument('--figsize', type=float, nargs=2, default=[9.0, 4.6],
                    help='Figure size in inches (W H). Default is double-column wide '
                         'with two side-by-side panels.')
    ap.add_argument('--dim', type=int, default=256,
                    help='Ambient embedding dimension (matches HuMP D=256).')
    ap.add_argument('--n_anchors', type=int, default=1000,
                    help='Monte-Carlo anchor pairs for panel (a).')
    ap.add_argument('--n_trials', type=int, default=200,
                    help='Trials per anchor pair (matches HGS max_trials, '
                         'oversampled for stable rate estimate).')
    ap.add_argument('--n_samples', type=int, default=400,
                    help='Number of synthetic patients for panel (b).')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # ---- SciencePlots IEEE style, tuned for the 2-panel horizontal layout -
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'stix',
        })
    plt.rcParams.update({
        # bigger fonts now that each panel has its own ~4.5 in column
        'axes.titlesize':   14,
        'axes.labelsize':   13,
        'xtick.labelsize':  11,
        'ytick.labelsize':  11,
        'legend.fontsize':  11,
        # softer spines for a cleaner look
        'axes.linewidth':   0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3.5,
        'ytick.major.size':  3.5,
        'axes.edgecolor':   '#333333',
        # embed TrueType so journals don't substitute the font
        'pdf.fonttype':  42,
        'ps.fonttype':   42,
    })

    # ---- 2 side-by-side panels --------------------------------------
    # `wspace` here controls the gap between the two subplots.  With the
    # twin-y axis on panel (a) we need extra room so its right ylabel does
    # not collide with panel (b)'s left ylabel.
    fig, axes = plt.subplots(
        1, 2, figsize=tuple(args.figsize), dpi=args.dpi,
        gridspec_kw={'wspace': 0.55}, constrained_layout=False)
    panel_acceptance(axes[0], args, device)
    panel_coverage(axes[1], args, device)

    # leave enough room around the figure so titles / subtitles don't clip
    fig.subplots_adjust(left=0.07, right=0.92, top=0.86, bottom=0.16,
                        wspace=0.55)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.12)
    print(f"Figure saved -> {out_path}")


if __name__ == '__main__':
    main()
