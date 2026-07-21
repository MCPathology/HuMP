"""
Fine-tune / train HuMP on the omics-free external AHM cohort (WSI + clinical),
with 5-fold cross-validation.  Reports two settings in one run via --pretrained:

    (1) from-scratch    : random init, 5-fold CV on AHM            (baseline)
    (2) fine-tune       : load LIHC-pretrained HuMP, then 5-fold   (transfer)

Because AHM has no omics, the model uses the bi-modal pathology+clinical
fusion path (the omics slot is zeroed in the mm head; omics encoders are
unused).  This is the same fusion sub-graph as HuMP's full forward, so a
LIHC-pretrained checkpoint transfers its ffpe_fc / clinic_fc / attention_fusion
/ mm / classifier weights directly.

Inputs
------
  --wsi_dir     per-case WSI features, 1024-dim (HuMP's encoder), .h5/.pt
  --clin_dir    per-case clinical CLIP embeddings (.pt, [Nc, 512])
  --label_csv   AHM_labels.csv  (case_id, os, death, ...)
  --pretrained  (optional) LIHC-pretrained MHSurv checkpoint .pt
  --time_col / --event_col   default os / death

Usage
-----
  # baseline: from scratch
  python scripts/finetune_ahm.py --wsi_dir ... --clin_dir ... \
      --label_csv AHM_labels.csv --tag scratch

  # transfer: fine-tune from LIHC checkpoint
  python scripts/finetune_ahm.py --wsi_dir ... --clin_dir ... \
      --label_csv AHM_labels.csv --pretrained results_lihc/s_0_checkpoint.pt \
      --tag finetune
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.model_HGNN import MHSurv          # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')


# ===========================================================================
# data
# ===========================================================================
def find_case_file(cid, directory, exts):
    """Fuzzy match, robust to leading-zero discrepancies (e.g. C3 WSI files
    are named 0XXXX while labels store XXXX)."""
    cid = str(cid).strip()
    # try: exact, zero-padded (WSI has leading 0s), zero-stripped (labels do)
    cands = [cid, '0' + cid, '00' + cid, cid.lstrip('0')]
    seen = set(); cands = [c for c in cands if c and not (c in seen or seen.add(c))]
    for c in cands:
        for e in exts:
            m = sorted(glob.glob(os.path.join(directory, f'{c}*{e}')))
            if m:
                return m[0]
    return None


def load_wsi(path, max_patches, dim):
    if path.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            key = 'features' if 'features' in f else list(f.keys())[0]
            feats = f[key][:]
    else:
        o = torch.load(path, map_location='cpu')
        feats = o['features'] if isinstance(o, dict) and 'features' in o else o
        feats = feats.numpy() if torch.is_tensor(feats) else np.asarray(feats)
    feats = np.asarray(feats, np.float32)
    if feats.ndim == 3:
        feats = feats.reshape(-1, feats.shape[-1])
    assert feats.shape[-1] == dim, f"WSI dim {feats.shape[-1]}!={dim} ({path})"
    if max_patches and feats.shape[0] > max_patches:
        idx = np.random.RandomState(0).choice(feats.shape[0], max_patches, False)
        feats = feats[idx]
    return torch.from_numpy(feats).float()


def load_clin(path, dim):
    o = torch.load(path, map_location='cpu')
    t = (o if torch.is_tensor(o) else torch.as_tensor(o)).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() == 3:
        t = t.squeeze(0)
    assert t.shape[-1] == dim, f"clin dim {t.shape[-1]}!={dim} ({path})"
    return t


# ===========================================================================
# survival discretization + NLL loss (matches HuMP's n-bin hazard)
# ===========================================================================
def make_bins(times, events, n_bins):
    """Quantile bin edges on uncensored times."""
    unc = times[events == 1]
    qs = np.quantile(unc, np.linspace(0, 1, n_bins + 1))
    qs[0] = times.min() - 1e-3
    qs[-1] = times.max() + 1e-3
    return qs


def disc_label(t, bins):
    return int(np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2))


def nll_surv(logits, y, c, alpha=0.0, eps=1e-7):
    """Discrete-time NLL survival loss (Zadeh & Schmid style)."""
    h = torch.sigmoid(logits)                       # hazards [B, n_bins]
    S = torch.cumprod(1 - h, dim=1)                 # survival
    S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
    y = y.long().view(-1, 1)
    c = c.float().view(-1)
    h_y = h.gather(1, y).clamp(eps, 1 - eps).view(-1)
    S_prev_y = S_prev.gather(1, y).clamp(eps).view(-1)
    S_y = S.gather(1, y).clamp(eps).view(-1)
    uncensored = -(1 - c) * (torch.log(S_prev_y) + torch.log(h_y))
    censored = -c * torch.log(S_y)
    return (uncensored + censored).mean()


def risk_score(logits):
    h = torch.sigmoid(logits)
    S = torch.cumprod(1 - h, dim=1)
    return -S.sum(dim=1)                              # higher = worse


def logrank_p(risk, time, event):
    """Median-risk split -> HR/LR; return the log-rank p-value (nan on fail)."""
    risk = np.asarray(risk); time = np.asarray(time); event = np.asarray(event)
    thr = np.median(risk)
    hi = risk >= thr; lo = ~hi
    if hi.sum() < 3 or lo.sum() < 3:
        return float('nan')
    try:
        from lifelines.statistics import logrank_test
        return logrank_test(time[hi], time[lo],
                            event_observed_A=event[hi],
                            event_observed_B=event[lo]).p_value
    except Exception:
        return float('nan')


def plot_km_bare(risk, time, event, out_path):
    """KM curves only, NO text (no title/legend/labels/p-value).
    Median-risk split, red HR / green LR, censor '+' marks. SciencePlots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'

    risk = np.asarray(risk); time = np.asarray(time); event = np.asarray(event)
    thr = np.median(risk)
    hi = risk >= thr; lo = ~hi

    from lifelines import KaplanMeierFitter
    kmf_low = KaplanMeierFitter(); kmf_high = KaplanMeierFitter()
    kmf_low.fit(time[lo],  event_observed=event[lo])
    kmf_high.fit(time[hi], event_observed=event[hi])

    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=300)
    kmf_low.plot_survival_function(ax=ax, show_censors=True, color='green',
                                   censor_styles={'marker': '+', 'ms': 8},
                                   legend=False)
    kmf_high.plot_survival_function(ax=ax, show_censors=True, color='red',
                                    censor_styles={'marker': '+', 'ms': 8},
                                    legend=False)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def c_index(risk, time, event):
    try:
        from sksurv.metrics import concordance_index_censored
        return concordance_index_censored(event.astype(bool), time, risk)[0]
    except Exception:
        n = len(risk); num = den = 0.0
        for i in range(n):
            for j in range(n):
                if time[i] < time[j] and event[i] == 1:
                    den += 1; num += (risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])
        return num / den if den else float('nan')


# ===========================================================================
# bi-modal forward (P + C), omics absent.  Reuses model submodules.
# ===========================================================================
def forward_pc(model, x_path, report):
    p = model.ffpe_fc(x_path)                          # [1, Np, 256]
    c = model.clinic_fc(report).unsqueeze(0)           # [1, Nc, 256]
    tc = model.attention_fusion(p, c, None)
    tc = model.feed_forward(tc)
    tc = model.layer_norm(tc)
    np_ = p.shape[1]
    p_emb = tc[:, :np_, :].mean(1)
    c_emb = tc[:, np_:, :].mean(1)
    g_emb = torch.zeros_like(p_emb)
    mm_in = torch.cat((g_emb, p_emb, c_emb), dim=-1)
    return model.classifier(model.mm(mm_in))           # [1, n_bins]


# ===========================================================================
# build model (+ optional pretrained) with arch inferred from checkpoint
# ===========================================================================
def build_model(n_classes, device, pretrained=None):
    g_sizes = [82, 328, 513, 452, 1536, 452]
    t_sizes = [100] * 331
    state = None
    if pretrained:
        state = torch.load(pretrained, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        import re
        def infer(prefix):
            d = {}
            pat = re.compile(rf'^{re.escape(prefix)}\.(\d+)\.0\.0\.weight$')
            for k, v in state.items():
                m = pat.match(k)
                if m:
                    d[int(m.group(1))] = v.shape[1]
            return [d[i] for i in sorted(d)]
        gi, ti = infer('gene_sig_networks'), infer('trans_sig_networks')
        if gi:
            g_sizes = gi
        if ti:
            t_sizes = ti
    model = MHSurv(genomic_sizes=g_sizes, transomic_sizes=t_sizes,
                   n_classes=n_classes, fusion='concat',
                   model_size='small', graph_type='HGNN').to(device)
    if state is not None:
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"  [pretrained] loaded, missing={len(miss)} unexpected={len(unexp)}")
    return model


# ===========================================================================
# main: 5-fold CV
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi_dir', required=True)
    ap.add_argument('--clin_dir', required=True)
    ap.add_argument('--label_csv', required=True)
    ap.add_argument('--pretrained', default=None)
    ap.add_argument('--case_col', default='case_id')
    ap.add_argument('--time_col', default='os')
    ap.add_argument('--event_col', default='death')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--wsi_dim', type=int, default=1024)
    ap.add_argument('--clin_dim', type=int, default=512)
    ap.add_argument('--max_patches', type=int, default=4096)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--center_col', default='center')
    ap.add_argument('--center', default='all',
                    help="'all' = run each center separately; or 'C1'/'C2'/'C3'.")
    ap.add_argument('--km_dir', default='km_curves',
                    help='Directory to save per-center KM curves + risk CSVs.')
    ap.add_argument('--km_fold', type=int, default=0,
                    help='(unused now) kept for compatibility.')
    ap.add_argument('--km_p', type=float, default=0.05,
                    help='Save a text-free KM only when the per-epoch log-rank '
                         'p-value is below this threshold (default 0.05).')
    ap.add_argument('--freeze_encoders', action='store_true',
                    help='Freeze ffpe_fc + clinic_fc, only train fusion+mm+classifier '
                         '(recommended for small centers to avoid overfitting).')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- assemble usable patients (label + WSI + clinical all present) ----
    df = pd.read_csv(args.label_csv)
    df[args.case_col] = df[args.case_col].astype(str).str.strip()
    df = df.dropna(subset=[args.time_col, args.event_col]).reset_index(drop=True)
    has_center = args.center_col in df.columns
    samples = []
    from collections import Counter as _C
    tot_by_c, ok_by_c, miss_wsi, miss_clin = _C(), _C(), _C(), _C()
    for _, r in df.iterrows():
        cid = r[args.case_col]
        ctr = str(r[args.center_col]) if has_center else 'ALL'
        tot_by_c[ctr] += 1
        wp = find_case_file(cid, args.wsi_dir, ['.h5', '.pt'])
        cp = find_case_file(cid, args.clin_dir, ['.pt'])
        if wp and cp:
            ok_by_c[ctr] += 1
            samples.append((cid, wp, cp,
                            float(r[args.time_col]), float(r[args.event_col]), ctr))
        else:
            if wp is None: miss_wsi[ctr] += 1
            if cp is None: miss_clin[ctr] += 1
    print(f"[data] usable patients: {len(samples)} / {len(df)}")
    for c in sorted(tot_by_c):
        print(f"  [{c}] matched {ok_by_c[c]}/{tot_by_c[c]} "
              f"(missing WSI={miss_wsi[c]}, clinical={miss_clin[c]})")

    # which centers to run
    if not has_center or args.center == 'pooled':
        center_list = ['ALL']
    elif args.center == 'all':
        center_list = sorted({s[5] for s in samples})
    else:
        center_list = [args.center]

    os.makedirs(args.km_dir, exist_ok=True)
    summary = {}
    for ctr in center_list:
        sub = samples if ctr == 'ALL' else [s for s in samples if s[5] == ctr]
        ev = np.array([s[4] for s in sub])
        print(f"\n########## CENTER {ctr}: n={len(sub)} "
              f"event_rate={ev.mean():.3f} ##########")
        if len(sub) < args.folds * 4:
            print(f"  [skip] too few patients for {args.folds}-fold")
            continue
        # KM for the designated fold is plotted inside run_5fold, the moment
        # that fold finishes (no waiting for the remaining folds).
        ci_mean, ci_std, per_fold, per_fold_preds = run_5fold(sub, args, device, ctr)
        summary[ctr] = (ci_mean, ci_std, per_fold, len(sub), float('nan'))

    print("\n" + "=" * 60)
    print(f"  [{args.tag}] per-center 5-fold C-Index summary")
    print("=" * 60)
    for ctr, (m, s, pf, n, p) in summary.items():
        print(f"  {ctr} (n={n}): C-Idx {m:.4f} +/- {s:.4f}  "
              f"log-rank p={p:.2e}  folds={[round(x,3) for x in pf]}")
    if summary:
        allm = np.mean([v[0] for v in summary.values()])
        print(f"  ---- mean C-Index over centers: {allm:.4f} ----")


def run_5fold(samples, args, device, ctr='ALL'):
    """Patient-level 5-fold CV (event-stratified) on one cohort/center.
    The KM curve for fold == args.km_fold is plotted IMMEDIATELY after that
    fold finishes (no need to wait for the remaining folds)."""
    rng = np.random.RandomState(args.seed)
    times = np.array([s[3] for s in samples])
    events = np.array([s[4] for s in samples])
    idx_all = np.arange(len(samples))
    folds = [[] for _ in range(args.folds)]
    for grp in (idx_all[events == 1], idx_all[events == 0]):
        g = grp.copy(); rng.shuffle(g)
        for k, i in enumerate(g):
            folds[k % args.folds].append(i)
    bins = make_bins(times, events, args.n_classes)

    def load_one(s):
        x = load_wsi(s[1], args.max_patches, args.wsi_dim).unsqueeze(0).to(device)
        r = load_clin(s[2], args.clin_dim).to(device)
        y = torch.tensor([disc_label(s[3], bins)], device=device)
        c = torch.tensor([1.0 - s[4]], device=device)
        return x, r, y, c

    fold_ci = []
    per_fold_preds = []   # [(risk, time, event)] per fold, last-epoch val set
    for fk in range(args.folds):
        val_idx = set(folds[fk])
        tr = [samples[i] for i in idx_all if i not in val_idx]
        va = [samples[i] for i in folds[fk]]
        model = build_model(args.n_classes, device, args.pretrained)
        if args.freeze_encoders:
            for p in model.ffpe_fc.parameters():  p.requires_grad = False
            for p in model.clinic_fc.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.wd)

        best = 0.0
        vr, vt, ve = [], [], []
        for ep in range(args.epochs):
            model.train()
            order = list(range(len(tr))); rng.shuffle(order)
            tot = 0.0
            for j in order:
                x, r, y, c = load_one(tr[j])
                loss = nll_surv(forward_pc(model, x, r), y, c)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tot += loss.item()
            model.eval()
            vr, vt, ve = [], [], []
            with torch.no_grad():
                for s in va:
                    x, r, _, _ = load_one(s)
                    risk = risk_score(forward_pc(model, x, r)).item()
                    if np.isfinite(risk):
                        vr.append(risk); vt.append(s[3]); ve.append(s[4])
            kr = np.array(vr); kt = np.array(vt); ke = np.array(ve)
            ci = c_index(kr, kt, ke)
            best = max(best, ci)
            # ---- per-epoch log-rank; save a text-free KM only if p < 0.05 ----
            pv = logrank_p(kr, kt, ke)
            saved = ''
            if np.isfinite(pv) and pv < args.km_p:
                os.makedirs(args.km_dir, exist_ok=True)
                km_path = os.path.join(
                    args.km_dir,
                    f'KM_{args.tag}_{ctr}_fold{fk}_ep{ep:02d}_p{pv:.1e}.png')
                plot_km_bare(kr, kt, ke, km_path)
                pd.DataFrame({'risk': kr, 'time': kt, 'event': ke}).to_csv(
                    km_path.replace('.png', '.csv'), index=False)
                saved = f'  [KM saved p={pv:.2e}]'
            print(f"  fold{fk} ep{ep:02d} loss={tot/max(len(tr),1):.4f} "
                  f"val_CI={ci:.4f} (best {best:.4f}) p={pv:.2e}{saved}")
        fold_ci.append(best)
        per_fold_preds.append((kr, kt, ke))
        print(f"  >> fold {fk} best C-Index = {best:.4f}")

    return float(np.mean(fold_ci)), float(np.std(fold_ci)), fold_ci, per_fold_preds


if __name__ == '__main__':
    main()
