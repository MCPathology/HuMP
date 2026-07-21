"""
Leave-One-Center-Out (LOCO) cross-institutional validation on AHM.

For each held-out hospital center, HuMP is trained on the OTHER two centers
and evaluated on the held-out center WITHOUT any adaptation on it -- a
genuine cross-institutional test ("train on one cohort, validate on
another"), directly answering the reviewer's request.

Init:
  --pretrained CKPT : start from a TCGA-LIHC-pretrained HuMP, then train on
                      the two source centers (transfer init).
  (omitted)         : train the two source centers from scratch, then test.

AHM has no omics, so the bi-modal pathology+clinical path is used.
Self-contained: no dependency on finetune_ahm.py.

Usage:
  python scripts/loco_ahm.py \
      --wsi_dir ../WSIdata/AHM/h5_files/ --clin_dir ../Table/clip/AHM/ \
      --label_csv ./AHM/AHM_labels.csv --time_col os --event_col death \
      --wsi_dim 512 [--pretrained CKPT] --tag loco
"""
import argparse
import glob
import os
import re
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
# data loaders
# ===========================================================================
def find_case_file(cid, directory, exts):
    """Fuzzy match robust to leading-zero discrepancies (C3 WSI = 0XXXX)."""
    cid = str(cid).strip()
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
# survival loss / metrics
# ===========================================================================
def make_bins(times, events, n_bins):
    unc = times[events == 1]
    qs = np.quantile(unc, np.linspace(0, 1, n_bins + 1))
    qs[0] = times.min() - 1e-3; qs[-1] = times.max() + 1e-3
    return qs


def disc_label(t, bins):
    return int(np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2))


def nll_surv(logits, y, c, eps=1e-7):
    h = torch.sigmoid(logits)
    S = torch.cumprod(1 - h, dim=1)
    S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
    y = y.long().view(-1, 1); c = c.float().view(-1)
    h_y = h.gather(1, y).clamp(eps, 1 - eps).view(-1)
    S_prev_y = S_prev.gather(1, y).clamp(eps).view(-1)
    S_y = S.gather(1, y).clamp(eps).view(-1)
    return (-(1 - c) * (torch.log(S_prev_y) + torch.log(h_y))
            - c * torch.log(S_y)).mean()


def risk_score(logits):
    h = torch.sigmoid(logits)
    return -torch.cumprod(1 - h, dim=1).sum(dim=1)


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


def logrank_p(risk, time, event):
    risk = np.asarray(risk); time = np.asarray(time); event = np.asarray(event)
    thr = np.median(risk); hi = risk >= thr; lo = ~hi
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
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        plt.rcParams['font.family'] = 'serif'
    risk = np.asarray(risk); time = np.asarray(time); event = np.asarray(event)
    thr = np.median(risk); hi = risk >= thr; lo = ~hi
    from lifelines import KaplanMeierFitter
    kl = KaplanMeierFitter(); kh = KaplanMeierFitter()
    kl.fit(time[lo], event_observed=event[lo])
    kh.fit(time[hi], event_observed=event[hi])
    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=300)
    kl.plot_survival_function(ax=ax, show_censors=True, color='green',
                              censor_styles={'marker': '+', 'ms': 8}, legend=False)
    kh.plot_survival_function(ax=ax, show_censors=True, color='red',
                              censor_styles={'marker': '+', 'ms': 8}, legend=False)
    ax.set_ylim(0, 1.02); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)


# ===========================================================================
# model + forward (bi-modal P+C, omics excluded)
# ===========================================================================
def build_model(n_classes, device, pretrained=None):
    g_sizes = [82, 328, 513, 452, 1536, 452]; t_sizes = [100] * 331
    state = None
    if pretrained:
        state = torch.load(pretrained, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        def infer(prefix):
            d = {}
            pat = re.compile(rf'^{re.escape(prefix)}\.(\d+)\.0\.0\.weight$')
            for k, v in state.items():
                m = pat.match(k)
                if m:
                    d[int(m.group(1))] = v.shape[1]
            return [d[i] for i in sorted(d)]
        gi, ti = infer('gene_sig_networks'), infer('trans_sig_networks')
        if gi: g_sizes = gi
        if ti: t_sizes = ti
    model = MHSurv(genomic_sizes=g_sizes, transomic_sizes=t_sizes,
                   n_classes=n_classes, fusion='concat',
                   model_size='small', graph_type='HGNN').to(device)
    if state is not None:
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"  [pretrained] loaded, missing={len(miss)} unexpected={len(unexp)}")
    return model


def forward_pc(model, x_path, report):
    p = model.ffpe_fc(x_path)
    c = model.clinic_fc(report).unsqueeze(0)
    tc = model.attention_fusion(p, c, None)
    tc = model.feed_forward(tc); tc = model.layer_norm(tc)
    np_ = p.shape[1]
    p_emb = tc[:, :np_, :].mean(1); c_emb = tc[:, np_:, :].mean(1)
    g_emb = torch.zeros_like(p_emb)
    return model.classifier(model.mm(torch.cat((g_emb, p_emb, c_emb), dim=-1)))


# ===========================================================================
# train on source centers, evaluate on held-out
# ===========================================================================
def train_eval(train_s, test_s, args, device, held):
    rng = np.random.RandomState(args.seed)
    tr_times = np.array([s[3] for s in train_s])
    tr_events = np.array([s[4] for s in train_s])
    bins = make_bins(tr_times, tr_events, args.n_classes)   # bins on TRAIN only

    def load_one(s):
        x = load_wsi(s[1], args.max_patches, args.wsi_dim).unsqueeze(0).to(device)
        r = load_clin(s[2], args.clin_dim).to(device)
        y = torch.tensor([disc_label(s[3], bins)], device=device)
        c = torch.tensor([1.0 - s[4]], device=device)
        return x, r, y, c

    model = build_model(args.n_classes, device, args.pretrained)
    if args.freeze_encoders:
        for p in model.ffpe_fc.parameters():  p.requires_grad = False
        for p in model.clinic_fc.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd)

    risks = times = events = np.array([])
    last_ci = float('nan')
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(train_s))); rng.shuffle(order)
        tot = 0.0
        for j in order:
            x, r, y, c = load_one(train_s[j])
            loss = nll_surv(forward_pc(model, x, r), y, c)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        model.eval()
        tr_, tt_, te_ = [], [], []
        with torch.no_grad():
            for s in test_s:
                x, r, _, _ = load_one(s)
                rk = risk_score(forward_pc(model, x, r)).item()
                if np.isfinite(rk):
                    tr_.append(rk); tt_.append(s[3]); te_.append(s[4])
        risks, times, events = np.array(tr_), np.array(tt_), np.array(te_)
        last_ci = c_index(risks, times, events)
        print(f"  [{held}] ep{ep:02d} loss={tot/max(len(train_s),1):.4f} "
              f"test_CI={last_ci:.4f}")
    return last_ci, risks, times, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi_dir', required=True)
    ap.add_argument('--clin_dir', required=True)
    ap.add_argument('--label_csv', required=True)
    ap.add_argument('--pretrained', default=None)
    ap.add_argument('--case_col', default='case_id')
    ap.add_argument('--center_col', default='center')
    ap.add_argument('--time_col', default='os')
    ap.add_argument('--event_col', default='death')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--wsi_dim', type=int, default=1024)
    ap.add_argument('--clin_dim', type=int, default=512)
    ap.add_argument('--max_patches', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--freeze_encoders', action='store_true')
    ap.add_argument('--km_dir', default='loco_km')
    ap.add_argument('--tag', default='loco')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.label_csv)
    df[args.case_col] = df[args.case_col].astype(str).str.strip()
    df = df.dropna(subset=[args.time_col, args.event_col]).reset_index(drop=True)
    if args.center_col not in df.columns:
        raise SystemExit(f"label_csv has no '{args.center_col}' column.")
    samples = []
    for _, r in df.iterrows():
        cid = r[args.case_col]
        wp = find_case_file(cid, args.wsi_dir, ['.h5', '.pt'])
        cp = find_case_file(cid, args.clin_dir, ['.pt'])
        if wp and cp:
            samples.append((cid, wp, cp, float(r[args.time_col]),
                            float(r[args.event_col]), str(r[args.center_col])))
    centers = sorted({s[5] for s in samples})
    print(f"[data] usable {len(samples)} patients, centers {centers}")
    print(f"[init] {'fine-tune from '+args.pretrained if args.pretrained else 'from scratch'}")

    os.makedirs(args.km_dir, exist_ok=True)
    summary = {}
    for held in centers:
        train_s = [s for s in samples if s[5] != held]
        test_s  = [s for s in samples if s[5] == held]
        print(f"\n########## LOCO: hold out {held} "
              f"(train {len(train_s)}, test {len(test_s)}) ##########")
        ci, risks, times, events = train_eval(train_s, test_s, args, device, held)
        pv = logrank_p(risks, times, events)
        if len(risks) >= 10 and np.isfinite(pv):
            km_path = os.path.join(args.km_dir, f'KM_{args.tag}_holdout_{held}.png')
            plot_km_bare(risks, times, events, km_path)
            pd.DataFrame({'risk': risks, 'time': times, 'event': events}).to_csv(
                os.path.join(args.km_dir, f'pred_{args.tag}_holdout_{held}.csv'),
                index=False)
            print(f"  [KM] held-out {held} -> {km_path}  (log-rank p={pv:.2e})")
        summary[held] = (ci, pv, len(test_s))

    print("\n" + "=" * 60)
    print(f"  [{args.tag}] LOCO cross-institutional C-Index")
    print("=" * 60)
    for held, (ci, pv, n) in summary.items():
        print(f"  hold-out {held} (n={n}): C-Index {ci:.4f}  log-rank p={pv:.2e}")
    if summary:
        print(f"  ---- mean over held-out centers: "
              f"{np.mean([v[0] for v in summary.values()]):.4f} ----")


if __name__ == '__main__':
    main()
