"""
Zero-/Few-shot cross-institutional data-efficiency on AHM.

Given a pretrained HuMP checkpoint (e.g. TCGA-LIHC- or two-center-pretrained),
for EACH hospital center we:
  1. hold out a fixed test split of that center (seed-controlled),
  2. adapt the checkpoint with an increasing fraction of the center's
     remaining data -- 0% (zero-shot), 10%, 25%, 50%, 100% (few-shot),
  3. evaluate on the SAME held-out test split.

Each (center, ratio) is repeated over several seeds (each seed redraws the
test split + the few-shot subset) and reported as mean +/- std, yielding a
classic data-efficiency curve.  AHM has no omics -> bi-modal P+C path.
Self-contained (no import from finetune_ahm.py / loco_ahm.py).

Usage:
  python scripts/fewshot_ahm.py \
      --wsi_dir ../WSIdata/AHM/h5_files/ --clin_dir ../Table/clip/AHM/ \
      --label_csv ./AHM/AHM_labels.csv --time_col os --event_col death \
      --wsi_dim 512 --pretrained CKPT \
      --ratios 0,0.1,0.25,0.5,1.0 --seeds 0,1,2,3,4 --tag fewshot
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.model_HGNN import MHSurv          # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')


# ===========================================================================
# data loaders  (identical to loco_ahm.py -- kept inline for self-containment)
# ===========================================================================
def find_case_file(cid, directory, exts):
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
    if len(unc) < n_bins + 1:        # too few events -> use all times
        unc = times
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
        model.load_state_dict(state, strict=False)
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
# stratified K-fold + NESTED subset sampling
#   * each fold's test set is fixed -> identical across all ratios
#   * subsets are nested (10% subset of 25% subset of 50% ...) so the
#     data-efficiency curve isolates "more data" from resampling noise
# ===========================================================================
def stratified_folds(samples, k, seed):
    """Return k index-lists, stratified by event (round-robin assignment)."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for ev in (0, 1):
        idx = [i for i, s in enumerate(samples) if int(round(s[4])) == ev]
        rng.shuffle(idx)
        for j, ix in enumerate(idx):
            folds[j % k].append(ix)
    return folds


def nested_subset(pool, ratio, seed):
    """Stratified, NESTED prefix of pool (seed fixed per fold -> nested)."""
    if ratio >= 1.0:
        return list(pool)
    if ratio <= 0.0:
        return []
    rng = np.random.RandomState(seed)
    sub = []
    for ev in (0, 1):
        lst = [s for s in pool if int(round(s[4])) == ev]
        rng.shuffle(lst)                       # same order every call (fixed seed)
        k = max(1, int(round(len(lst) * ratio))) if lst else 0
        sub += lst[:k]
    return sub


# ===========================================================================
# adapt on a (possibly empty) subset, evaluate on fixed test split
# ===========================================================================
def adapt_eval(train_sub, test_s, bins, args, device, seed):
    def load_x(s):
        x = load_wsi(s[1], args.max_patches, args.wsi_dim).unsqueeze(0).to(device)
        r = load_clin(s[2], args.clin_dim).to(device)
        return x, r

    model = build_model(args.n_classes, device, args.pretrained)

    if len(train_sub) > 0:                      # few-shot adaptation
        if args.freeze_encoders:
            for p in model.ffpe_fc.parameters():  p.requires_grad = False
            for p in model.clinic_fc.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.wd)
        rng = np.random.RandomState(seed)
        for ep in range(args.ft_epochs):
            model.train()
            order = list(range(len(train_sub))); rng.shuffle(order)
            for j in order:
                s = train_sub[j]
                x, r = load_x(s)
                y = torch.tensor([disc_label(s[3], bins)], device=device)
                c = torch.tensor([1.0 - s[4]], device=device)
                loss = nll_surv(forward_pc(model, x, r), y, c)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    model.eval()
    rk_, tt_, te_ = [], [], []
    with torch.no_grad():
        for s in test_s:
            x, r = load_x(s)
            rk = risk_score(forward_pc(model, x, r)).item()
            if np.isfinite(rk):
                rk_.append(rk); tt_.append(s[3]); te_.append(s[4])
    return c_index(np.array(rk_), np.array(tt_), np.array(te_))


# ===========================================================================
def plot_curve(records, ratios, centers, out_path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'STIXGeneral',
                                        'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    COL = {'C1': '#52ADAD', 'C2': '#DB5C56', 'C3': '#7E9BC9'}
    xs = np.arange(len(ratios))
    xlab = ['Zero-shot' if r == 0 else f'{int(r*100)}%' for r in ratios]
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=600)
    for ci_, ctr in enumerate(centers):
        mu, sd = [], []
        for r in ratios:
            vals = [rec['ci'] * 100 for rec in records
                    if rec['center'] == ctr and rec['ratio'] == r]
            mu.append(np.mean(vals)); sd.append(np.std(vals))
        mu = np.array(mu); sd = np.array(sd)
        col = COL.get(ctr, None)
        ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.15, zorder=2)
        ax.plot(xs, mu, '-o', color=col, lw=1.8, ms=5, label=ctr, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(xlab)
    ax.set_xlabel('Fraction of target-center data used for adaptation')
    ax.set_ylabel('C-Index (%)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.18, lw=0.5)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi_dir', required=True)
    ap.add_argument('--clin_dir', required=True)
    ap.add_argument('--label_csv', required=True)
    ap.add_argument('--pretrained', required=True,
                    help='checkpoint to adapt from (required for zero/few-shot)')
    ap.add_argument('--case_col', default='case_id')
    ap.add_argument('--center_col', default='center')
    ap.add_argument('--time_col', default='os')
    ap.add_argument('--event_col', default='death')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--wsi_dim', type=int, default=1024)
    ap.add_argument('--clin_dim', type=int, default=512)
    ap.add_argument('--max_patches', type=int, default=4096)
    ap.add_argument('--ratios', default='0,0.1,0.25,0.5,1.0',
                    help='comma list; 0 = zero-shot')
    ap.add_argument('--folds', type=int, default=5,
                    help='stratified K-fold; test set fixed per fold across ratios')
    ap.add_argument('--seed', type=int, default=42,
                    help='fold-shuffle + subset-sampling seed')
    ap.add_argument('--ft_epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--freeze_encoders', action='store_true')
    ap.add_argument('--center', default='all', help='all or C1/C2/C3')
    ap.add_argument('--out_csv', default='fewshot_results.csv')
    ap.add_argument('--plot', default='fewshot_curve.png')
    ap.add_argument('--tag', default='fewshot')
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(',') if x != '']
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
    all_centers = sorted({s[5] for s in samples})
    centers = all_centers if args.center == 'all' else [args.center]
    print(f"[data] usable {len(samples)} patients, centers {all_centers}")
    print(f"[cfg]  ratios={ratios} folds={args.folds} seed={args.seed} "
          f"ft_epochs={args.ft_epochs}")
    print(f"[init] adapt from {args.pretrained}")

    records = []
    for ctr in centers:
        c_samp = [s for s in samples if s[5] == ctr]
        folds = stratified_folds(c_samp, args.folds, args.seed)
        print(f"\n########## center {ctr} (n={len(c_samp)}, "
              f"{args.folds}-fold) ##########")
        for fk in range(args.folds):
            torch.manual_seed(args.seed + fk); np.random.seed(args.seed + fk)
            test = [c_samp[i] for i in folds[fk]]
            pool = [c_samp[i] for f in range(args.folds) if f != fk
                    for i in folds[f]]
            bins = make_bins(np.array([s[3] for s in pool]),
                             np.array([s[4] for s in pool]), args.n_classes)
            sub_seed = args.seed * 100 + fk            # fixed per fold -> nested
            for r in ratios:
                sub = nested_subset(pool, r, sub_seed)
                ci = adapt_eval(sub, test, bins, args, device, args.seed + fk)
                records.append({'center': ctr, 'ratio': r, 'fold': fk,
                                'n_adapt': len(sub), 'n_test': len(test),
                                'ci': ci})
                tag = 'zero-shot' if r == 0 else f'{int(r*100):>3d}%'
                print(f"  {ctr} fold{fk} {tag} "
                      f"(adapt={len(sub):>3d}, test={len(test)}) CI={ci:.4f}")

    pd.DataFrame(records).to_csv(args.out_csv, index=False)
    print(f"\n[csv] {args.out_csv}")

    print("\n" + "=" * 64)
    print(f"  [{args.tag}] zero/few-shot C-Index  (mean +/- std over {args.folds} folds)")
    print("=" * 64)
    for ctr in centers:
        print(f"  {ctr}:")
        for r in ratios:
            vals = [rec['ci'] for rec in records
                    if rec['center'] == ctr and rec['ratio'] == r]
            tag = 'zero-shot' if r == 0 else f'{int(r*100):>3d}%'
            print(f"      {tag:>9s}: {np.mean(vals)*100:5.2f} +/- "
                  f"{np.std(vals)*100:4.2f}   {[round(v*100,1) for v in vals]}")

    try:
        plot_curve(records, ratios, centers, args.plot)
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == '__main__':
    main()
