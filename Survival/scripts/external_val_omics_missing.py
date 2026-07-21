"""
External validation of a trained HuMP (MHSurv) model on an in-house cohort
that has WSI + clinical but NO omics.

Methodology
-----------
The in-house cohort is treated as the single-missing scenario "O missing":
for each patient we provide the observed pathology (P) and clinical (C)
modalities, set omics (O) as missing, and let HuMP's hierarchy-guided
sampling (HGS) impute O inside cone(P) ∩ cone(C) using the training-set
prototype.  The fused tri-modal representation then yields a survival risk,
which is scored against the in-house survival labels.  This simultaneously
(i) validates cross-cohort generalization and (ii) demonstrates that HGS
works on a real omics-missing dataset.

Hard compatibility requirements (MUST hold or results are meaningless)
----------------------------------------------------------------------
  * WSI features:  1024-dim, extracted with the SAME encoder HuMP was
                   trained on (NOT the in-house CONCH-512 features).
  * Clinical:      512-dim CLIP text embeddings (same encoder as HuMP's
                   clinic_embedding/clip/), one .pt per case_id:
                   shape [N_attributes, 512].
  * Checkpoint:    a trained MHSurv state_dict; the running prototype
                   model._global_prototype should be present (or will be
                   left as None -> HGS falls back to its zero-prototype
                   path, which is weaker).

Inputs
------
  --ckpt        trained MHSurv checkpoint (.pt)
  --wsi_dir     dir of per-case WSI h5/pt, 1024-dim patch features
  --clin_dir    dir of per-case clinical CLIP embeddings (.pt, [Nc,512])
  --label_xlsx  in-house clinical_data.xlsx with case_id + os + death
  --time_col / --event_col   survival columns (default os / death)

Output
------
  Prints external C-Index (+ t-AUC at 1/3/5y if requested) and writes a
  per-patient risk CSV.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

# ---- repo import (place this file under HHSurv/scripts/) ----
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model_HGNN import MHSurv          # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')


# ===========================================================================
# loaders
# ===========================================================================
def load_wsi(path, max_patches, feat_dim_expected, device):
    """Load a per-case WSI feature file (.h5 or .pt) -> [1, N, D] on device."""
    if path.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            key = 'features' if 'features' in f else list(f.keys())[0]
            feats = f[key][:]
    else:
        obj = torch.load(path, map_location='cpu')
        feats = obj['features'] if isinstance(obj, dict) and 'features' in obj else obj
        feats = feats.numpy() if torch.is_tensor(feats) else np.asarray(feats)

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 3:
        feats = feats.reshape(-1, feats.shape[-1])
    assert feats.shape[-1] == feat_dim_expected, (
        f"WSI feat dim {feats.shape[-1]} != expected {feat_dim_expected}. "
        f"Re-extract in-house WSI with HuMP's encoder. ({path})")

    if max_patches and feats.shape[0] > max_patches:
        idx = np.random.RandomState(0).choice(feats.shape[0], max_patches, replace=False)
        feats = feats[idx]
    return torch.from_numpy(feats).float().unsqueeze(0).to(device)   # [1, N, D]


def load_clinical(path, clin_dim_expected, device):
    """Load per-case CLIP clinical embedding -> [Nc, 512] on device."""
    obj = torch.load(path, map_location='cpu')
    t = obj if torch.is_tensor(obj) else torch.as_tensor(obj)
    t = t.float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() == 3:
        t = t.squeeze(0)
    assert t.shape[-1] == clin_dim_expected, (
        f"Clinical dim {t.shape[-1]} != expected {clin_dim_expected}. "
        f"Re-encode in-house clinical with CLIP (not BiomedCLIP). ({path})")
    return t.to(device)                                              # [Nc, 512]


def find_case_file(case_id, directory, exts):
    """Fuzzy match case_id*.{ext} in directory; return first match or None."""
    for ext in exts:
        m = sorted(glob.glob(os.path.join(directory, f'{case_id}*{ext}')))
        if m:
            return m[0]
    return None


# ===========================================================================
# survival metrics
# ===========================================================================
@torch.no_grad()
def forward_skip_pc(model, x_path, report, device):
    """No-imputation baseline: fuse ONLY pathology (P) + clinical (C), omics
    excluded entirely.  Replicates the model's Branch-D fusion path but with
    just two modality streams and a zeroed omics slot in the mm head.

    Mirrors model_HGNN.py forward:
        token_cross = attention_fusion(g, p, c) -> here (p, c) only
        token_cross = feed_forward -> layer_norm
        *_embed = mean over each modality's tokens
        token_mm = cat([gene(=0), path, clinic]) -> mm -> classifier
    """
    p_feat = model.ffpe_fc(x_path).to(device)                  # [1, Np, 256]
    c_feat = model.clinic_fc(report).unsqueeze(0)              # [1, Nc, 256]

    # SAFusion is symmetric self-attention; pass (p, c) as the two streams
    token_cross = model.attention_fusion(p_feat, c_feat, None)  # cat((p,c))
    token_cross = model.feed_forward(token_cross)
    token_cross = model.layer_norm(token_cross)

    np_ = p_feat.shape[1]
    path_embed   = token_cross[:, :np_, :].mean(dim=1)         # [1, 256]
    clinic_embed = token_cross[:, np_:, :].mean(dim=1)         # [1, 256]
    gene_embed   = torch.zeros_like(path_embed)               # omics slot = 0

    token_mm = torch.cat((gene_embed, path_embed, clinic_embed), dim=-1)  # [1, 768]
    fusion = model.mm(token_mm)
    logits = model.classifier(fusion)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits


def risk_from_logits(logits):
    """NLL-survival hazard -> scalar risk (higher = worse)."""
    hazards = torch.sigmoid(logits)
    surv = torch.cumprod(1 - hazards, dim=1)
    risk = -surv.sum(dim=1)                       # [B]
    return risk.detach().cpu().numpy()


def c_index(risk, time, event):
    """Harrell's C-index via sksurv if available, else a simple O(n^2)."""
    try:
        from sksurv.metrics import concordance_index_censored
        ci = concordance_index_censored(event.astype(bool), time, risk)[0]
        return ci
    except Exception:
        n = len(risk); num = den = 0.0
        for i in range(n):
            for j in range(n):
                if time[i] < time[j] and event[i] == 1:
                    den += 1
                    num += (risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])
        return num / den if den > 0 else float('nan')


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--wsi_dir', required=True)
    ap.add_argument('--clin_dir', required=True)
    ap.add_argument('--label_xlsx', required=True)
    ap.add_argument('--case_col', default='case_id')
    ap.add_argument('--time_col', default='os')
    ap.add_argument('--event_col', default='death')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--wsi_dim', type=int, default=1024)
    ap.add_argument('--clin_dim', type=int, default=512)
    ap.add_argument('--max_patches', type=int, default=4096)
    ap.add_argument('--n_genomic', type=int, default=6)
    ap.add_argument('--n_transomic', type=int, default=331)
    ap.add_argument('--mode', choices=['hgs', 'skip'], default='hgs',
                    help="'hgs' = omics missing, imputed by HGS (tri-modal fusion); "
                         "'skip' = omics excluded, fuse WSI+clinical only (bi-modal).")
    ap.add_argument('--out_csv', default='external_val_risks.csv')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # ---- load checkpoint FIRST, infer the omics encoder shapes from it ----
    state = torch.load(args.ckpt, map_location='cpu')
    state = state.get('state_dict', state) if isinstance(state, dict) else state

    def infer_sizes(prefix):
        """Read {prefix}.{idx}.0.0.weight -> input gene count per group,
        ordered by idx, directly from the checkpoint."""
        import re as _re
        sizes = {}
        pat = _re.compile(rf'^{_re.escape(prefix)}\.(\d+)\.0\.0\.weight$')
        for k, v in state.items():
            m = pat.match(k)
            if m:
                sizes[int(m.group(1))] = v.shape[1]   # [256, gene_count]
        return [sizes[i] for i in sorted(sizes)]

    genomic_sizes = infer_sizes('gene_sig_networks')
    transomic_sizes = infer_sizes('trans_sig_networks')
    if not genomic_sizes:
        genomic_sizes = [82, 328, 513, 452, 1536, 452]   # fallback
    if not transomic_sizes:
        transomic_sizes = [100] * args.n_transomic
    print(f"[arch] inferred genomic_sizes={genomic_sizes}")
    print(f"[arch] inferred transomic_sizes: {len(transomic_sizes)} pathways "
          f"(min={min(transomic_sizes)}, max={max(transomic_sizes)})")
    # keep n_transomic consistent with the real architecture for the
    # missing-mode knock-out loop later
    args.n_transomic = len(transomic_sizes)
    args.n_genomic = len(genomic_sizes)

    # ---- build model with the EXACT trained shapes, then load ----
    model = MHSurv(
        genomic_sizes=genomic_sizes,
        transomic_sizes=transomic_sizes,
        n_classes=args.n_classes,
        fusion='concat', model_size='small', graph_type='HGNN',
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"       e.g. missing: {missing[:5]}")
    model.eval()

    proto = getattr(model, '_global_prototype', None)
    print(f"[proto] global prototype present: "
          f"{isinstance(proto, dict) and any(v is not None for v in proto.values())}")

    # ---- labels (accept .csv or .xlsx) ----
    lp = args.label_xlsx
    if lp.lower().endswith('.csv'):
        df = pd.read_csv(lp)
    else:
        df = pd.read_excel(lp)
    df[args.case_col] = df[args.case_col].astype(str).str.strip()
    # drop rows lacking time/event
    df = df.dropna(subset=[args.time_col, args.event_col]).reset_index(drop=True)
    print(f"[label] {len(df)} patients in {lp}; "
          f"event rate = {pd.to_numeric(df[args.event_col], errors='coerce').mean():.3f}")

    # ---- three-way case_id overlap check (labels / WSI / clinical) ----
    wsi_ids = {os.path.basename(p).split('.')[0]
               for p in glob.glob(os.path.join(args.wsi_dir, '*'))}
    clin_ids = {os.path.basename(p).split('.')[0]
                for p in glob.glob(os.path.join(args.clin_dir, '*.pt'))}
    lab_ids = set(df[args.case_col])
    inter = lab_ids & wsi_ids & clin_ids if (wsi_ids and clin_ids) else lab_ids
    print(f"[overlap] labels={len(lab_ids)} wsi={len(wsi_ids)} "
          f"clin={len(clin_ids)} | usable intersection(approx)={len(inter)}")

    # ---- per-patient inference ----
    risks, times, events, used_ids, skipped = [], [], [], [], []
    for _, row in df.iterrows():
        cid = row[args.case_col]
        wsi_path = find_case_file(cid, args.wsi_dir, ['.h5', '.pt'])
        clin_path = find_case_file(cid, args.clin_dir, ['.pt'])
        if wsi_path is None or clin_path is None:
            skipped.append(cid)
            continue
        try:
            x_path = load_wsi(wsi_path, args.max_patches, args.wsi_dim, device)
            report = load_clinical(clin_path, args.clin_dim, device)
        except AssertionError as e:
            print(f"[skip] {cid}: {e}")
            skipped.append(cid)
            continue

        with torch.no_grad():
            if args.mode == 'skip':
                # bi-modal: WSI + clinical only, omics excluded (no imputation)
                logits = forward_skip_pc(model, x_path, report, device)
            else:
                # tri-modal: omics MISSING -> HGS imputes it
                kwargs = {
                    'x_path': x_path, 'report': report, 'protein': None,
                    'valid': True, 'missing_mode': 'O', 'prototype': proto,
                }
                for i in range(1, args.n_genomic + 1):
                    kwargs[f'x_genomic{i}'] = None
                for i in range(1, args.n_transomic + 1):
                    kwargs[f'x_transomic{i}'] = None
                out = model(**kwargs)
                logits = out[0] if isinstance(out, tuple) else out
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
        r = risk_from_logits(logits)[0]
        if not np.isfinite(r):
            skipped.append(cid)
            continue

        try:
            t_val = float(row[args.time_col]); e_val = float(row[args.event_col])
        except (ValueError, TypeError):
            skipped.append(cid); continue
        risks.append(r)
        times.append(t_val)
        events.append(e_val)
        used_ids.append(cid)

    risks = np.array(risks); times = np.array(times); events = np.array(events)
    print(f"\n[run] evaluated {len(risks)} / {len(df)} patients "
          f"(skipped {len(skipped)})")

    if len(risks) < 10:
        raise SystemExit("Too few evaluable patients; check feature dirs / dims.")

    ci = c_index(risks, times, events)
    mode_desc = ('HGS-imputed tri-modal' if args.mode == 'hgs'
                 else 'WSI+clinical bi-modal (no imputation)')
    print("=" * 56)
    print(f"  External C-Index [{mode_desc}]: {ci:.4f}")
    print(f"  N = {len(risks)}, events = {int(events.sum())} "
          f"({events.mean()*100:.1f}%)")
    print("=" * 56)

    pd.DataFrame({
        'case_id': used_ids, 'risk': risks,
        args.time_col: times, args.event_col: events,
    }).to_csv(args.out_csv, index=False)
    print(f"[out] per-patient risks -> {args.out_csv}")
    if skipped:
        print(f"[note] {len(skipped)} skipped (missing files / dim mismatch): "
              f"{skipped[:10]}{' ...' if len(skipped) > 10 else ''}")


if __name__ == '__main__':
    main()
