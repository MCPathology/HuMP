"""
Benchmark MHSurv (models/model_HGNN.py) at different WSI patch counts.

Reports per (N_patch) row:
    Params (M)           — fixed across N (depends only on architecture)
    FLOPs  (G)           — measured via fvcore (preferred) or thop fallback
    Peak Mem (GB)        — torch.cuda.max_memory_allocated() during forward
    Time   (s)           — median of `repeat` timed forwards after `warmup`

Modality shapes (modelled after the actual training pipeline):
    x_path     : [1, N_patch, enc_dim]          # WSI patch bag, enc_dim=1024 default
    x_genomic{i}     : [1, sig_size_i]          # 6 heads, sizes read from
                                                # genomics_signatures.csv (1 head per column)
    x_transomic{j}   : [1, sig_size_j]          # k heads, sizes read from
                                                # transcripts_signatures.csv (1 head per column)
    protein    : [1, 100, 1280]                 # 100 protein tokens, 1280 dim each
    report     : [15, 512]                      # 15 clinical tokens, 512 dim each
                                                # (clinic_fc does .unsqueeze(0) -> [1,15,256])

Example:
    python scripts/bench_mhsurv.py \
        --n_patch_list 256 512 1024 2048 4096 8192 \
        --warmup 3 --repeat 10 \
        --csv_out bench_mhsurv.csv

The signature CSV defaults are resolved relative to the repository root
(parent of `scripts/`).  Override with `--genomic_csv` / `--transomic_csv`.
Disable with `--no_protein` or `--no_transomic`.
"""

import argparse
import gc
import os
import statistics
import sys
import time
import warnings
from contextlib import contextmanager

import torch

# ---- make 'models' importable when running from repo root or /scripts -----
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname('/mnt/pfs-mc0p4k/cvg/team/didonglin/yangguang/resource/')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model_HGNN import MHSurv  # noqa: E402

warnings.filterwarnings("ignore")


# ===========================================================================
# Signature-CSV reader — counts non-empty cells per column ↦ input dim
# ===========================================================================
def read_signature_sizes(csv_path):
    """Read a HuMP signature CSV and return the per-column input dimensions.

    The CSV format is:
        row 0   : column headers (1 signature per column)
        rows >0 : gene names; blank cells are 'no more genes for this column'

    Returns a list `[n_genes_col0, n_genes_col1, ...]` where each entry is
    the number of non-empty gene cells in that column.
    """
    import csv

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"signature CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"signature CSV {csv_path} has < 2 rows")

    header = rows[0]
    n_cols = len(header)
    sizes = [0] * n_cols
    for r in rows[1:]:
        for c in range(min(n_cols, len(r))):
            cell = (r[c] or "").strip()
            if cell:
                sizes[c] += 1
    return sizes


# ===========================================================================
# Input builders — one tensor per modality, with controllable shapes
# ===========================================================================
def _t(shape, device, dtype=torch.float32):
    """Allocate a random tensor on `device`."""
    return torch.randn(*shape, device=device, dtype=dtype)


def build_inputs(n_patch, enc_dim, genomic_sizes, transomic_sizes,
                 protein_shape, report_shape,
                 use_protein, use_transomic, device):
    """Build the kwargs dict MHSurv.forward expects.

    Shapes are chosen to match what `_collate_MCAT` (utils/general_utils.py)
    actually delivers to the model in real training:
        x_path        : [1, N_patch, enc_dim]      stack of 1 WSI bag
        x_genomic{i}  : [size_i]                   1D — collate does torch.cat(..., dim=0)
                                                    over a list of 1D pandas-Series tensors
        x_transomic{j}: [size_j]                   1D — same as genomic
        protein       : protein_shape              3D, e.g. [1, 100, 1280] (collate stacks once)
        report        : report_shape               2D, e.g. [15, 512]
                                                    clinic_fc + .unsqueeze(0) → [1, 15, 256]

    After SNN_Block + stack inside the model:
        SNN_Block([size_i])    → [256]
        torch.stack([6× 256])  → [6, 256]
        .unsqueeze(0)          → [1, 6, 256]      ✓  matches transomics / clinic / protein on dim -2 (256-dim cat)
    """
    kwargs = {
        "x_path": _t((1, n_patch, enc_dim), device),
        "report": _t(report_shape, device),
        "valid": False,
        "missing_mode": "",
        "skip_imputation": False,
    }

    # genomic SNN heads (always 6 in MHSurv).  1D per real collate.
    assert len(genomic_sizes) == 6, \
        f"MHSurv expects exactly 6 genomic groups, got {len(genomic_sizes)}"
    for i, sz in enumerate(genomic_sizes, start=1):
        kwargs[f"x_genomic{i}"] = _t((sz,), device)

    # transomic SNN heads (count matches num_pathways).  1D per real collate.
    if use_transomic:
        for i, sz in enumerate(transomic_sizes, start=1):
            kwargs[f"x_transomic{i}"] = _t((sz,), device)
    # if use_transomic=False, the model was constructed with transomic_sizes=[]
    # so the full-modality branch will not index into x_transomic*.

    kwargs["protein"] = _t(protein_shape, device) if use_protein else None
    return kwargs


# ===========================================================================
# Metric helpers
# ===========================================================================
def count_params(model):
    """Total trainable parameter count in millions."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


def measure_flops(model, kwargs):
    """Return forward FLOPs in GFLOPs (best-effort).

    Order of preference (each falls back to next on failure):
        1. torch.utils.flop_counter.FlopCounterMode  — built-in PyTorch >= 2.0,
                                                       no extra deps, accurate for
                                                       matmul / addmm / conv / einsum
        2. fvcore.nn.FlopCountAnalysis               — needs `pip install fvcore`
        3. thop.profile                              — needs `pip install thop`,
                                                       MACs-based, coarser
        4. NaN
    """
    # ---- 1) torch built-in FlopCounterMode (PyTorch >= 2.0, no install) ----
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                _ = model(**kwargs)

        # Three API shapes across torch versions; try in order.
        total = 0
        # (a) torch >= 2.2: explicit getter
        if hasattr(fc, "get_total_flops"):
            try:
                total = int(fc.get_total_flops())
            except Exception:
                total = 0
        # (b) torch 2.0 / 2.1: sum the per-op dict under "Global"
        if total == 0 and hasattr(fc, "flop_counts"):
            counts = fc.flop_counts.get("Global", {})
            if isinstance(counts, dict):
                total = int(sum(counts.values()))
        # (c) some forks: top-level `_total_flops`
        if total == 0 and hasattr(fc, "_total_flops"):
            total = int(getattr(fc, "_total_flops"))

        if total > 0:
            return total / 1e9
        print("  [torch.flop_counter returned 0 — model may use only "
              "unsupported ops; trying fvcore ...]")
    except Exception as e:
        print(f"  [torch.flop_counter unavailable: {type(e).__name__}: {e}]  trying fvcore ...")

    # ---- 2) fvcore ----
    try:
        from fvcore.nn import FlopCountAnalysis

        # fvcore traces with positional args; wrap forward to accept kwargs.
        class _Wrap(torch.nn.Module):
            def __init__(self, m, kw):
                super().__init__()
                self.m = m
                self._kw = kw

            def forward(self, dummy):
                return self.m(**self._kw)

        wrap = _Wrap(model, kwargs).to(next(model.parameters()).device).eval()
        dummy = torch.zeros(1, device=next(model.parameters()).device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flops = FlopCountAnalysis(wrap, dummy)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total = flops.total()
        return total / 1e9
    except Exception as e:
        print(f"  [fvcore unavailable: {type(e).__name__}: {e}]  trying thop ...")

    # ---- thop fallback ----
    try:
        from thop import profile

        # thop also expects positional args; same wrap trick.
        class _Wrap(torch.nn.Module):
            def __init__(self, m, kw):
                super().__init__()
                self.m = m
                self._kw = kw

            def forward(self, dummy):
                return self.m(**self._kw)

        wrap = _Wrap(model, kwargs).to(next(model.parameters()).device).eval()
        dummy = torch.zeros(1, device=next(model.parameters()).device)
        macs, _ = profile(wrap, inputs=(dummy,), verbose=False)
        return (2 * macs) / 1e9  # MACs → FLOPs ≈ 2× MACs
    except Exception as e:
        print(f"  [thop unavailable: {type(e).__name__}: {e}]  FLOPs = NaN")
        return float("nan")


@contextmanager
def cuda_mem_scope(device):
    """Resets and records peak CUDA memory in the `with` block."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    yield
    torch.cuda.synchronize(device)


def measure_time_and_mem(model, kwargs, device, warmup=3, repeat=10):
    """Return median forward time (s) and peak memory (GB) across `repeat` runs."""
    model.eval()
    times = []

    # ---- warmup ----
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**kwargs)
    torch.cuda.synchronize(device)

    # ---- reset peak mem after warmup so we capture *one* steady forward ----
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(repeat):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**kwargs)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return statistics.median(times), peak_bytes / (1024 ** 3)


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Benchmark MHSurv at different WSI patch counts.")
    p.add_argument("--n_patch_list", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 8192],
                   help="WSI patch counts to sweep (one row per value).")
    p.add_argument("--enc_dim", type=int, default=1024,
                   help="WSI patch feature dimension.")

    # ---- signature-CSV-driven sizes (default: read from repo root) -------
    repo_default_g = os.path.join(ROOT, "genomics_signatures.csv")
    repo_default_t = os.path.join(ROOT, "transcripts_signatures.csv")
    p.add_argument("--genomic_csv", type=str, default=repo_default_g,
                   help="CSV with 6 columns of gene names. Column-wise non-empty "
                        "count is used as the SNN head input dim. "
                        f"Default: {repo_default_g}")
    p.add_argument("--transomic_csv", type=str, default=repo_default_t,
                   help="CSV with k columns of gene names (one column per pathway). "
                        f"Default: {repo_default_t}")
    p.add_argument("--genomic_sizes", type=int, nargs="+", default=None,
                   help="Override genomic SNN head input sizes (6 ints). "
                        "If unset, sizes are computed from --genomic_csv.")
    p.add_argument("--transomic_sizes", type=int, nargs="*", default=None,
                   help="Override transomic SNN head input sizes (k ints). "
                        "If unset, sizes are computed from --transomic_csv.")
    p.add_argument("--no_transomic", action="store_true",
                   help="Disable transomic modality (num_pathways = 0).")

    # ---- protein / report shapes -----------------------------------------
    p.add_argument("--protein_shape", type=int, nargs="+", default=[1, 100, 1280],
                   help="Protein input tensor shape, e.g. '1 100 1280' for [1,100,1280].")
    p.add_argument("--no_protein", action="store_true",
                   help="Disable protein modality.")
    p.add_argument("--report_shape", type=int, nargs="+", default=[15, 512],
                   help="Report (clinical) input tensor shape, e.g. '15 512' for [15,512]. "
                        "clinic_fc + .unsqueeze(0) yields [1, 15, 256] tokens.")

    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv_out", type=str, default=None,
                   help="If set, also write rows to this CSV.")
    p.add_argument("--skip_flops", action="store_true",
                   help="Skip FLOPs measurement (fvcore/thop can be slow on big bags).")
    args = p.parse_args()

    # ---- resolve genomic_sizes ----
    if args.genomic_sizes is None:
        args.genomic_sizes = read_signature_sizes(args.genomic_csv)
        print(f"[csv] genomic sizes from {args.genomic_csv} -> {args.genomic_sizes}")
    if len(args.genomic_sizes) != 6:
        raise ValueError(
            f"genomic CSV must have exactly 6 columns; got {len(args.genomic_sizes)} "
            f"from {args.genomic_csv}")

    # ---- resolve transomic_sizes ----
    if args.no_transomic:
        args.transomic_sizes = []
    elif args.transomic_sizes is None:
        args.transomic_sizes = read_signature_sizes(args.transomic_csv)
        print(f"[csv] transomic sizes from {args.transomic_csv} "
              f"-> {len(args.transomic_sizes)} pathways "
              f"(min={min(args.transomic_sizes)}, max={max(args.transomic_sizes)})")
    use_transomic = len(args.transomic_sizes) > 0
    use_protein = not args.no_protein

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This script requires CUDA (peak-mem stats need it).")

    # ---- build model ONCE; weights are reused across all N_patch settings ----
    model = MHSurv(
        genomic_sizes=args.genomic_sizes,
        transomic_sizes=args.transomic_sizes,
        n_classes=args.n_classes,
        fusion="concat",
        model_size="small",
        graph_type="HGNN",
    ).to(device).eval()

    n_params_M = count_params(model)

    # ---- header ----
    transomic_brief = (f"{len(args.transomic_sizes)} pathways "
                       f"[{min(args.transomic_sizes)}..{max(args.transomic_sizes)}]"
                       if use_transomic else "disabled")
    print("=" * 88)
    print(f"MHSurv benchmark")
    print(f"  enc_dim={args.enc_dim}")
    print(f"  genomic_sizes  = {args.genomic_sizes}   (6 SNN heads)")
    print(f"  transomic      = {transomic_brief}")
    print(f"  protein        = {tuple(args.protein_shape) if use_protein else 'disabled'}")
    print(f"  report         = {tuple(args.report_shape)}")
    print(f"  n_classes={args.n_classes}   device={device}   "
          f"warmup={args.warmup}   repeat={args.repeat}")
    print(f"  Params(M) = {n_params_M:.3f}    (constant across N_patch)")
    print("=" * 88)
    hdr = f"{'N_patch':>8} | {'Params(M)':>10} | {'FLOPs(G)':>10} | " \
          f"{'PeakMem(GB)':>12} | {'Time(s)':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n_patch in args.n_patch_list:
        # clear any allocator caching from previous sweep step
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        kwargs = build_inputs(
            n_patch=n_patch,
            enc_dim=args.enc_dim,
            genomic_sizes=args.genomic_sizes,
            transomic_sizes=args.transomic_sizes,
            protein_shape=tuple(args.protein_shape),
            report_shape=tuple(args.report_shape),
            use_protein=use_protein,
            use_transomic=use_transomic,
            device=device,
        )

        # ---- FLOPs (optional) ----
        flops_G = float("nan") if args.skip_flops else measure_flops(model, kwargs)

        # ---- Time + Peak memory ----
        t_med, mem_GB = measure_time_and_mem(
            model, kwargs, device,
            warmup=args.warmup, repeat=args.repeat,
        )

        row = (n_patch, n_params_M, flops_G, mem_GB, t_med)
        rows.append(row)
        print(f"{n_patch:>8d} | {n_params_M:>10.3f} | "
              f"{flops_G:>10.3f} | {mem_GB:>12.3f} | {t_med:>10.4f}")

    # ---- optional CSV ----
    if args.csv_out is not None:
        with open(args.csv_out, "w") as f:
            f.write("N_patch,Params_M,FLOPs_G,PeakMem_GB,Time_s\n")
            for r in rows:
                f.write(",".join([str(r[0])] + [f"{x:.6f}" for x in r[1:]]) + "\n")
        print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
