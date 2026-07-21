"""
Subset the full LIHC RNA matrix to exactly the gene columns (and order) used
by the existing cohorts, so LIHC drops in with an identical feature schema.

Usage:
    python scripts/align_lihc_genes.py \
        --lihc     lihc_rna_clean.csv \
        --ref      datasets_csv/raw_rna_data/combine/brca/rna_clean.csv \
        --out      datasets_csv/raw_rna_data/combine/lihc/rna_clean.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lihc', required=True,
                    help='Full LIHC matrix (samples x all genes).')
    ap.add_argument('--ref', default='datasets_csv/raw_rna_data/combine/brca/rna_clean.csv',
                    help='Reference cohort rna_clean.csv whose gene columns define the schema.')
    ap.add_argument('--out', default='datasets_csv/raw_rna_data/combine/lihc/rna_clean.csv')
    ap.add_argument('--fill_missing', type=float, default=0.0)
    ap.add_argument('--check_all', nargs='*', default=None,
                    help='Optional: extra cohort rna_clean.csv paths to confirm '
                         'they all share the SAME gene set as --ref.')
    args = ap.parse_args()

    # ---- reference gene columns (order matters) ----
    ref_cols = list(pd.read_csv(args.ref, index_col=0, nrows=0).columns)
    print(f"[ref]  {args.ref}: {len(ref_cols)} genes")

    # ---- optional: verify all cohorts share the same gene schema ----
    if args.check_all:
        ref_set = set(ref_cols)
        for p in args.check_all:
            cols = set(pd.read_csv(p, index_col=0, nrows=0).columns)
            same = (cols == ref_set)
            print(f"[chk]  {os.path.basename(os.path.dirname(p))}: "
                  f"{len(cols)} genes | identical_to_ref={same} | "
                  f"diff={len(ref_set ^ cols)}")

    # ---- load full LIHC, subset + reorder to ref columns ----
    print(f"[lihc] loading {args.lihc} (this is large, ~250MB) ...")
    lihc = pd.read_csv(args.lihc, index_col=0)
    print(f"[lihc] {lihc.shape[0]} samples x {lihc.shape[1]} genes (full)")

    missing = [g for g in ref_cols if g not in lihc.columns]
    if missing:
        print(f"[warn] {len(missing)} ref genes absent from LIHC "
              f"(filled with {args.fill_missing}): {missing[:10]}")

    aligned = lihc.reindex(columns=ref_cols)          # subset + reorder
    n_nan = int(aligned.isna().values.sum())
    if n_nan:
        aligned = aligned.fillna(args.fill_missing)
        print(f"[fill] filled {n_nan} NaN cells with {args.fill_missing}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    aligned.to_csv(args.out)
    print(f"[done] wrote {args.out}  shape {aligned.shape} (samples x genes)")
    print(f"[done] columns identical to ref: "
          f"{list(aligned.columns) == ref_cols}")


if __name__ == '__main__':
    main()
