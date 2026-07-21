"""
Extract TCGA-LIHC RNA expression from the UCSC Xena pan-cancer TPM matrix
into the datasets_csv layout used by HuMP.

Input
    tcga_RSEM_gene_tpm.gz   (UCSC Xena Toil hub)
        rows = genes (Ensembl IDs, e.g. ENSG00000242268.2)
        cols = TCGA sample barcodes (e.g. TCGA-EP-A3RK-01)
        value = log2(tpm + 0.001)

LIHC samples must be selected from this pan-cancer matrix.  Three ways,
in order of reliability:
    1. --case_list FILE   explicit TCGA case_ids (TCGA-XX-XXXX) or sample
                          barcodes, one per line OR a CSV with a 'case_id'
                          column (your metadata works directly).
    2. --phenotype FILE   Xena pan-cancer phenotype TSV that maps each
                          sample to a cancer type; rows whose type matches
                          --cancer_name (default 'Liver Hepatocellular
                          Carcinoma') are kept.
    3. --tss_codes ...    fallback: keep barcodes whose tissue-source-site
                          code is in the given list (approximate).

Gene IDs in the TPM file are Ensembl; your pipeline uses HUGO symbols, so a
probemap is required:
    --probemap FILE       Xena gene probemap (Ensembl<TAB>symbol<...>) used
                          to translate Ensembl -> symbol.  If the matrix
                          already uses symbols, pass --no_probemap.

Output
    datasets_csv/raw_rna_data/combine/lihc/rna_clean.csv
        rows = LIHC samples (index = case_id), cols = signature genes

Usage
    # inspect format first (header + first gene id + sample barcodes)
    python scripts/extract_lihc_rna.py --gene_tpm /path/tcga_RSEM_gene_tpm.gz --inspect

    # extract by case list (recommended)
    python scripts/extract_lihc_rna.py \
        --gene_tpm /path/tcga_RSEM_gene_tpm.gz \
        --probemap /path/probeMap_gencode.v23.annotation.gene.probemap \
        --case_list datasets_csv/metadata/tcga_lihc.csv
"""

import argparse
import csv
import gzip
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def _find_sig_dir(user_dir):
    """Locate the directory holding the two signature CSVs."""
    candidates = []
    if user_dir:
        candidates.append(user_dir)
    candidates += [
        ROOT,                                   # parent of scripts/
        HERE,                                   # alongside the script
        os.getcwd(),                            # current working dir
        os.path.join(os.getcwd(), 'Survival'),
    ]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, 'genomics_signatures.csv')):
            return d
    return None


def load_signature_genes(sig_dir):
    d = _find_sig_dir(sig_dir)
    if d is None:
        raise FileNotFoundError(
            "Could not find genomics_signatures.csv / transcripts_signatures.csv. "
            "Pass --sig_dir pointing to the folder that contains them "
            "(e.g. your Survival root), or pass --no_align to keep all genes.")
    print(f"[sig]  using signatures in {d}")
    need, seen = [], set()
    for f in ['genomics_signatures.csv', 'transcripts_signatures.csv']:
        with open(os.path.join(d, f)) as fh:
            rows = list(csv.reader(fh))
        for r in rows[1:]:
            for c in r:
                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    need.append(c)
    return need


def load_probemap(path):
    """Xena probemap: col0 = Ensembl id (maybe versioned), col1 = symbol."""
    m = {}
    with _open(path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            ens, sym = parts[0], parts[1]
            m[ens] = sym
            m[ens.split('.')[0]] = sym          # also map unversioned id
    return m


def case_of(barcode):
    """TCGA-EP-A3RK-01A-... -> TCGA-EP-A3RK"""
    return '-'.join(barcode.split('-')[:3])


def sample_type(barcode):
    """4th field, e.g. '01A' -> '01' (01=primary tumor, 11=normal)."""
    parts = barcode.split('-')
    if len(parts) < 4:
        return None
    return parts[3][:2]


def tss_of(barcode):
    """TCGA-EP-A3RK -> 'EP'"""
    parts = barcode.split('-')
    return parts[1] if len(parts) > 1 else None


# ---------------------------------------------------------------------------
# select LIHC columns from the header
# ---------------------------------------------------------------------------
def select_lihc_columns(header_cols, args):
    """Return {col_index: case_id} for the LIHC primary-tumor samples."""
    keep = {}

    # build the membership predicate
    if args.case_list:
        wanted = _read_case_list(args.case_list)
        print(f"[sel]  case_list provided: {len(wanted)} ids")
        def is_lihc(bc):
            return (case_of(bc) in wanted) or (bc in wanted)
    elif args.phenotype:
        wanted = _read_phenotype(args.phenotype, args.cancer_name)
        print(f"[sel]  phenotype LIHC samples: {len(wanted)}")
        def is_lihc(bc):
            return (bc in wanted) or (case_of(bc) in wanted)
    elif args.tss_codes:
        codes = set(c.strip().upper() for c in args.tss_codes.split(','))
        print(f"[sel]  TSS codes: {sorted(codes)}")
        def is_lihc(bc):
            return tss_of(bc) in codes
    else:
        raise SystemExit("Provide one of --case_list / --phenotype / --tss_codes")

    seen_case = set()
    for i, bc in enumerate(header_cols):
        if not bc.startswith('TCGA'):
            continue
        if args.primary_only and sample_type(bc) != '01':
            continue
        if not is_lihc(bc):
            continue
        cid = case_of(bc)
        if cid in seen_case:                  # one aliquot per case
            continue
        seen_case.add(cid)
        keep[i] = cid
    return keep


def _read_case_list(path):
    wanted = set()
    if path.endswith('.csv'):
        df = pd.read_csv(path, dtype=str)
        col = 'case_id' if 'case_id' in df.columns else df.columns[
            1 if df.columns[0].startswith('Unnamed') else 0]
        for v in df[col].dropna():
            wanted.add(str(v).strip())
    else:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s:
                    wanted.add(s)
    return wanted


def _read_phenotype(path, cancer_name):
    wanted = set()
    cancer_name = cancer_name.lower()
    with _open(path) as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = next(reader)
        # find the column holding cancer-type text
        type_idx = None
        for j, h in enumerate(header):
            if any(k in h.lower() for k in
                   ['detailed_category', 'disease', 'primary_disease',
                    '_primary_site', 'cancer type', 'project']):
                type_idx = j
        if type_idx is None:
            type_idx = 1
        for row in reader:
            if len(row) <= type_idx:
                continue
            if cancer_name in row[type_idx].lower():
                wanted.add(row[0])
    return wanted


# ---------------------------------------------------------------------------
# main extraction
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene_tpm', required=True,
                    help='Path to tcga_RSEM_gene_tpm.gz')
    ap.add_argument('--probemap', default=None,
                    help='Xena gene probemap (Ensembl -> symbol).')
    ap.add_argument('--no_probemap', action='store_true',
                    help='Matrix already uses HUGO symbols; skip mapping.')
    ap.add_argument('--case_list', default=None)
    ap.add_argument('--phenotype', default=None)
    ap.add_argument('--cancer_name', default='Liver Hepatocellular Carcinoma')
    ap.add_argument('--tss_codes', default=None,
                    help='Comma-separated LIHC TSS codes (fallback).')
    ap.add_argument('--primary_only', action='store_true', default=True,
                    help='Keep only primary-tumor (-01) samples (default).')
    ap.add_argument('--fill_missing', type=float, default=0.0)
    ap.add_argument('--sig_dir', default=None,
                    help='Folder containing genomics_signatures.csv and '
                         'transcripts_signatures.csv (defaults to repo root / cwd).')
    ap.add_argument('--no_align', action='store_true',
                    help='Skip signature-gene alignment; keep ALL genes.')
    ap.add_argument('--out', default=os.path.join(
        ROOT, 'datasets_csv', 'raw_rna_data', 'combine', 'lihc', 'rna_clean.csv'))
    ap.add_argument('--inspect', action='store_true',
                    help='Print format diagnostics and exit.')
    args = ap.parse_args()

    # ---- inspect mode ----
    if args.inspect:
        with _open(args.gene_tpm) as fh:
            header = fh.readline().rstrip('\n').split('\t')
            first = fh.readline().rstrip('\n').split('\t')
        print(f"[inspect] columns (incl. gene-id col): {len(header)}")
        print(f"[inspect] first header cell (gene-id col name): {header[0]!r}")
        print(f"[inspect] sample barcodes (first 5): {header[1:6]}")
        print(f"[inspect] first gene id: {first[0]!r}")
        print(f"[inspect] -> looks like "
              f"{'ENSEMBL (needs --probemap)' if first[0].upper().startswith('ENSG') else 'SYMBOL (use --no_probemap)'}")
        tss = sorted({tss_of(c) for c in header[1:] if c.startswith('TCGA')})
        print(f"[inspect] distinct TSS codes present: {len(tss)}")
        return

    # ---- header + LIHC column selection ----
    with _open(args.gene_tpm) as fh:
        header = fh.readline().rstrip('\n').split('\t')
    keep = select_lihc_columns(header, args)
    if not keep:
        raise SystemExit("No LIHC columns matched. Try --inspect, or check "
                         "your --case_list / --phenotype / --tss_codes.")
    col_idx = sorted(keep.keys())
    case_ids = [keep[i] for i in col_idx]
    print(f"[sel]  matched LIHC samples: {len(col_idx)}")

    # ---- probemap ----
    probemap = None
    if not args.no_probemap:
        if not args.probemap:
            raise SystemExit("Matrix uses Ensembl IDs -> need --probemap "
                             "(or --no_probemap if it is already symbols).")
        probemap = load_probemap(args.probemap)
        print(f"[map]  probemap entries: {len(probemap)}")

    # ---- stream genes, collect LIHC columns ----
    gene2vals = {}       # symbol -> list of np arrays (collapse dups later)
    n_rows = 0
    with _open(args.gene_tpm) as fh:
        fh.readline()                                   # skip header
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            gid = parts[0]
            sym = gid
            if probemap is not None:
                sym = probemap.get(gid) or probemap.get(gid.split('.')[0])
                if sym is None:
                    continue
            vals = np.array([_safe_float(parts[i]) for i in col_idx],
                            dtype=np.float32)
            gene2vals.setdefault(sym, []).append(vals)
            n_rows += 1
            if n_rows % 10000 == 0:
                print(f"  ...{n_rows} gene rows")
    print(f"[stream] parsed {n_rows} gene rows -> {len(gene2vals)} symbols")

    # collapse duplicate symbols by max (common convention for log-expr)
    mat = {}
    for sym, lst in gene2vals.items():
        mat[sym] = lst[0] if len(lst) == 1 else np.max(np.stack(lst), axis=0)

    # ---- align to the model's signature genes (or keep all) ----
    if args.no_align:
        cols = sorted(mat.keys())
        print(f"[align] --no_align: keeping all {len(cols)} genes")
        data = np.stack([mat[g] for g in cols], axis=1)   # [samples, genes]
        df = pd.DataFrame(data, index=case_ids, columns=cols)
    else:
        need = load_signature_genes(args.sig_dir)
        present = [g for g in need if g in mat]
        missing = [g for g in need if g not in mat]
        print(f"[align] signature genes: {len(need)} | present: {len(present)} | "
              f"missing: {len(missing)}")
        data = np.full((len(case_ids), len(need)), args.fill_missing,
                       dtype=np.float32)
        for j, g in enumerate(need):
            if g in mat:
                data[:, j] = mat[g]
        df = pd.DataFrame(data, index=case_ids, columns=need)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out)
    print(f"[done]  wrote {args.out}  shape {df.shape} (samples x genes)")
    print(f"[done]  value range: min={np.nanmin(data):.3f} "
          f"max={np.nanmax(data):.3f} mean={np.nanmean(data):.3f}")
    print("\n[NOTE] These are log2(tpm+0.001) values from Xena. If your other "
          "cohorts used a different transform (e.g. RSEM log2 from HiSeqV2), "
          "the absolute scale will differ. Verify the per-gene normalization "
          "in your dataset class re-centers both consistently.")


def _safe_float(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


if __name__ == '__main__':
    main()
