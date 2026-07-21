"""
Convert raw CPTAC-BRCA LinkedOmics files into the datasets_csv layout used
by the HuMP pipeline, for external validation.

Inputs (place under CPTAC-BRCA/):
    HS_CPTAC_BRCA_2018_RNA_GENE.cct                      RNA-seq, genes x samples
    HS_CPTAC_BRCA_2018_CLI.tsi                           clinical, samples x fields
    HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct  proteome (not used here)

Outputs (mirrors the TCGA layout, written under datasets_csv/.../cptac_brca):
    raw_rna_data/combine/cptac_brca/rna_clean.csv        samples x genes (z-scored)
    clinical_data/cptac_brca_clinical.csv                index, case_id, stage, subtype, grade
    metadata/cptac_brca.csv                              metadata skeleton
                                                         (SURVIVAL COLUMNS LEFT BLANK)

Survival columns (survival_months*, censorship*) are intentionally left as
NaN because the CPTAC-BRCA 2018 clinical table does NOT contain follow-up /
vital-status data.  They must be filled from the GDC CPTAC-BRCA clinical
export before any survival metric can be computed.

Usage:
    python scripts/prepare_cptac_brca.py
    python scripts/prepare_cptac_brca.py --fill_missing_genes 0.0
"""

import argparse
import os

import numpy as np
import pandas as pd


HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Genes the model consumes, read from the signature CSVs
# ---------------------------------------------------------------------------
def load_signature_genes():
    import csv
    need = []
    seen = set()
    for f in ['genomics_signatures.csv', 'transcripts_signatures.csv']:
        path = os.path.join(ROOT, f)
        with open(path) as fh:
            rows = list(csv.reader(fh))
        for r in rows[1:]:
            for c in r:
                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    need.append(c)
    return need


# ---------------------------------------------------------------------------
# 1. RNA: transpose genes x samples -> samples x genes, align to model genes
# ---------------------------------------------------------------------------
def convert_rna(cptac_dir, out_dir, fill_missing):
    src = os.path.join(cptac_dir, 'HS_CPTAC_BRCA_2018_RNA_GENE.cct')
    print(f"[rna]  reading {src}")
    # genes in rows (index), samples in columns
    df = pd.read_csv(src, sep='\t', index_col=0)
    df.index = df.index.astype(str)
    print(f"[rna]  raw shape: {df.shape}  (genes x samples)")

    # transpose -> samples x genes
    mat = df.T                                          # [samples, genes]
    mat.index = mat.index.astype(str)

    # align to the gene set the model expects
    need = load_signature_genes()
    present = [g for g in need if g in mat.columns]
    missing = [g for g in need if g not in mat.columns]
    print(f"[rna]  signature genes: {len(need)} | "
          f"present: {len(present)} | missing: {len(missing)}")

    aligned = mat.reindex(columns=need)                 # missing genes -> NaN
    if fill_missing is not None:
        aligned = aligned.fillna(fill_missing)
        print(f"[rna]  filled {len(missing)} missing genes with {fill_missing}")

    # the TCGA rna_clean.csv has a leading unnamed index column whose values
    # are the case_ids; we mirror that by writing index=True.
    out_path = os.path.join(out_dir, 'rna_clean.csv')
    os.makedirs(out_dir, exist_ok=True)
    aligned.to_csv(out_path)
    print(f"[rna]  wrote {out_path}  shape {aligned.shape} (samples x genes)")
    return list(aligned.index)


# ---------------------------------------------------------------------------
# 2. Clinical: map CPTAC fields -> {case_id, stage, subtype, grade}
# ---------------------------------------------------------------------------
def _norm_stage(s):
    """Map 'Stage IIA' / 'Stage III' -> 'II' / 'III' to match TCGA style."""
    if pd.isna(s) or str(s).strip().upper() in ('NA', ''):
        return 'N/A'
    t = str(s).upper().replace('STAGE', '').strip()
    # strip trailing A/B/C subdivision to roman numeral
    for roman in ['IV', 'III', 'II', 'I']:
        if t.startswith(roman):
            return roman
    return 'N/A'


def convert_clinical(cptac_dir, clin_out, meta_out, rna_samples):
    src = os.path.join(cptac_dir, 'HS_CPTAC_BRCA_2018_CLI.tsi')
    print(f"[cli]  reading {src}")
    # row 0 = header, row 1 = data-type annotation (IDX/CAT/BIN/...), data from row 2
    raw = pd.read_csv(src, sep='\t', dtype=str)
    # drop the data-type annotation row (first data row)
    raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.rename(columns={'Sample.ID': 'case_id'})
    raw['case_id'] = raw['case_id'].astype(str)
    print(f"[cli]  {len(raw)} samples, {raw.shape[1]} fields")

    # ---- clinical_data csv: case_id, stage, subtype, grade ----
    clin = pd.DataFrame({
        'case_id': raw['case_id'],
        'stage':   raw['Stage'].map(_norm_stage),
        'subtype': raw['PAM50'].fillna('BRCA'),         # PAM50 as subtype label
        'grade':   'N/A',                               # CPTAC table has no grade
    })
    os.makedirs(os.path.dirname(clin_out), exist_ok=True)
    clin.to_csv(clin_out)                               # leading index col like TCGA
    print(f"[cli]  wrote {clin_out}  ({len(clin)} rows)")

    # ---- metadata skeleton: same columns as tcga_brca.csv ----
    # SURVIVAL COLUMNS ARE LEFT AS NaN (must come from GDC follow-up).
    age_years = pd.to_numeric(raw['Age.in.Month'], errors='coerce') / 12.0
    is_female = 1.0   # BRCA cohort: assume female unless a sex column exists
    meta = pd.DataFrame({
        'case_id':              raw['case_id'],
        'slide_id':             '',                     # FILL: WSI filename from TCIA
        'age':                  age_years.round(1),
        'site':                 raw['case_id'].str[:4],
        'survival_months':      np.nan,                 # FILL from GDC
        'survival_months_dss':  np.nan,                 # FILL from GDC
        'survival_months_pfi':  np.nan,                 # FILL from GDC
        'censorship':           np.nan,                 # FILL from GDC
        'censorship_dss':       np.nan,                 # FILL from GDC
        'censorship_pfi':       np.nan,                 # FILL from GDC
        'is_female':            is_female,
        'oncotree_code':        raw['PAM50'].fillna('BRCA'),
        'train':                0.0,                    # 0 = external test only
    })
    os.makedirs(os.path.dirname(meta_out), exist_ok=True)
    meta.to_csv(meta_out)
    print(f"[meta] wrote {meta_out}  ({len(meta)} rows)")

    # ---- sample-overlap diagnostic ----
    rna_set = set(rna_samples)
    cli_set = set(raw['case_id'])
    inter = rna_set & cli_set
    print(f"[join] RNA samples: {len(rna_set)} | "
          f"clinical samples: {len(cli_set)} | "
          f"intersection: {len(inter)}")
    return inter


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cptac_dir', default=os.path.join(ROOT, 'CPTAC-BRCA'))
    ap.add_argument('--fill_missing_genes', type=float, default=0.0,
                    help='Value to fill the ~162 signature genes absent from '
                         'the CPTAC RNA matrix (default 0.0 = cohort z-score mean).')
    args = ap.parse_args()

    rna_out_dir = os.path.join(ROOT, 'datasets_csv', 'raw_rna_data',
                               'combine', 'cptac_brca')
    clin_out = os.path.join(ROOT, 'datasets_csv', 'clinical_data',
                            'cptac_brca_clinical.csv')
    meta_out = os.path.join(ROOT, 'datasets_csv', 'metadata',
                            'cptac_brca.csv')

    print("=" * 70)
    rna_samples = convert_rna(args.cptac_dir, rna_out_dir, args.fill_missing_genes)
    print("-" * 70)
    inter = convert_clinical(args.cptac_dir, clin_out, meta_out, rna_samples)
    print("=" * 70)

    # ---- final gap report ----
    print("\n###################  GAP REPORT  ###################")
    print(f"[OK]  RNA matrix         -> {rna_out_dir}/rna_clean.csv")
    print(f"[OK]  Clinical labels    -> {clin_out}")
    print(f"[OK]  Metadata skeleton  -> {meta_out}")
    print(f"[OK]  O+C overlap        -> {len(inter)} samples")
    print("")
    print("[MISSING - blocks survival validation]")
    print("  1. SURVIVAL DATA: survival_months* and censorship* are NaN.")
    print("     CPTAC-BRCA 2018 clinical has NO follow-up. Download the GDC")
    print("     CPTAC-BRCA clinical export (vital_status, days_to_death,")
    print("     days_to_last_follow_up) and merge into metadata/cptac_brca.csv.")
    print("  2. WSI PATHOLOGY: slide_id is empty. Download CPTAC-BRCA WSIs from")
    print("     TCIA, extract features with the SAME encoder used in training,")
    print("     and place under ../WSIdata/cptac_brca/s_files/<slide>.pt.")
    print("  3. PROTEIN BRANCH: the gene-level proteome .cct does not match the")
    print("     ESM-sequence protein input HuMP expects. Either run protein as a")
    print("     missing modality, or regenerate ESM embeddings.")
    print("####################################################")


if __name__ == '__main__':
    main()
