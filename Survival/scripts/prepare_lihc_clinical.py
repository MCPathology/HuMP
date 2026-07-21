"""
Extract LIHC clinical features from the GDC clinical.tsv into the datasets_csv
layout used by HuMP.  Survival / label columns are deliberately ignored
(the user already has labels).

Outputs:
  clinical_data/tcga_lihc_clinical.csv     index, case_id, stage, subtype, grade
                                           (same schema as tcga_brca_clinical.csv)
  clinical_data/tcga_lihc_clinical_full.csv  richer demographic + diagnostic
                                           attributes for the clinical (C) text
                                           encoder, with all leakage columns
                                           (survival, vital_status, days_*) removed.

Usage:
  python scripts/prepare_lihc_clinical.py \
      --clinical_tsv lihc_clinical/clinical.tsv \
      --out_dir datasets_csv/clinical_data
"""

import argparse
import os

import numpy as np
import pandas as pd


# columns that would leak the survival label -> never used as C-modality input
LEAK_KEYS = (
    'days_to_death', 'days_to_last_follow', 'days_to_last_known',
    'vital_status', 'year_of_death', 'cause_of_death', 'survival',
    'days_to_recurrence', 'progression', 'days_to_diagnosis',
    'days_to_consent', 'lost_to_followup', 'days_to_birth',
)


def _norm_stage(s):
    """'Stage IIIA' / 'Stage I' -> 'III' / 'I' ; missing -> 'N/A'."""
    if pd.isna(s) or str(s).strip().lower() in ('na', '', 'not reported', 'unknown'):
        return 'N/A'
    t = str(s).upper().replace('STAGE', '').strip()
    for roman in ['IV', 'III', 'II', 'I']:
        if t.startswith(roman):
            return roman
    return 'N/A'


def _norm_grade(g):
    """'G2' -> 'G2' ; missing -> 'N/A'."""
    if pd.isna(g) or str(g).strip().lower() in ('na', '', 'not reported', 'unknown', 'gx'):
        return 'N/A'
    return str(g).strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clinical_tsv', required=True)
    ap.add_argument('--out_dir', default='datasets_csv/clinical_data')
    ap.add_argument('--embed_dir', default='csv',
                    help="Where to write the embedding.py-ready CSV "
                         "(defaults to 'csv/' so it lands at csv/<cohort>.csv).")
    ap.add_argument('--cohort', default='lihc')
    args = ap.parse_args()

    df = pd.read_csv(args.clinical_tsv, sep='\t', dtype=str).replace("'--", np.nan)
    df = df.drop_duplicates('cases.submitter_id').reset_index(drop=True)
    print(f"[load] {len(df)} unique cases, {df.shape[1]} fields")

    cid = df['cases.submitter_id'].astype(str)

    # ---- 1. simple clinical_data csv: case_id, stage, subtype, grade ----
    simple = pd.DataFrame({
        'case_id': cid,
        'stage':   df['diagnoses.ajcc_pathologic_stage'].map(_norm_stage),
        'subtype': df['diagnoses.primary_diagnosis'].fillna('LIHC'),
        'grade':   df['diagnoses.tumor_grade'].map(_norm_grade),
    })
    os.makedirs(args.out_dir, exist_ok=True)
    simple_path = os.path.join(args.out_dir, f'tcga_{args.cohort}_clinical.csv')
    simple.to_csv(simple_path)
    print(f"[out]  {simple_path}  ({len(simple)} rows)")
    print(f"       stage:  {dict(simple['stage'].value_counts())}")
    print(f"       grade:  {dict(simple['grade'].value_counts())}")

    # ---- 2. richer clinical table for the C-modality text encoder ----
    # keep demographic + diagnostic descriptive fields; drop leakage columns.
    wanted = {
        'demographic.gender':                 'gender',
        'demographic.race':                   'race',
        'demographic.ethnicity':              'ethnicity',
        'demographic.age_at_index':           'age',
        'diagnoses.ajcc_pathologic_stage':    'pathologic_stage',
        'diagnoses.ajcc_pathologic_t':        'pathologic_t',
        'diagnoses.ajcc_pathologic_n':        'pathologic_n',
        'diagnoses.ajcc_pathologic_m':        'pathologic_m',
        'diagnoses.tumor_grade':              'grade',
        'diagnoses.primary_diagnosis':        'primary_diagnosis',
        'diagnoses.morphology':               'morphology',
        'diagnoses.tissue_or_organ_of_origin':'tissue_of_origin',
        'diagnoses.site_of_resection_or_biopsy': 'site_of_resection',
        'diagnoses.prior_malignancy':         'prior_malignancy',
        'diagnoses.prior_treatment':          'prior_treatment',
        'diagnoses.classification_of_tumor':  'classification_of_tumor',
    }
    cols = {'case_id': cid}
    for src, dst in wanted.items():
        if src in df.columns and not any(k in src for k in LEAK_KEYS):
            cols[dst] = df[src]
    full = pd.DataFrame(cols)
    # normalize obvious 'Not Reported'/'Unknown' to NaN for cleaner prompts
    full = full.replace(
        {'Not Reported': np.nan, 'not reported': np.nan, 'Unknown': np.nan})
    full_path = os.path.join(args.out_dir, f'tcga_{args.cohort}_clinical_full.csv')
    full.to_csv(full_path, index=False)
    print(f"[out]  {full_path}  ({len(full)} rows, "
          f"{full.shape[1]-1} attribute cols)")

    # ---- 3. embedding-ready CSV for embedding.py (GDC-portal column style) ----
    # embedding.py expects ID column 'Case Submitter ID' and turns every other
    # column into a sentence "<col with _->space> is <value>".  We therefore use
    # Title-Case GDC-portal column names so the prompts match the other cohorts.
    rename = {
        'case_id':                 'Case Submitter ID',
        'gender':                  'Gender',
        'race':                    'Race',
        'ethnicity':               'Ethnicity',
        'age':                     'Age at Index',
        'pathologic_stage':        'AJCC Pathologic Stage',
        'pathologic_t':            'AJCC Pathologic T',
        'pathologic_n':            'AJCC Pathologic N',
        'pathologic_m':            'AJCC Pathologic M',
        'grade':                   'Tumor Grade',
        'primary_diagnosis':       'Primary Diagnosis',
        'morphology':              'Morphology',
        'tissue_of_origin':        'Tissue or Organ of Origin',
        'site_of_resection':       'Site of Resection or Biopsy',
        'prior_malignancy':        'Prior Malignancy',
        'prior_treatment':         'Prior Treatment',
        'classification_of_tumor': 'Classification of Tumor',
    }
    embed = full.rename(columns=rename)
    embed_dir = args.embed_dir
    os.makedirs(embed_dir, exist_ok=True)
    embed_path = os.path.join(embed_dir, f'{args.cohort}.csv')
    embed.to_csv(embed_path, index=False)
    print(f"[out]  {embed_path}  (embedding.py input; ID col = 'Case Submitter ID')")
    print(f"       columns: {list(embed.columns)}")
    print("\n[NOTE] Feed the embedding-ready CSV to embedding.py "
          "(place it at its 'csv/<cohort>.csv'). All survival/vital/days "
          "columns were excluded to prevent label leakage.")


if __name__ == '__main__':
    main()
