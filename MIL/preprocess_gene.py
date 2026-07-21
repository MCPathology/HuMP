import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "CPTAC-CCRCC": [ROOT / "CPTAC-3.star_counts.tsv.gz"],
    "NSCLC": [ROOT / "TCGA-LUAD.star_counts.tsv.gz", ROOT / "TCGA-LUSC.star_counts.tsv.gz"],
}


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return Path(paths[0])


DEFAULT_PROBEMAP = first_existing(
    [
        ROOT / "LUAD.probemap",
        ROOT / "LUSC.probemap",
        ROOT / "CPTAC.probemap",
    ]
)

DEFAULT_PATHWAY_COMPS = first_existing(
    [
        ROOT / "Survival" / "datasets_csv" / "pathway_compositions" / "combine_comps.csv",
        ROOT / "combine_comps.csv",
    ]
)


def strip_ensembl_version(gene_id):
    return str(gene_id).split(".")[0]


def load_probemap(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "id" not in df.columns or "gene" not in df.columns:
        raise ValueError(f"probemap must contain 'id' and 'gene' columns: {path}")
    df = df[["id", "gene"]].dropna()
    df["id"] = df["id"].astype(str)
    df["gene"] = df["gene"].astype(str).str.strip()
    df["id_stripped"] = df["id"].map(strip_ensembl_version)

    exact = dict(zip(df["id"], df["gene"]))
    stripped = dict(zip(df["id_stripped"], df["gene"]))
    return exact, stripped


def read_star_counts(path, exact_map, stripped_map):
    print(f"[read] {path}")
    df = pd.read_csv(path, sep="\t")
    gene_col = df.columns[0]
    df[gene_col] = df[gene_col].astype(str)

    symbols = df[gene_col].map(exact_map)
    missing = symbols.isna()
    if missing.any():
        symbols.loc[missing] = df.loc[missing, gene_col].map(
            lambda x: stripped_map.get(strip_ensembl_version(x))
        )

    df = df.loc[symbols.notna()].copy()
    df.insert(0, "gene_symbol", symbols.loc[symbols.notna()].astype(str).str.strip().values)
    df = df.drop(columns=[gene_col])

    sample_cols = [c for c in df.columns if c != "gene_symbol"]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df = df.groupby("gene_symbol", sort=False).mean(numeric_only=True)
    print(f"[read] mapped genes={df.shape[0]} samples={df.shape[1]}")
    return df


def patient_id(sample_id, dataset):
    sample_id = str(sample_id)
    if dataset == "NSCLC":
        return sample_id[:12]
    if dataset == "CPTAC-CCRCC":
        parts = sample_id.split("-")
        if len(parts) >= 2 and parts[0] in ("C3L", "C3N"):
            return "-".join(parts[:2])
    return sample_id


def keep_sample(sample_id, dataset):
    sample_id = str(sample_id)
    if dataset == "NSCLC":
        # TCGA barcode sample type lives at characters 14-15 (0-based 13:15).
        return sample_id.startswith("TCGA-") and len(sample_id) >= 15 and sample_id[13:15] == "01"
    if dataset == "CPTAC-CCRCC":
        return sample_id.startswith("C3L-") or sample_id.startswith("C3N-")
    return True


def load_pathways(path):
    comps = pd.read_csv(path)
    if "gene" not in comps.columns:
        raise ValueError(f"pathway composition file must contain a 'gene' column: {path}")
    comps["gene"] = comps["gene"].astype(str).str.strip()
    pathway_cols = [c for c in comps.columns if c != "gene"]
    pathways = {}
    for col in pathway_cols:
        members = comps.loc[pd.to_numeric(comps[col], errors="coerce").fillna(0) > 0, "gene"]
        pathways[col] = [g for g in members.astype(str).str.strip().tolist() if g]
    return pathways


def zscore_rows(df):
    values = df.to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    values = (values - mean) / std
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def build_patient_tokens(expr, dataset, pathways, dim):
    kept_cols = [c for c in expr.columns if keep_sample(c, dataset)]
    if not kept_cols:
        raise ValueError(f"No usable sample columns found for {dataset}")
    expr = expr.loc[:, kept_cols]

    union_genes = sorted({g for genes in pathways.values() for g in genes})
    present_genes = [g for g in union_genes if g in expr.index]
    expr = expr.loc[present_genes]
    print(f"[pathway] present pathway genes={len(present_genes)} / {len(union_genes)}")

    expr = zscore_rows(expr)
    pid_cols = [patient_id(c, dataset) for c in expr.columns]
    expr = expr.T.groupby(pid_cols, sort=True).mean().T
    variance = expr.var(axis=1).sort_values(ascending=False)

    pathway_names = list(pathways.keys())
    pathway_indices = []
    used_counts = {}
    for name in pathway_names:
        genes = [g for g in pathways[name] if g in expr.index]
        genes = sorted(genes, key=lambda g: float(variance.get(g, 0.0)), reverse=True)
        genes = genes[:dim]
        pathway_indices.append(genes)
        used_counts[name] = len(genes)

    patients = list(expr.columns)
    tokens_by_patient = {}
    for idx, pid in enumerate(patients):
        if idx % 100 == 0:
            print(f"[tokens] {dataset} {idx}/{len(patients)}")
        patient_expr = expr[pid]
        arr = np.zeros((len(pathway_names), dim), dtype=np.float32)
        for row, genes in enumerate(pathway_indices):
            if genes:
                values = patient_expr.loc[genes].to_numpy(dtype=np.float32)
                arr[row, : len(values)] = values
        tokens_by_patient[pid] = arr
    print(f"[tokens] {dataset} {len(patients)}/{len(patients)}")
    return tokens_by_patient, pathway_names, pathway_indices, used_counts


def write_dataset(out_root, dataset, tokens_by_patient, pathway_names, pathway_indices, used_counts, args):
    dataset_dir = out_root / dataset
    sample_dir = dataset_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for stale in sample_dir.glob("*.npy"):
        stale.unlink()

    rows = []
    for pid, arr in tokens_by_patient.items():
        out_path = sample_dir / f"{pid}.npy"
        np.save(out_path, arr)
        rows.append({"patient_id": pid, "path": f"samples/{pid}.npy", "shape": "x".join(map(str, arr.shape))})

    pd.DataFrame(rows).to_csv(dataset_dir / "manifest.csv", index=False)
    (dataset_dir / "pathways.txt").write_text("\n".join(pathway_names) + "\n", encoding="utf-8")
    pathway_genes = {name: genes for name, genes in zip(pathway_names, pathway_indices)}
    (dataset_dir / "pathway_genes.json").write_text(
        json.dumps(pathway_genes, indent=2, sort_keys=True), encoding="utf-8"
    )
    meta = {
        "dataset": dataset,
        "patients": len(tokens_by_patient),
        "num_pathways": len(pathway_names),
        "token_dim": args.dim,
        "pathway_comps": str(args.pathway_comps),
        "probemap": str(args.probemap),
        "used_gene_counts": used_counts,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[write] {dataset}: {len(rows)} patients -> {sample_dir}")


def preprocess_dataset(dataset, args, exact_map, stripped_map, pathways):
    exprs = []
    for path in DEFAULT_INPUTS[dataset]:
        if not path.exists():
            raise FileNotFoundError(path)
        exprs.append(read_star_counts(path, exact_map, stripped_map))

    if len(exprs) == 1:
        expr = exprs[0]
    else:
        common = exprs[0].index
        for frame in exprs[1:]:
            common = common.intersection(frame.index)
        expr = pd.concat([frame.loc[common] for frame in exprs], axis=1)
    print(f"[merge] {dataset}: genes={expr.shape[0]} samples={expr.shape[1]}")

    tokens, pathway_names, pathway_indices, used_counts = build_patient_tokens(
        expr, dataset, pathways, args.dim
    )
    write_dataset(args.output_root, dataset, tokens, pathway_names, pathway_indices, used_counts, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess transcriptomics into pathway-grouped patient tokens."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "NSCLC", "CPTAC-CCRCC", "nsclc", "cptac-ccrcc"],
        default="all",
    )
    parser.add_argument("--dim", type=int, default=512, help="Genes retained per pathway token.")
    parser.add_argument("--output_root", type=Path, default=ROOT / "gene_features")
    parser.add_argument("--probemap", type=Path, default=DEFAULT_PROBEMAP)
    parser.add_argument("--pathway_comps", type=Path, default=DEFAULT_PATHWAY_COMPS)
    return parser.parse_args()


def main():
    args = parse_args()
    args.probemap = Path(args.probemap)
    args.pathway_comps = Path(args.pathway_comps)
    args.output_root = Path(args.output_root)

    if not args.probemap.exists():
        raise FileNotFoundError(f"probemap not found: {args.probemap}")
    if not args.pathway_comps.exists():
        raise FileNotFoundError(f"pathway composition file not found: {args.pathway_comps}")

    exact_map, stripped_map = load_probemap(args.probemap)
    pathways = load_pathways(args.pathway_comps)
    print(f"[config] pathways={len(pathways)} dim={args.dim}")
    print(f"[config] probemap={args.probemap}")
    print(f"[config] pathway_comps={args.pathway_comps}")

    dataset_arg = args.dataset.upper()
    datasets = ["NSCLC", "CPTAC-CCRCC"] if dataset_arg == "ALL" else [dataset_arg]
    datasets = ["CPTAC-CCRCC" if d == "CPTAC-CCRCC" else "NSCLC" for d in datasets]
    for dataset in datasets:
        preprocess_dataset(dataset, args, exact_map, stripped_map, pathways)


if __name__ == "__main__":
    main()
