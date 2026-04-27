# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import scienceplots

# ========================== 配置 ==========================
PT_ROOT = "./heat_pt"
OUT_DIR = "./gene_attr_plot"
DPI = 600
TOP_K_GENE = 10
TOP_K_PATHWAY = 10
GROUPS = ["protein"]
# ==========================================================


def plot_topk(pid, group, importance, out_path, top_k, label_prefix="Gene"):
    # 防护: top_k 不超过实际数据长度
    top_k = min(top_k, len(importance))
    if top_k == 0:
        print(f"[skip] {pid}/{group}: empty importance array")
        return

    order = np.argsort(np.abs(importance))[::-1][:top_k]
    vals = importance[order]
    labs = [f"{label_prefix} #{i}" for i in order]

    vmax = max(np.max(np.abs(importance)), 1e-9)
    cmap = plt.cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colors = [cmap(norm(v)) for v in vals]

    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(2.4, max(1.4, 0.22 * top_k)))
        y_pos = np.arange(len(vals))
        ax.barh(y_pos, vals[::-1], color=colors[::-1],
                edgecolor="black", linewidth=0.3, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labs[::-1], fontsize=7)
        ax.axvline(0, color="black", lw=0.4)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    for group in GROUPS:
        pt_dir = os.path.join(PT_ROOT, group)
        pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        if not pt_files:
            print(f"[{group}] No .pt files in {pt_dir}, skip.")
            continue

        gene_out = os.path.join(OUT_DIR, group, "genes")
        path_out = os.path.join(OUT_DIR, group, "pathways")
        os.makedirs(gene_out, exist_ok=True)
        os.makedirs(path_out, exist_ok=True)

        print(f"\n========== {group} ({len(pt_files)} patients) ==========")

        for pt_path in pt_files:
            d = torch.load(pt_path, map_location="cpu")
            pid = d["patient_id"]
            gene_imp = d["gene_importance"].numpy().astype(np.float64)
            pw_imp = d["pathway_importance"].numpy().astype(np.float64)

            # Top-K 基因柱状图
            plot_topk(pid, group, gene_imp,
                      os.path.join(gene_out, f"{pid}.png"),
                      top_k=TOP_K_GENE, label_prefix="Gene")

            # Top-K Pathway 柱状图
            # plot_topk(pid, group, pw_imp,
            #           os.path.join(path_out, f"{pid}.png"),
            #           top_k=TOP_K_PATHWAY, label_prefix="Pathway")

            print(f"[ok] {pid}  genes={len(gene_imp)} (top-{min(TOP_K_GENE, len(gene_imp))})  "
                  f"pathways={len(pw_imp)} (top-{min(TOP_K_PATHWAY, len(pw_imp))})")

    print(f"\nAll plots saved under: {OUT_DIR}/")


if __name__ == "__main__":
    main()
