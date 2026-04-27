# -*- coding: utf-8 -*-
import os
import glob
import torch
import h5py
import openslide
import numpy as np
from PIL import Image
from matplotlib import cm
from tqdm import tqdm
from scipy.spatial import KDTree
from wsi_core.WholeSlideImage import WholeSlideImage

K_NEIGHBORS = 25
MODALITIES = ["genomics", "transomics", "protein"]

# ==================== 论文级输出参数 ====================
TARGET_DPI = 600          # Nature / Cell 主图标准
TARGET_PHYSICAL_INCH = 3.0  # 目标打印宽度 (英寸). 典型单栏图 ~3.3in, 双栏 ~7in
OUTPUT_FORMAT = "png"     # "png" 无损 / "tiff" 更稳 / "jpg" 最省空间
JPG_QUALITY = 95          # 仅在 OUTPUT_FORMAT="jpg" 时生效
# ======================================================


def smooth_scores_by_knn(coords, scores, k=10):
    if k <= 1:
        return scores
    kdtree = KDTree(coords)
    _, neighbor_indices = kdtree.query(coords, k=k)
    smoothed = np.zeros_like(scores, dtype=np.float64)
    for i in range(len(coords)):
        smoothed[i] = np.mean(scores[neighbor_indices[i]])
    return smoothed


def get_wsi_physical_inches(svs_path):
    """
    从 WSI 读取 level-0 的物理尺寸 (英寸).
    如果没有 mpp 元数据, 回退到一个合理的默认值 (mpp=0.5, 20x 扫描).
    """
    slide = openslide.OpenSlide(svs_path)
    try:
        mpp_x = float(slide.properties.get("openslide.mpp-x", 0.5))
        mpp_y = float(slide.properties.get("openslide.mpp-y", 0.5))
    except Exception:
        mpp_x = mpp_y = 0.5

    w0, h0 = slide.level_dimensions[0]
    slide.close()

    # μm -> inch: 1 inch = 25400 μm
    w_inch = w0 * mpp_x / 25400.0
    h_inch = h0 * mpp_y / 25400.0
    return w_inch, h_inch


def resize_to_publication_dpi(img, w_inch, h_inch,
                               target_dpi=TARGET_DPI,
                               target_inch=TARGET_PHYSICAL_INCH):
    """
    把 heatmap 缩放到论文级 DPI.

    策略: 让图片在期刊版面上以 target_inch (长边) 显示时, 刚好是 target_dpi.
    即最终像素 = target_inch * target_dpi (长边), 另一边按原始长宽比.
    """
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h

    # 按长边对齐到 target_inch * target_dpi
    if w_inch >= h_inch:
        new_w = int(round(target_inch * target_dpi))
        new_h = int(round(new_w / aspect))
    else:
        new_h = int(round(target_inch * target_dpi))
        new_w = int(round(new_h * aspect))

    # 仅在会缩小时 resize (避免把小图放大造成糊)
    if new_w < orig_w or new_h < orig_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def save_publication(img, out_path, dpi=TARGET_DPI, fmt=OUTPUT_FORMAT):
    """带 DPI 元数据保存."""
    save_kwargs = {"dpi": (dpi, dpi)}
    if fmt == "png":
        save_kwargs["optimize"] = True
    elif fmt == "tiff":
        save_kwargs["compression"] = "tiff_lzw"
    elif fmt == "jpg" or fmt == "jpeg":
        # JPG 不支持 alpha, 需要先转 RGB
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        save_kwargs["quality"] = JPG_QUALITY
        save_kwargs["optimize"] = True

    img.save(out_path, **save_kwargs)


def process_one_modality(modality, pt_dir, h5_dir, svs_dir, output_dir, vis_level=1):
    os.makedirs(output_dir, exist_ok=True)
    pt_files = sorted(glob.glob(os.path.join(pt_dir, '*.pt')))

    if not pt_files:
        print(f"[{modality}] No .pt in '{pt_dir}', skip.")
        return

    print(f"\n========== Modality: {modality}  ({len(pt_files)} patients) ==========")

    for pt_path in tqdm(pt_files, desc=f"[{modality}]"):
        base_name = os.path.basename(pt_path)
        patient_id = os.path.splitext(base_name)[0]

        h5_path = os.path.join(h5_dir, f"{patient_id}.h5")
        svs_path = os.path.join(svs_dir, f"{patient_id}.svs")

        if not os.path.exists(h5_path):
            print(f"[{modality}] missing h5: {h5_path}, skip.")
            continue
        if not os.path.exists(svs_path):
            print(f"[{modality}] missing svs: {svs_path}, skip.")
            continue

        try:
            attn_data = torch.load(pt_path, map_location='cpu')
            scores = attn_data['cross_attn_scores'].numpy()
            if scores.ndim > 1:
                scores = scores.squeeze()

            with h5py.File(h5_path, 'r') as hf:
                coords = hf['coords'][:]

            if len(coords) != scores.shape[0]:
                print(f"[{modality}] {patient_id}: coords({len(coords)}) vs scores({scores.shape[0]}) mismatch, skip.")
                continue

            smoothed = smooth_scores_by_knn(coords, scores, k=K_NEIGHBORS)

            wsi_object = WholeSlideImage(svs_path)
            heatmap_image = wsi_object.visHeatmap(
                scores=smoothed,
                coords=coords,
                convert_to_percentiles=True,
                cmap='jet',
                vis_level=vis_level,
            )

            # ---- 论文级 DPI 处理 ----
            w_inch, h_inch = get_wsi_physical_inches(svs_path)
            heatmap_image = resize_to_publication_dpi(
                heatmap_image, w_inch, h_inch,
                target_dpi=TARGET_DPI,
                target_inch=TARGET_PHYSICAL_INCH,
            )

            ext = OUTPUT_FORMAT if OUTPUT_FORMAT != "jpeg" else "jpg"
            out_path = os.path.join(
                output_dir,
                f"{patient_id}_{modality}_level{vis_level}_{TARGET_DPI}dpi.{ext}"
            )
            save_publication(heatmap_image, out_path, dpi=TARGET_DPI, fmt=OUTPUT_FORMAT)

            # 打印确认信息
            final_w, final_h = heatmap_image.size
            print(f"[{modality}] {patient_id}: "
                  f"{final_w}x{final_h}px @ {TARGET_DPI}DPI "
                  f"(~{final_w/TARGET_DPI:.2f}x{final_h/TARGET_DPI:.2f} inch)")

        except Exception as e:
            print(f"[{modality}] {patient_id} failed: {e}")
            import traceback; traceback.print_exc()


def process_all_modalities(pt_root, h5_dir, svs_dir, output_root, vis_level=1):
    for modality in MODALITIES:
        pt_dir = os.path.join(pt_root, modality)
        output_dir = os.path.join(output_root, modality)
        process_one_modality(
            modality=modality,
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            svs_dir=svs_dir,
            output_dir=output_dir,
            vis_level=vis_level,
        )


# =============================================================================
if __name__ == '__main__':
    study = 'brca'

    PT_ROOT = './heat_pt'
    H5_FILES_DIRECTORY = f'../WSIdata/{study}/h5_files/'
    SVS_FILES_DIRECTORY = '../BRCA/'
    OUTPUT_ROOT = './heatmap'

    process_all_modalities(
        pt_root=PT_ROOT,
        h5_dir=H5_FILES_DIRECTORY,
        svs_dir=SVS_FILES_DIRECTORY,
        output_root=OUTPUT_ROOT,
        vis_level=1,
    )