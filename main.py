"""HuMP entry point.

Usage examples
--------------

Train (5-fold cross-validation)::

    python main.py --study tcga_brca --modality hump \
        --data_root_dir /path/to/wsi_features \
        --results_dir ./results

Inference with a saved checkpoint::

    python main.py --study tcga_brca --modality hump --phase test \
        --ckpt_path /path/to/checkpoint.pt \
        --attn_save_root ./heat_pt
"""

import os
import warnings
from timeit import default_timer as timer

import torch

from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val, _infer_only
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
from utils.process_args import _process_args

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def _main(args):
    folds = _get_start_end(args)

    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    if args.study == "tcga_":
        sp = [2, 3, 4]
    else:
        sp = [0, 1, 2, 3, 4]

    for i in sp:
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path="{}/splits_{}.csv".format(args.split_dir, i),
            fold=i,
        )
        print("Created train and val datasets for fold {}".format(i))

        if getattr(args, "phase", "train") == "test":
            # Inference path: load a saved checkpoint and optionally dump
            # attention / gradient maps for visualization.
            ckpt_path = args.ckpt_path
            if ckpt_path is None:
                raise ValueError(
                    "--ckpt_path must be provided when --phase test."
                )
            results, metrics = _infer_only(
                datasets,
                i,
                args,
                ckpt_path,
                attn_save_root=getattr(args, "attn_save_root", "./heat_pt"),
                save_attn_heatmap=getattr(args, "save_attn_heatmap", False),
                save_clinic_attr=getattr(args, "save_clinic_attr", False),
                save_molecular_group=getattr(args, "save_molecular_group", False),
            )
        else:
            # Training path.
            results, metrics = _train_val(datasets, i, args)

        val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = metrics
        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)

        # Persist per-fold results for offline aggregation.
        out_path = os.path.join(args.results_dir, "split_{}_results.pkl".format(i))
        try:
            _save_pkl(out_path, results)
        except Exception as e:
            print("[warn] failed to save {}: {}".format(out_path, e))

    print("\n=== Cross-validation summary ===")
    print("C-Index : mean={:.4f}".format(sum(all_val_cindex) / len(all_val_cindex)))


def main(data):
    start = timer()

    args = _process_args()

    args.study = f"tcga_{data}"

    # Defaults can be overridden via CLI flags. They are kept as sane fall-backs
    # so the script remains runnable from a fresh checkout.
    if not getattr(args, "data_root_dir", None):
        args.data_root_dir = os.environ.get(
            "HUMP_DATA_ROOT", f"./data/{data}/wsi_features"
        )
    if not getattr(args, "label_file", None):
        args.label_file = f"datasets_csv/metadata/tcga_{data}.csv"
    if not getattr(args, "omics_dir", None):
        args.omics_dir = f"datasets_csv/raw_rna_data/combine/{data}"
    if not getattr(args, "results_dir", None):
        args.results_dir = f"results_{data}"

    args = _prepare_for_experiment(args)

    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat=True if "coattn" in args.modality else False,
        is_survpath=True,
        type_of_pathway=args.type_of_path,
    )
    args.memory_name = "memory/c_1.h5"

    _main(args)

    end = timer()
    print("finished!")
    print("Script Time: %f seconds" % (end - start))


if __name__ == "__main__":
    # By default we iterate over BRCA only; pass `--study tcga_<cohort>` and
    # extend this loop, or wrap a shell driver, to run multiple cohorts.
    for data in ["brca"]:  # 'coadread', 'brca', 'hnsc', 'stad', 'blca'
        main(data)
