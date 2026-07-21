#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
import warnings
warnings.filterwarnings('ignore')
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val, _infer_only
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args

torch.multiprocessing.set_sharing_strategy('file_system')

def _main(args):

    #----> prep for k-fold cv study
    folds = _get_start_end(args)

    #----> per-fold metric containers
    all_val_cindex      = []
    all_val_cindex_ipcw = []
    all_val_BS          = []
    all_val_IBS         = []
    all_val_iauc        = []
    all_val_loss        = []
    all_val_t_auc_1     = []
    all_val_t_auc_2     = []
    all_val_t_auc_3     = []
    all_val_t_auc_mean  = []
    fold_ids            = []

    if args.study == "tcga_":
        sp = [2, 3, 4]
    else:
        sp = [0, 1, 2, 3, 4]

    for i in sp:

        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        print("Created train and val datasets for fold {}".format(i))

        results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc,
                  total_loss, t_auc_scores) = _train_val(datasets, i, args)

        # ---- per-fold accounting (incl. t-AUC) ----
        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
        t1 = float(t_auc_scores.get('t_auc_1_year', np.nan))
        t2 = float(t_auc_scores.get('t_auc_2_year', np.nan))
        t3 = float(t_auc_scores.get('t_auc_3_year', np.nan))
        all_val_t_auc_1.append(t1)
        all_val_t_auc_2.append(t2)
        all_val_t_auc_3.append(t3)
        all_val_t_auc_mean.append(np.nanmean([t1, t2, t3]))
        fold_ids.append(i)

        # ---- write per-fold results to pkl (was previously printed only) ----
        pkl_path = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print(f"Saving results -> {pkl_path}")
        _save_pkl(pkl_path, {
            'results':         results,
            'val_cindex':      val_cindex,
            'val_cindex_ipcw': val_cindex_ipcw,
            'val_BS':          val_BS,
            'val_IBS':         val_IBS,
            'val_iauc':        val_iauc,
            'val_t_auc_1y':    t1,
            'val_t_auc_2y':    t2,
            'val_t_auc_3y':    t3,
            'val_t_auc_mean':  all_val_t_auc_mean[-1],
            'total_loss':      total_loss,
        })

    # =========================================================================
    #   k-fold AGGREGATE: per-fold + mean / std rows, written to CSV
    # =========================================================================
    per_fold_df = pd.DataFrame({
        'fold'         : fold_ids,
        'val_cindex'   : all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS'      : all_val_IBS,
        'val_iauc'     : all_val_iauc,
        't_auc_1y'     : all_val_t_auc_1,
        't_auc_2y'     : all_val_t_auc_2,
        't_auc_3y'     : all_val_t_auc_3,
        't_auc_mean'   : all_val_t_auc_mean,
        'val_loss'     : all_val_loss,
    })

    # numeric columns (everything except 'fold')
    metric_cols = [c for c in per_fold_df.columns if c != 'fold']

    summary_row_mean = {'fold': 'mean'}
    summary_row_std  = {'fold': 'std'}
    for c in metric_cols:
        v = pd.to_numeric(per_fold_df[c], errors='coerce').to_numpy()
        summary_row_mean[c] = float(np.nanmean(v))
        summary_row_std[c]  = float(np.nanstd(v))

    final_df = pd.concat(
        [per_fold_df, pd.DataFrame([summary_row_mean, summary_row_std])],
        ignore_index=True
    )

    summary_csv = os.path.join(args.results_dir, 'summary_kfold.csv')
    final_df.to_csv(summary_csv, index=False)
    print("\n" + "=" * 72)
    print(f"K-fold summary written to: {summary_csv}")
    print("=" * 72)
    print(final_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)

    return final_df



def main(data):
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    
    args.study = f"tcga_{data}"
    args.data_root_dir = f"WSIdata/{data}"
    args.label_file = f"datasets_csv/metadata/tcga_{data}.csv"
    args.omics_dir = f"datasets_csv/raw_rna_data/combine/{data}"
    
    args.results_dir = f"results_{data}"
    args = _prepare_for_experiment(args)

    #----> create dataset factory
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
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True , 
        type_of_pathway=args.type_of_path)
    args.memory_name = 'memory/c_1.h5'
    #---> perform the experiment
    results = _main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))

if __name__ == "__main__":
    for data in ['coadread', 'brca', 'hnsc', 'stad' , 'blca']: #'coadread', 'brca', 'hnsc', 'stad' , 'blca', 
        main(data)
