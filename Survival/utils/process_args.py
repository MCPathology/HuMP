import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--task', type=str, choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="hallmarks", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4000, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')
    parser.add_argument("--wsi_projection_dim", type=int, default=1)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="10foldcv", help='where are splits')
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv"], default='ce',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)

    #---> partial-modality training simulation
    # These options randomly hide modalities only during training. Validation and
    # test-time missingness are still controlled by --test_missing_mode below.
    parser.add_argument('--train_missing_mode', type=str, default='',
                        choices=['', 'P', 'O', 'C'],
                        help="Training-time partial-modality experiment. "
                             "'' = use per-modality probabilities below; "
                             "'P'/'O'/'C' = randomly drop only that modality.")
    parser.add_argument('--train_missing_prob', type=float, default=0.0,
                        help='Probability of dropping --train_missing_mode in each training sample. '
                             'Used only when --train_missing_mode is not empty.')
    parser.add_argument('--train_missing_p_prob', type=float, default=0.0,
                        help='Probability of dropping pathology during each training batch.')
    parser.add_argument('--train_missing_o_prob', type=float, default=0.0,
                        help='Probability of dropping omics during each training batch.')
    parser.add_argument('--train_missing_c_prob', type=float, default=0.0,
                        help='Probability of dropping clinical data during each training batch.')
    parser.add_argument('--train_missing_warmup_epochs', type=int, default=1,
                        help='Number of initial epochs trained with full modalities before applying random missingness.')
    parser.add_argument('--train_missing_allow_double', action='store_true', default=False,
                        help='Allow more than one modality to be dropped in a training batch. '
                             'By default, if multiple modalities are sampled, one is chosen at random.')
    parser.add_argument('--train_skip_imputation', action='store_true', default=True,
                        help='When a modality is dropped during training, SKIP it entirely '
                             '(no HGS imputation): the modality is removed from fusion + loss. '
                             'Inference is unaffected (always uses HGS chain). Default True.')
    parser.add_argument('--no_train_skip_imputation', dest='train_skip_imputation',
                        action='store_false',
                        help='Disable train-time skip: train-missing batches run HGS imputation '
                             'just like inference does.')

    #---> model related
    parser.add_argument('--fusion', type=str, default=None)
    parser.add_argument('--modality', type=str, default="wsi")
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    #---> missing-modality simulation at inference (validation/test)
    # Empty string  : full modality
    # 'P' / 'O' / 'C'      : single-modality missing (P=pathology, O=omics, C=clinical)
    # 'PO' / 'PC' / 'OC'   : double-modality missing (chain HGS imputation)
    parser.add_argument('--test_missing_mode', type=str, default='',
                        choices=['', 'P', 'O', 'C', 'PO', 'PC', 'OC',
                                 'OP', 'CP', 'CO'],   # accept both orderings
                        help="Inference-time missing modality. "
                             "'' = full, 'OC' = only P observed, etc.")

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args
