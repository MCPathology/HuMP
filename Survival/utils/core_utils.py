from ast import Lambda
import numpy as np
import pdb
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from custom_optims.radam import RAdam
try:
    from models.model_ABMIL import ABMIL
    from models.model_DeepMISL import DeepMISL
    from models.model_MLPOmics import MLPOmics
    from models.model_MLPWSI import MLPWSI
    from models.model_SNNOmics import SNNOmics
    from models.model_MaskedOmics import MaskedOmics
    from models.model_MCATPathways import MCATPathways
    from models.model_SurvPath import SurvPath
    from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
except ImportError:
    ABMIL = DeepMISL = MLPOmics = MLPWSI = SNNOmics = None
    MaskedOmics = MCATPathways = SurvPath = SurvPath_with_nystrom = None
# from models.model_OmniSurv import OmniSurv

# from models.model_motcat import MCATPathwaysMotCat
from models.model_HGNN import MHSurv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from scipy.ndimage import convolve
from sksurv.util import Surv
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
except ImportError:
    scienceplots = None

import numpy as np
from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score
#torch.autograd.set_detect_anomaly(True)

#----> pytorch imports
import torch

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    elif args.modality == "hgnn":

        model_dict = {
            'fusion': args.fusion, 'genomic_sizes': args.genomic_sizes, 'transomic_sizes': args.transomic_sizes, 'n_classes': args.n_classes, "model_size": "small",
        }
        model = MHSurv(**model_dict) # MHSurv(**model_dict)

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """
    
    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat", "hgnn"]:
        
        data_WSI = data[0].to(device)
        # print(data[1])
        protein = data[1]
        genomics = []
        transomics = []
        for item in data[2]:
            genomics.append(item.to(device))
        for item in data[3][0]:
            transomics.append(item.to(device))
        '''data_omic1 = data[3].type(torch.FloatTensor).to(device)
        data_omic2 = data[4].type(torch.FloatTensor).to(device)
        data_omic3 = data[5].type(torch.FloatTensor).to(device)
        data_omic4 = data[6].type(torch.FloatTensor).to(device)
        data_omic5 = data[7].type(torch.FloatTensor).to(device)
        data_omic6 = data[8].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]'''
        table_features = data[4].to(device)
        y_disc, event_time, censor, clinical_data_list, patient = data[5], data[6], data[7], data[8], data[9]
        # mask = mask.to(device)

    elif modality in ["survpath"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, protein, y_disc, event_time, censor, genomics, transomics, table_features, clinical_data_list, patient

def _process_data_and_forward(model, modality, device, data, missing_mode=''):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    data_WSI, protein, y_disc, event_time, censor, genomics, transomics, table_features, clinical_data_list, _ = _unpack_data(modality, device, data)
    
    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=genomics[0], 
            x_omic2=genomics[1], 
            x_omic3=genomics[2], 
            x_omic4=genomics[3], 
            x_omic5=genomics[4], 
            x_omic6=genomics[5]
            )  
    elif modality == "hgnn":
        input_args = {'x_path': data_WSI, 'protein':protein, "valid":False, "report":table_features}
        if missing_mode:
            input_args['missing_mode'] = missing_mode
        # Partial-training: when set, the model SKIPS the missing modality
        # (no HGS imputation, no compensating tokens).  Inference path is
        # unaffected — `model._test_missing_mode` is the inference channel.
        input_args['skip_imputation'] = getattr(model, '_skip_imputation_on_missing', False)

        for i in range(len(genomics)):
            input_args['x_genomic%s' % str(i+1)] = genomics[i].type(torch.FloatTensor).to(device)
        for i in range(len(transomics)):
            input_args['x_transomic%s' % str(i+1)] = transomics[i].type(torch.FloatTensor).to(device)
        out, dist, proto_features_dict = model(**input_args)   
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list, dist, proto_features_dict


def _sample_train_missing_mode(args, epoch):
    """Sample a train-time missing-modality mode for partial-modality training."""
    if epoch < getattr(args, 'train_missing_warmup_epochs', 0):
        return ''

    fixed_mode = (getattr(args, 'train_missing_mode', '') or '').upper()
    fixed_prob = getattr(args, 'train_missing_prob', 0.0)
    if fixed_mode:
        return fixed_mode if np.random.rand() < fixed_prob else ''

    probs = {
        'P': getattr(args, 'train_missing_p_prob', 0.0),
        'O': getattr(args, 'train_missing_o_prob', 0.0),
        'C': getattr(args, 'train_missing_c_prob', 0.0),
    }
    dropped = [m for m, p in probs.items() if p > 0 and np.random.rand() < p]
    if not dropped:
        return ''

    if len(dropped) > 1 and not getattr(args, 'train_missing_allow_double', False):
        dropped = [np.random.choice(dropped)]

    # Never train with all three modalities missing; there is no observed signal.
    if len(dropped) >= 3:
        dropped = list(np.random.choice(dropped, size=2, replace=False))

    return ''.join(dropped)


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, args=None):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    total_loss = 0.

    # NaN-protection counters (printed at epoch end for diagnostics)
    n_accum         = 0
    n_skipped_logit = 0
    n_skipped_loss  = 0
    n_skipped_grad  = 0
    train_missing_counts = {}

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    proto_G_list, proto_P_list, proto_C_list = [], [], []
    simulate_missing = True

    # ----- helper for safe scalarisation of dist (entailment) loss -----
    def _safe_scalar(x):
        if isinstance(x, torch.Tensor):
            return x if torch.isfinite(x).all() else torch.zeros((), device=x.device)
        try:
            return float(x) if np.isfinite(x) else 0.0
        except Exception:
            return 0.0

    # one epoch
    for batch_idx, data in enumerate(loader):

        # ★ clean grads at the start of every batch (set_to_none is more robust)
        optimizer.zero_grad(set_to_none=True)

        train_missing_mode = ''
        if args is not None and modality == 'hgnn':
            train_missing_mode = _sample_train_missing_mode(args, epoch)
            train_missing_counts[train_missing_mode or 'full'] = (
                train_missing_counts.get(train_missing_mode or 'full', 0) + 1
            )

        h, y_disc, event_time, censor, clinical_data_list, dist, proto_dict = \
            _process_data_and_forward(model, modality, device, data, missing_mode=train_missing_mode)

        # ---- (1) logits non-finite -> skip whole batch ----
        if not torch.isfinite(h).all():
            n_skipped_logit += 1
            print(f"[batch {batch_idx}] logits non-finite, skip")
            del h
            continue

        # collect prototypes for missing-modality simulation
        if simulate_missing and proto_dict is not None:
            proto_G_list.append(proto_dict)

        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)

        # ---- (2) survival loss non-finite -> skip ----
        if not torch.isfinite(loss):
            n_skipped_loss += 1
            print(f"[batch {batch_idx}] surv loss non-finite ({loss.item()}), skip")
            del h, loss
            continue

        loss_value = loss.item()
        alpha = 1.0
        # ensure dist scalar is finite before adding
        dist_safe = _safe_scalar(dist)
        loss = loss / y_disc.shape[0] + dist_safe * alpha

        # ---- (3) combined loss non-finite (e.g. dist NaN) -> skip ----
        if not torch.isfinite(loss):
            n_skipped_loss += 1
            print(f"[batch {batch_idx}] combined loss non-finite, skip")
            del h, loss
            continue

        # backward
        loss.backward()

        # ---- (4) any param's gradient non-finite -> skip step ----
        finite_grads = all(
            (p.grad is None) or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not finite_grads:
            n_skipped_grad += 1
            print(f"[batch {batch_idx}] non-finite grad, skip step")
            optimizer.zero_grad(set_to_none=True)
            del h, loss
            continue

        # ★ gradient clipping (critical: prevents Adam state poisoning)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # only after a successful step do we record metrics
        risk, _ = _calculate_risk(h)
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times,
            all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value
        n_accum += 1

        if (batch_idx % 100) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))

        del h, loss

    # =================================================================
    #   epoch end: NaN-safe aggregation
    # =================================================================
    if n_accum == 0:
        msg = (f"[Epoch {epoch}] all batches skipped "
               f"(logits={n_skipped_logit}, loss={n_skipped_loss}, "
               f"grad={n_skipped_grad}). Returning 0.")
        print(msg)
        return 0.0, 0.0, proto_G_list

    if any(k != 'full' for k in train_missing_counts):
        print(f"[Epoch {epoch}] train missing-mode counts: {train_missing_counts}")

    total_loss /= max(n_accum, 1)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)

    # mask any residual NaN/Inf rows before sksurv
    finite_mask = (np.isfinite(all_risk_scores)
                   & np.isfinite(all_censorships)
                   & np.isfinite(all_event_times))
    n_total = len(all_risk_scores)
    n_bad   = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"[Epoch {epoch}] masking {n_bad}/{n_total} non-finite samples "
              f"before C-Index computation")
        all_risk_scores = all_risk_scores[finite_mask]
        all_censorships = all_censorships[finite_mask]
        all_event_times = all_event_times[finite_mask]

    if len(all_risk_scores) == 0:
        print(f"[Epoch {epoch}] no finite samples left, C-Index = 0")
        return 0.0, total_loss, proto_G_list

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08,
    )[0]

    if simulate_missing and len(proto_G_list) > 0:
        proto_avg_dict = {}
    else:
        proto_avg_dict = None

    print(f"Epoch: {epoch}, train_loss: {total_loss:.4f}, "
          f"train_c_index: {c_index:.4f}  "
          f"(skipped: logits={n_skipped_logit}, loss={n_skipped_loss}, "
          f"grad={n_skipped_grad}, masked={n_bad}/{n_total})")

    return c_index, total_loss, proto_G_list  # proto_avg_dict

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc
    
def calculate_t_auc(
        predict_probs: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        target_time: float = None
) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the survival model.

    Parameters
    ----------
    predict_probs: np.ndarray
        The predicted survival probabilities
    event_times: np.ndarray
        The event or censoring times for the test data
    event_indicators: np.ndarray
        The binary indicators of whether the event occurred (1) or was censored (0)
    target_time: float, optional
        The specific time point at which to calculate the AUC. If not specified, the median of the event times is used.

    Returns
    -------
    auc: float
        The AUC value calculated at the specified target time.
    """
    # if the target time is not specified, then we use the median of the event times
    if target_time is None:
        target_time = np.median(event_times)

    # for censored data, if the censor time is earlier than the target time,
    # (since we cannot observe the real status at the target time)
    # then we just exclude its prediction and observation from the calculation
    exclude_indicators = np.logical_and(event_times < target_time, event_indicators == 0)
    event_times = event_times[~exclude_indicators]
    predict_probs = predict_probs[~exclude_indicators]

    # get the binary status of the test data, given the target time
    binary_status = (event_times <= target_time).astype(int)

    # check if the binary status is all zeros or all ones
    if np.all(binary_status == 0) or np.all(binary_status == 1):
        raise ValueError(f"Survival status is all zeros or all ones at time: {target_time}, AUC cannot be computed.")

    # computing the AUC, given the predicted probabilities and the binary status
    risks = 1 - predict_probs
    return roc_auc_score(binary_status, risks)

def _load_checkpoint(model, ckpt_path):
    """加载训练好的 checkpoint 权重"""
    print(f'\nLoading checkpoint from: {ckpt_path}')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dict = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
    print('Checkpoint loaded.')
    return model

def _summary(dataset_factory, model, proto_list, modality, loader, loss_fn, memory, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    # 注意: 不再使用全局 torch.no_grad(), 因为 hgnn+clinic 归因需要梯度.
    # 非归因情况下在 batch 内部局部 no_grad, 语义等价, 开销可忽略.
    for data in loader:

        data_WSI, protein, y_disc, event_time, censor, genomics, transomics, report_features, clinical_data_list, patient= _unpack_data(modality, device, data)

        if modality == "hgnn":
            if data_WSI.shape[1] == 40000:
                count += 1
                continue

            save_attn = getattr(model, "_save_attn_flag", False)
            save_clinic_attr = getattr(model, "_save_clinic_attr_flag", False)
            save_mol_group = getattr(model, "_save_molecular_group_flag", False)

            # 准备 x_table_raw (leaf + requires_grad) 仅在做 clinic 归因时需要
            x_table_raw = None
            if save_clinic_attr:
                x_table_raw = report_features
                if x_table_raw.dim() == 3:
                    x_table_raw = x_table_raw.squeeze(0)
                x_table_raw = x_table_raw.detach().clone().requires_grad_(True)

            # 准备 genomics/transomics/protein 原始输入 (leaf + requires_grad)
            # 仅在做分子三组归因时需要
            genomics_raw = None
            transomics_raw = None
            protein_raw = None
            if save_mol_group:
                genomics_raw = [g.detach().clone().requires_grad_(True) for g in genomics]
                transomics_raw = [t.detach().clone().requires_grad_(True) for t in transomics]
                if protein is not None:
                    if isinstance(protein, torch.Tensor):
                        protein_raw = protein.detach().clone().requires_grad_(True)
                    else:
                        protein_raw = protein.to(device).detach().clone().requires_grad_(True)

            input_args = {
                'x_path': data_WSI,
                'protein': protein_raw if (save_mol_group and protein_raw is not None) else protein,
                "valid": True,
                'report': x_table_raw if x_table_raw is not None else report_features,
                'patient': patient,
            }

            # 如果做分子归因, 用 requires_grad 版本替换原始输入
            if save_mol_group and genomics_raw is not None:
                for i in range(len(genomics_raw)):
                    input_args['x_genomic%s' % str(i+1)] = genomics_raw[i].type(torch.FloatTensor).to(device)
                for i in range(len(transomics_raw)):
                    input_args['x_transomic%s' % str(i+1)] = transomics_raw[i].type(torch.FloatTensor).to(device)
                input_args['genomics_raw'] = genomics_raw
                input_args['transomics_raw'] = transomics_raw
                input_args['protein_raw'] = protein_raw
            else:
                for i in range(len(genomics)):
                    input_args['x_genomic%s' % str(i+1)] = genomics[i].type(torch.FloatTensor).to(device)
                for i in range(len(transomics)):
                    input_args['x_transomic%s' % str(i+1)] = transomics[i].type(torch.FloatTensor).to(device)
                # ★ removed:  input_args['x_genomic1'] = None
                # 该硬编码与 missing_mode 冲突,导致 'OC'='PC'='QC' 结果一致.
                # 推理阶段的缺失模拟现在统一由 missing_mode 控制 (见下).

            # ★ 推理期缺失模态控制 (替代原来的硬编码 x_genomic1=None)
            input_args['missing_mode'] = getattr(model, '_test_missing_mode', '')
            # Inference NEVER skips imputation — always run HGS chain on val.
            input_args['skip_imputation'] = False

            # ★ HGS prototype: 优先用 model 上累积的全局 prototype, 否则 None
            #    (lhyperbolic.py 已加 None -> zero-tensor fallback)
            input_args['prototype'] = getattr(model, '_global_prototype', None)

            # 统一 patient_id
            if save_attn or save_clinic_attr or save_mol_group:
                pid = patient
                if isinstance(pid, (list, tuple)):
                    pid = pid[0]
                if hasattr(pid, "item"):
                    pid = pid.item()
                pid = str(pid)
                input_args['patient_id'] = pid
                input_args['attn_save_root'] = getattr(model, "_attn_save_root", "./heat_pt")

            if save_attn:
                input_args['return_attn_maps'] = True
            if save_clinic_attr:
                input_args['return_clinic_attr'] = True
                input_args['x_table_raw'] = x_table_raw
            if save_mol_group:
                input_args['return_molecular_group_maps'] = True

            # forward: 需归因则开梯度, 否则 no_grad
            if save_clinic_attr or save_mol_group:
                with torch.enable_grad():
                    h, _, _ = model(**input_args)
            else:
                with torch.no_grad():
                    h, _, _ = model(**input_args)

            # 清理梯度, 避免累积到下一个 patient
            model.zero_grad(set_to_none=True)

        else:
            with torch.no_grad():
                h = model(
                    data_omics=data_omics,
                    data_WSI=data_WSI,
                    mask=mask
                )

        # 后续统计一律 detach, 避免无意中累计计算图内存
        h = h.detach()

        if len(h.shape) == 1:
            h = h.unsqueeze(0)
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]

        risk, risk_by_bin = _calculate_risk(h)
        all_risk_by_bin_scores.append(risk_by_bin)
        all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
        all_logits.append(h.detach().cpu().numpy())
        total_loss += loss_value
        all_slide_ids.append(slide_ids.values[count])
        count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    t_auc_scores = {}
    target_times_in_days = [12 * 1, 12 * 2, 12 * 3]
    target_years = [1, 2, 3]

    for year, time_point in zip(target_years, target_times_in_days):
        try:
            auc_value = calculate_t_auc(
                predict_probs=all_risk_scores,
                event_times=all_event_times,
                event_indicators=all_censorships,
                target_time=time_point
            )
            t_auc_scores[f't_auc_{year}_year'] = auc_value
        except ValueError as e:
            print(f"cannot compute {year} t-AUC: {e}")
            t_auc_scores[f't_auc_{year}_year'] = np.nan
            
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, all_risk_scores, all_event_times, all_censorships, BS, IBS, iauc, total_loss, t_auc_scores

def _infer_only(datasets, cur, args, ckpt_path,
                attn_save_root="./heat_pt",
                save_attn_heatmap=False,
                save_clinic_attr=False,
                save_molecular_group=True):
    """
    加载 checkpoint 直接推理, 可选地保存:
      - 三模态 attention heatmap    (save_attn_heatmap=True)
      - clinic 列级梯度归因          (save_clinic_attr=True)
      - 分子三组 heatmap + 基因归因  (save_molecular_group=True)
    """
    train_split, val_split = _get_splits(datasets, cur, args)
    loss_fn = _init_loss_function(args)

    model = _init_model(args)
    model = _load_checkpoint(model, ckpt_path)
    model.eval()

    # 外挂开关
    model._save_attn_flag = bool(save_attn_heatmap)
    model._save_clinic_attr_flag = bool(save_clinic_attr)
    model._save_molecular_group_flag = bool(save_molecular_group)
    model._attn_save_root = attn_save_root
    os.makedirs(attn_save_root, exist_ok=True)

    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    all_survival = _extract_survival_metadata(train_loader, val_loader)

    memory = None
    results_dict, val_cindex, val_cindex_ipcw, risk, event, censor, \
        val_BS, val_IBS, val_iauc, total_loss, t_auc_scores = _summary(
            args.dataset_factory, model, None, args.modality,
            val_loader, loss_fn, memory, all_survival
        )

    print(f"\n=== Inference Done (fold {cur}) ===")
    print(f"Val loss       : {total_loss:.4f}")
    print(f"Val C-Index    : {val_cindex:.4f}")
    print(f"Val C-Index IPCW: {val_cindex_ipcw:.4f}")
    print(f"Val IBS        : {val_IBS:.4f}")
    print(f"Val iAUC       : {val_iauc:.4f}")
    print(f"t-AUC (1/2/3yr): {t_auc_scores.get('t_auc_1_year', 0):.4f} / "
          f"{t_auc_scores.get('t_auc_2_year', 0):.4f} / "
          f"{t_auc_scores.get('t_auc_3_year', 0):.4f}")
    if save_attn_heatmap:
        print(f"Attention maps saved under: {attn_save_root}/(gene|path|clinic)/*.pt")
    if save_clinic_attr:
        print(f"Clinic column attributions saved under: {attn_save_root}/clinic_col_attr/*.pt")
    if save_molecular_group:
        print(f"Molecular group maps saved under: {attn_save_root}/(genomics|transomics|protein)/*.pt")

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, t_auc_scores)

def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_val_index = -np.inf
    best_epoch = -1
    best_state_dict = None
    best_global_prototype = None
    memory = None # MemoryBank(args.memory_name, theta = 0.75)
    # memory.clear()
    best_p = 1

    # ★ propagate inference-time missing-modality switch to the model so that
    #   _summary can read it via getattr(model, '_test_missing_mode', '').
    model._test_missing_mode = getattr(args, 'test_missing_mode', '')
    # ★ partial-training: when train_missing_* args drop a modality, the model
    #   should SKIP it (not HGS-impute it).  Inference still imputes regardless.
    model._skip_imputation_on_missing = getattr(args, 'train_skip_imputation', True)

    for epoch in range(args.max_epochs):
        _, _, proto_list = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args=args)
        _, val_cindex, _, risk, event, censor, _, _, _, total_loss, t_auc_scores = _summary(args.dataset_factory, model, None, args.modality, val_loader, loss_fn, memory, all_survival)
        print(f"Val loss: {total_loss:.4f}, C-Index: {val_cindex:.4f}, "
            f"t-AUC (1/2/3yr): {t_auc_scores.get('t_auc_1_year', 0):.4f}/"
            f"{t_auc_scores.get('t_auc_2_year', 0):.4f}/"
            f"{t_auc_scores.get('t_auc_3_year', 0):.4f}")
        if val_cindex>best_val_index:
            best_val_index=val_cindex
            best_epoch=epoch
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            global_proto = getattr(model, '_global_prototype', None)
            if isinstance(global_proto, dict):
                best_global_prototype = {
                    k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in global_proto.items()
                }
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_{}_{:.4f}checkpoint.pt".format(cur, args.study, best_val_index)))
        '''risk = np.abs(risk)
        threshold = np.median(risk)
        plt.style.use('science')
        #plt.rcParams.update({'font.size': plt.rcParams['font.size'] + 5})

        # Separate into high-risk and low-risk groups
        high_risk_mask = risk >= threshold
        low_risk_mask = risk < threshold

        # Extract data for each group
        high_risk_times = event[high_risk_mask]
        high_risk_events = 1-censor[high_risk_mask]

        low_risk_times = event[low_risk_mask]
        low_risk_events = 1-censor[low_risk_mask]
        
        high_risk_times = event[low_risk_mask]
        high_risk_events = 1-censor[low_risk_mask]

        low_risk_times = event[high_risk_mask]
        low_risk_events = 1-censor[high_risk_mask]

        # Create Kaplan-Meier fitter instances
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        #print(high_risk_events)
        # Fit the data
        kmf_high.fit(high_risk_times, event_observed=high_risk_events, label='HR')
        kmf_low.fit(low_risk_times, event_observed=low_risk_events, label='LR')

        # Perform Log-rank test to calculate p-value
        results = logrank_test(high_risk_times, low_risk_times,
                       event_observed_A=high_risk_events,
                       event_observed_B=low_risk_events)
 
        p_value = results.p_value
        if p_value<0.05:
            plt.figure(figsize=(10, 6))
            kmf_low.plot_survival_function(show_censors=True, color='green')
            kmf_high.plot_survival_function(show_censors=True, color='red',censor_styles={'marker': '+', 'ms': 15},)
            plt.text(0.97, 0.97, f'P-value: {p_value:.2e}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=13)
            plt.title('{}'.format(args.study[5:]))
            legend_font_size = plt.rcParams['font.size'] - 2  # Reduce by one point from current font size
            legend = plt.legend(loc='lower left', fancybox=True, framealpha=1,
                        fontsize=legend_font_size)
            # plt.ylim(0, 1)            
            plt.xlabel('time(month)')
            plt.savefig("km1/{}/s_{}_epoch{}_{:e}.jpg".format(args.study[5:], cur, epoch, p_value),dpi=600)
            print("save:{:e}".format(p_value))'''
    

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if best_global_prototype is not None:
            device = next(model.parameters()).device
            model._global_prototype = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in best_global_prototype.items()
            }

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # ★ keep t_auc_scores so it can be propagated up to main.py
    results_dict, val_cindex, val_cindex_ipcw, _, _, _, val_BS, val_IBS, val_iauc, total_loss, t_auc_scores = _summary(
        args.dataset_factory, model, proto_list, args.modality, val_loader, loss_fn, memory, all_survival
    )

    print('Best Val c-index: {:.4f} at epoch {}'.format(best_val_index, best_epoch))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, t_auc_scores)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss, t_auc_scores) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss, t_auc_scores)
