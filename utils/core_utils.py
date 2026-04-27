import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from custom_optims.radam import RAdam
from models.model_HuMP import HuMP
from sklearn.metrics import roc_auc_score
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    brier_score,
    integrated_brier_score,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss



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
        from custom_optims.lamb import Lamb
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):

    print('\nInit Model...', end=' ')

    model_dict = {
        'fusion':          args.fusion,
        'genomic_sizes':   args.genomic_sizes,
        'transomic_sizes': args.transomic_sizes,
        'n_classes':       args.n_classes,
        'model_size':      'small',
    }
    model = HuMP(**model_dict)

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
    Unpack a data batch from the HuMP dataloader and move tensors to device.

    Data layout (set by the HuMP collate function):
        data[0] : WSI patch features  [N_patch, D]
        data[1] : protein features    (or None)
        data[2] : list of genomic group tensors
        data[3] : (list of transcriptomic group tensors,)
        data[4] : clinical feature tensor  [M, 512]
        data[5] : y_disc
        data[6] : event_time
        data[7] : censor
        data[8] : clinical_data_list
        data[9] : patient ID

    Returns:
        data_WSI, protein, y_disc, event_time, censor,
        genomics, transomics, clinical_features,
        clinical_data_list, patient
    """
    data_WSI         = data[0].to(device)
    protein          = data[1]
    genomics         = [item.to(device) for item in data[2]]
    transomics       = [item.to(device) for item in data[3][0]]
    clinical_features = data[4].to(device)
    y_disc, event_time, censor, clinical_data_list, patient = (
        data[5], data[6], data[7], data[8], data[9]
    )

    y_disc     = y_disc.to(device)
    event_time = event_time.to(device)
    censor     = censor.to(device)

    return data_WSI, protein, y_disc, event_time, censor, genomics, transomics, clinical_features, clinical_data_list, patient

def _process_data_and_forward(model, modality, device, data):
    r"""
    Unpack a batch, build the HuMP input dict, and run a forward pass.

    Args:
        - model    : HuMP instance
        - modality : String (expected: "hump")
        - device   : torch.device
        - data     : tuple from the dataloader

    Returns:
        - out               : torch.Tensor  logits [1, n_classes]
        - y_disc            : torch.Tensor
        - event_time        : torch.Tensor
        - censor            : torch.Tensor
        - clinical_data_list: List
        - dist              : scalar loss from the model
        - proto_features_dict: dict of prototype tensors
    """
    data_WSI, protein, y_disc, event_time, censor, genomics, transomics, clinical_features, clinical_data_list, _ = \
        _unpack_data(modality, device, data)

    input_args = {
        'x_path':    data_WSI,
        'protein':   protein,
        'valid':     False,
        'report':    clinical_features,
        'prototype': None,
    }
    for i in range(len(genomics)):
        input_args['x_genomic%s' % str(i + 1)] = genomics[i].type(torch.FloatTensor).to(device)
    for i in range(len(transomics)):
        input_args['x_transomic%s' % str(i + 1)] = transomics[i].type(torch.FloatTensor).to(device)

    out, dist, proto_features_dict = model(**input_args)

    if len(out.shape) == 1:
        out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list, dist, proto_features_dict


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

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    proto_G_list, proto_P_list, proto_C_list = [], [], []
    simulate_missing = True
    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc, event_time, censor, clinical_data_list, dist, proto_dict = _process_data_and_forward(model, modality, device, data)
        # simulate modality missing ,calculate prototype
        if simulate_missing and proto_dict is not None:
            # if 'G' in proto_dict:
            #     proto_G_list.append(proto_dict['G'].detach().cpu())
            # if 'P' in proto_dict:
            #     proto_P_list.append(proto_dict['P'].detach().cpu())
            # if 'C' in proto_dict:
            #     proto_C_list.append(proto_dict['C'].detach().cpu())
            proto_G_list.append(proto_dict)

        #print()
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        #print(loss)
        loss_value = loss.item()
        alpha = 1.0
        loss = loss / y_disc.shape[0] + dist*alpha
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        scheduler.step()
        
        if (batch_idx % 100) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    if simulate_missing and len(proto_G_list) > 0:
        proto_avg_dict = {}
        # proto_avg_dict['G'] = torch.mean(torch.stack(proto_G_list, dim=0), dim=0)  # [1, N, D]
        # proto_avg_dict['P'] = torch.mean(torch.stack(proto_P_list, dim=0), dim=0)
        # proto_avg_dict['C'] = torch.mean(torch.stack(proto_C_list, dim=0), dim=0)
    else:
        proto_avg_dict = None

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss, proto_G_list# proto_avg_dict

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
    """Load pretrained checkpoint weights into the model."""
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
    Run a validation / inference loop and return survival metrics.

    Missing-modality simulation is handled transparently: if a sample's
    data tensor is None (e.g. x_path=None), the model's HGS completion
    branch is triggered automatically inside forward().

    Args:
        - dataset_factory : SurvivalDatasetFactory
        - model           : Pytorch model
        - proto_list      : unused; kept for API compatibility
        - modality        : String
        - loader          : Pytorch DataLoader
        - loss_fn         : NLLSurvLoss instance
        - memory          : unused; kept for API compatibility
        - survival_train  : np.array from _extract_survival_metadata

    Returns:
        - patient_results  : dict
        - c_index          : Float
        - c_index_ipcw     : Float
        - all_risk_scores  : np.ndarray
        - all_event_times  : np.ndarray
        - all_censorships  : np.ndarray
        - BS               : np.ndarray
        - IBS              : Float
        - iauc             : Float
        - total_loss       : Float
        - t_auc_scores     : dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    with torch.no_grad():
        for data in loader:
            data_WSI, protein, y_disc, event_time, censor, genomics, transomics, clinical_features, clinical_data_list, patient = \
                _unpack_data(modality, device, data)

            input_args = {
                'x_path':    data_WSI,
                'protein':   protein,
                'valid':     True,
                'report':    clinical_features,
                'prototype': None,
            }
            for i in range(len(genomics)):
                input_args['x_genomic%s' % str(i + 1)] = genomics[i].type(torch.FloatTensor).to(device)
            for i in range(len(transomics)):
                input_args['x_transomic%s' % str(i + 1)] = transomics[i].type(torch.FloatTensor).to(device)

            h, _, _ = model(**input_args)
            h = h.detach()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)

            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data,
                event_time, censor, risk, clinical_data_list,
            )
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
    for year, time_point in zip([1, 2, 3], [12, 24, 36]):
        try:
            t_auc_scores[f't_auc_{year}_year'] = calculate_t_auc(
                predict_probs=all_risk_scores,
                event_times=all_event_times,
                event_indicators=all_censorships,
                target_time=time_point,
            )
        except ValueError as e:
            print(f"cannot compute {year}-yr t-AUC: {e}")
            t_auc_scores[f't_auc_{year}_year'] = np.nan

    patient_results = {}
    for i in range(len(all_slide_ids)):
        case_id = slide_ids.values[i][:12]
        patient_results[case_id] = {
            "time":        all_event_times[i],
            "risk":        all_risk_scores[i],
            "censorship":  all_censorships[i],
            "clinical":    all_clinical_data[i],
            "logits":      all_logits[i],
        }

    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(
        loader, dataset_factory, survival_train,
        all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores,
    )

    return patient_results, c_index, c_index2, all_risk_scores, all_event_times, all_censorships, BS, IBS, iauc, total_loss, t_auc_scores

def _infer_only(datasets, cur, args, ckpt_path,
                attn_save_root="./heat_pt",
                save_attn_heatmap=False,
                save_clinic_attr=False,
                save_molecular_group=True):
    """Load a checkpoint and run inference, optionally saving:

      * per-modality attention heatmaps    (``save_attn_heatmap=True``)
      * clinical-column gradient attribution (``save_clinic_attr=True``)
      * per-group molecular heatmap + per-gene attribution
        (``save_molecular_group=True``)
    """
    train_split, val_split = _get_splits(datasets, cur, args)
    loss_fn = _init_loss_function(args)

    model = _init_model(args)
    model = _load_checkpoint(model, ckpt_path)
    model.eval()

    # External switches consumed by the model's optional visualization hooks.
    model._save_attn_flag = bool(save_attn_heatmap)
    model._save_clinic_attr_flag = bool(save_clinic_attr)
    model._save_molecular_group_flag = bool(save_molecular_group)
    model._attn_save_root = attn_save_root
    os.makedirs(attn_save_root, exist_ok=True)

    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    all_survival = _extract_survival_metadata(train_loader, val_loader)

    results_dict, val_cindex, val_cindex_ipcw, risk, event, censor, \
        val_BS, val_IBS, val_iauc, total_loss, t_auc_scores = _summary(
            args.dataset_factory, model, None, args.modality,
            val_loader, loss_fn, None, all_survival
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

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

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
    best_val_index = 0
    for epoch in range(args.max_epochs):
        _, _, proto_list = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        _, val_cindex, _, risk, event, censor, _, _, _, total_loss, t_auc_scores = _summary(args.dataset_factory, model, None, args.modality, val_loader, loss_fn, None, all_survival)
        print(f"Val loss: {total_loss:.4f}, C-Index: {val_cindex:.4f}, "
            f"t-AUC (1/2/3yr): {t_auc_scores.get('t_auc_1_year', 0):.4f}/"
            f"{t_auc_scores.get('t_auc_2_year', 0):.4f}/"
            f"{t_auc_scores.get('t_auc_3_year', 0):.4f}")
        if val_cindex>best_val_index:
            best_val_index=val_cindex
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_{}_{:.4f}checkpoint.pt".format(cur, args.study, best_val_index)))
   
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    results_dict, val_cindex, val_cindex_ipcw, _, _, _,val_BS, val_IBS, val_iauc, total_loss, _ = _summary(args.dataset_factory, model, proto_list, args.modality, val_loader, loss_fn, None, all_survival)
    
    print('Final Val c-index: {:.4f}'.format(best_val_index))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

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
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
