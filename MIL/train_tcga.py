import enum
import re
# from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
try:
    import redis
except Exception:
    redis = None
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
# torch.multiprocessing.set_sharing_strategy('file_system')
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR).lower() == 'mil':
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
else:
    PROJECT_ROOT = SCRIPT_DIR
PROJECT_PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))


def _resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _dataset_kind(dataset):
    key = dataset.upper()
    if key.startswith('TCGA') or key.startswith('NSCLC') or key.startswith('LUAD') or key.startswith('LUSC'):
        return 'tcga'
    if key.startswith('CCRCC') or key.startswith('CPTAC') or key.startswith('KIRC'):
        return 'ccrcc'
    return 'other'


def _patient_id_from_feature_path(feats_path, dataset):
    filename = os.path.basename(str(feats_path))
    if filename.endswith('.pt'):
        filename = filename[:-3]
    stem = os.path.splitext(filename)[0]
    kind = _dataset_kind(dataset)
    if kind == 'tcga':
        return stem[:12]
    if kind == 'ccrcc':
        return stem[:9]
    return stem


def _as_token_matrix(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 3 and x.size(0) == 1:
        x = x.squeeze(0)
    return x


def _load_feature_tensor(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return torch.from_numpy(np.load(path)).float()
    return torch.load(path).float()


def _default_gene_dir(dataset):
    kind = _dataset_kind(dataset)
    if kind == 'tcga':
        return os.path.join(PROJECT_ROOT, 'gene_features', 'NSCLC', 'samples')
    if kind == 'ccrcc':
        return os.path.join(PROJECT_ROOT, 'gene_features', 'CPTAC-CCRCC', 'samples')
    return os.path.join(PROJECT_ROOT, 'gene_features')


def _first_existing_dir(*paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    return paths[0]


def _default_clinical_path(feats_path, args):
    filename = os.path.basename(str(feats_path))
    table_root = _first_existing_dir(
        os.path.join(PROJECT_PARENT, 'Table'),
        os.path.join(PROJECT_ROOT, 'Table'),
    )
    if _dataset_kind(args.dataset) == 'tcga':
        return os.path.join(table_root, 'clip', 'tcga', filename[:12] + '.pt')
    return os.path.join(table_root, 'clip', 'CCRCC', filename[:9] + '.pt')


def _gene_path_for_patient(patient_id, args):
    gene_dir = _resolve_path(args.gene_dir) if args.gene_dir else _default_gene_dir(args.dataset)
    candidates = [
        os.path.join(gene_dir, patient_id + '.pt'),
        os.path.join(gene_dir, patient_id + '.npy'),
        os.path.join(gene_dir, 'samples', patient_id + '.pt'),
        os.path.join(gene_dir, 'samples', patient_id + '.npy'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_modality_tokens(feats_path, args, apply_missing=False):
    if args.model not in ('hyp_a', 'concat_mil', 'tri_concat', 'survpath'):
        empty = torch.zeros(0, args.table_dim).float()
        return empty, empty

    patient_id = _patient_id_from_feature_path(feats_path, args.dataset)
    modalities = set(args.fusion_modalities.split('+'))
    missing = args.test_missing if apply_missing else 'none'
    clinical = torch.zeros(0, args.table_dim).float()
    gene = torch.zeros(0, args.table_dim).float()

    if 'clinical' in modalities and missing not in ('c', 'clinical', 'cg', 'all'):
        clinical_path = args.clinical_dir
        if clinical_path:
            clinical_path = os.path.join(_resolve_path(clinical_path), patient_id + '.pt')
        else:
            clinical_path = _default_clinical_path(feats_path, args)
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(
                f"Clinical feature for patient '{patient_id}' not found: {clinical_path}. "
                "Use --fusion_modalities gene for the molecular-missing experiment "
                "when clinical features are unavailable."
            )
        clinical = _as_token_matrix(_load_feature_tensor(clinical_path))

    if 'gene' in modalities and missing not in ('g', 'gene', 'cg', 'all'):
        gene_path = _gene_path_for_patient(patient_id, args)
        if gene_path is not None:
            gene = _as_token_matrix(_load_feature_tensor(gene_path))

    return clinical, gene



class BagDataset(Dataset):
    def __init__(self,train_path, args, apply_test_missing=False) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args
        self.apply_test_missing = apply_test_missing
        # self.database = redis.Redis(host='localhost', port=6379)

    def get_bag_feats(self,csv_file_df, args):
        # if args.dataset == 'TCGA-lung-default':
        #     feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        feats_csv_path = csv_file_df.iloc[0]
        if _dataset_kind(args.dataset) == 'tcga' and not str(feats_csv_path).endswith('.pt'):
            feats_csv_path = feats_csv_path + '.pt'
        # if self.database is None:
        else:
            feats_csv_path = feats_csv_path
        feats = torch.load(feats_csv_path).float()
        clinical, gene = _load_modality_tokens(feats_csv_path, args, self.apply_test_missing)
        
        label = np.zeros(args.num_classes)
        if args.num_classes==1:
            label[0] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1
        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()
        # print(feats.shape)
        if torch.isnan(feats).any():
            print(f'Nan! {feats_csv_path}')
        return label, feats, clinical, gene

    def dropout_patches(self,feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats
    
    def __getitem__(self, idx):
        label, feats, clinical_feats, gene_feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        # print(feats.shape)
        return label, feats, clinical_feats, gene_feats
        
    def __len__(self):
        return len(self.train_path)


def train(train_df, milnet, criterion, optimizer, args, log_path, epoch=0):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    
    for i,(bag_label,bag_feats,clinical_feats,gene_feats) in enumerate(train_df):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        optimizer.zero_grad()
        if args.model == 'dsmil' or args.model == 'hyp_d':
            ins_prediction, bag_prediction, attention, atten_B = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            # print(bag_prediction, max_prediction,bag_label.long())      
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            # bag_loss = criterion(bag_prediction, bag_label.long())
            # max_loss = criterion(max_prediction.view(1, -1), bag_label.long())
            loss = 0.5*bag_loss + 0.5*max_loss
        elif args.model == 'hyp_a':
            clinical_feats = None if clinical_feats.numel() == 0 else clinical_feats.cuda()
            gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
            bag_prediction, pred, attention, loss1 = milnet(bag_feats, clinical_feats, gene_feats)
            # loss1 =  criterion(bag_prediction.view(1, -1), bag_label.repeat_interleave(5).view(1, -1)) # .repeat_interleave(5).view(1, -1))
            loss2 = criterion(pred.view(1, -1), bag_label.view(1, -1))
            # print(loss1)
            loss = loss1 + loss2 # + p2c
        elif args.model == 'hyperpath':
            # HyperPath baseline: returns (logits, logits, attn, aux_loss).
            # aux_loss already contains the InfoNCE + entailment terms.
            bag_prediction, pred, attention, aux_loss = milnet(bag_feats, None, bag_label)
            cls_loss = criterion(pred.view(1, -1), bag_label.view(1, -1))
            loss = cls_loss + aux_loss
        elif args.model in ('abmil', 'wikg', 'ilra'):
            # ABMIL / WiKG / ILRA share the same 3-tuple pure-WSI interface:
            #   (bag_prediction[1,C], pred[1,C], attention[1,N]).
            bag_prediction, pred, attention = milnet(bag_feats)
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        elif args.model in ('concat_mil', 'tri_concat'):
            # Concat-MIL: token-level concat attention.
            # Tri-Concat: modality-level WSI/gene/clinical embedding concat.
            clinical_feats = None if clinical_feats.numel() == 0 else clinical_feats.cuda()
            gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
            bag_prediction, pred, attention = milnet(bag_feats, clinical_feats, gene_feats)
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        elif args.model == 'survpath':
            # SurvPath baseline: WSI patch bag + pathway-grouped gene tokens.
            gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
            bag_prediction, pred, attention = milnet(bag_feats, gene_feats)
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        else:
            bag_prediction, pred, attention = milnet(bag_feats,)
            loss1 =  criterion(bag_prediction.view(1, -1), bag_label.repeat_interleave(5).view(1, -1)) # .repeat_interleave(5).view(1, -1))
            loss2 = criterion(pred.view(1, -1), bag_label.view(1, -1))
            # print(loss1)
            loss = loss1 +loss2
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        atten_max = atten_max + attention.max().item()
        atten_min = atten_min + attention.min().item()
        atten_mean = atten_mean +  attention.mean().item()
        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f, attention max:%.4f, min:%.4f, mean:%.4f' % (i, len(train_df), loss.item(), 
                        attention.max().item(), attention.min().item(), attention.mean().item()))
    atten_max = atten_max / len(train_df)
    atten_min = atten_min / len(train_df)
    atten_mean = atten_mean / len(train_df)
    with open(log_path,'a+') as log_txt:
            log_txt.write('\n atten_max'+str(atten_max))
            log_txt.write('\n atten_min'+str(atten_min))
            log_txt.write('\n atten_mean'+str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i,(bag_label,bag_feats,clinical_feats,gene_feats) in enumerate(test_df):
            label = bag_label.numpy()
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil' or args.model == 'hyp_d':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)  
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                # bag_loss = criterion(bag_prediction, bag_label.long())
                # max_loss = criterion(max_prediction.view(1, -1), bag_label.long())
                loss = 0.5*bag_loss + 0.5*max_loss
            elif args.model in ['abmil', 'max', 'mean', 'wikg', 'ilra']:
                bag_prediction, _, _ =  milnet(bag_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model in ('concat_mil', 'tri_concat'):
                clinical_feats = None if clinical_feats.numel() == 0 else clinical_feats.cuda()
                gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
                bag_prediction, _, _ = milnet(bag_feats, clinical_feats, gene_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'survpath':
                gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
                bag_prediction, _, _ = milnet(bag_feats, gene_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model in ['hyp_a']:
                clinical_feats = None if clinical_feats.numel() == 0 else clinical_feats.cuda()
                gene_feats = None if gene_feats.numel() == 0 else gene_feats.cuda()
                # HypABMIL returns (softmax probabilities, logits, attention,
                # hierarchy_loss).  Use logits for BCEWithLogitsLoss, and keep
                # probabilities untouched for scoring.
                bag_prediction, logits, _, _ = milnet(bag_feats, clinical_feats, gene_feats)
                max_prediction = logits
                loss = criterion(logits.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'hyperpath':
                bag_prediction, _, _, _ = milnet(bag_feats, None, None)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model in ['acmil', 'hyp_c']:
                _, bag_prediction, _ =  milnet(bag_feats)
                #print(bag_label.shape)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend(label)
            if args.model == 'hyp_a':
                # HypABMIL bag_prediction is already a softmax probability vector.
                test_predictions.extend([bag_prediction.squeeze().cpu().numpy()])
            elif args.model in ('survpath', 'tri_concat') and args.num_classes > 1:
                test_predictions.extend([torch.softmax(bag_prediction, dim=1).squeeze().cpu().numpy()])
            elif args.average:   # notice args.average here
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)


    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print('\n')
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
        bag_score = 0
        for i in range(0, len(test_df)):
            bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
        avg_score = bag_score / len(test_df)  # ACC
        cls_report = classification_report(test_labels, test_predictions, digits=4, zero_division=0)
    else:
        # The TCGA/CCRCC tasks are mutually exclusive multi-class
        # classification tasks encoded as one-hot labels.  Use argmax for ACC
        # instead of one-vs-rest thresholding, which can produce invalid
        # multi-hot/all-zero predictions such as [0, 0] for a two-class task.
        label_idx = np.argmax(test_labels, axis=1)
        pred_idx = np.argmax(test_predictions, axis=1)
        class_predictions = np.zeros_like(test_predictions)
        class_predictions[np.arange(len(pred_idx)), pred_idx] = 1
        for i in range(args.num_classes):
            print(confusion_matrix(test_labels[:, i], class_predictions[:, i]))
            info = confusion_matrix(test_labels[:, i], class_predictions[:, i])
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
        test_predictions = class_predictions
        avg_score = accuracy_score(label_idx, pred_idx)
        cls_report = classification_report(label_idx, pred_idx, digits=4, zero_division=0)
    valid_auc = [a for a in auc_value if np.isfinite(a)]
    auc_mean = float(np.mean(valid_auc)) if len(valid_auc) > 0 else 0.0
    print('\n  Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, auc_mean*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, auc_mean*100))
        log_txt.write('\n' + cls_report)
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if len(np.unique(label)) < 2:
            aucs.append(np.nan)
            thresholds_optimal.append(0.5)
            continue
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train IBMIL for abmil and dsmil')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--bags_csv', default=None, type=str,
                        help='Optional explicit WSI bag CSV. If unset, inferred from --dataset and --backbone.')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', type=str,help='which agg')
    parser.add_argument('--backbone', default='r18', type=str,help='ctrans or r18')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    # for ablations only
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')
    parser.add_argument('--c_dim', default=128, type=int, help='Dimension of the projected confounders')
    parser.add_argument('--freeze_epoch', default=999, type=int, help='freeze confounders during this many epoch from the start')
    parser.add_argument('--c_merge', type=str, default='cat', help='cat or add or sub')
    parser.add_argument('--hyperpath_task', type=str, default=None,
                        help='Override HyperPath prompt task (NSCLC / CCRCC / C16). '
                             'If unset, inferred from --dataset prefix.')
    parser.add_argument('--conch_ckpt', type=str,
                        default='/mnt/pfs-mc0p4k/cvg/team/didonglin/yangguang/WSI/CLAM/pytorch_model.bin',
                        help='Local CONCH checkpoint path, or "hf_hub:MahmoodLab/conch" '
                             'to pull from HuggingFace. Overridable via env var CONCH_CKPT.')
    # ===== multimodal table-side features =====
    parser.add_argument('--fusion_modalities', type=str, default='clinical',
                        choices=['clinical', 'gene', 'clinical+gene'],
                        help='Table-side modalities used by hyp_a / concat_mil / tri_concat. '
                             'gene features are produced by MIL/preprocess_gene.py.')
    parser.add_argument('--clinical_dir', type=str, default=None,
                        help='Directory containing patient-level clinical .pt files. '
                             'Default: Table/clip/{tcga,CCRCC}.')
    parser.add_argument('--gene_dir', type=str, default=None,
                        help='Directory containing patient-level gene .pt/.npy files. '
                             'Default: gene_features/{NSCLC,CPTAC-CCRCC}/samples.')
    parser.add_argument('--table_dim', type=int, default=512,
                        help='Feature dimension of each table-side token. The gene '
                             'preprocessor defaults to 512 to match clinical tokens.')
    parser.add_argument('--num_gene_pathways', type=int, default=331,
                        help='Number of pathway-grouped gene tokens used by HypABMIL / SurvPath.')
    parser.add_argument('--hierarchy_weight', type=float, default=1.0,
                        help='Weight for the cross-modal hierarchy loss L_T in HypABMIL.')
    # ===== missing-modality (HypABMIL only) =====
    parser.add_argument('--test_missing', type=str, default='none',
                        choices=['none', 'c', 'clinical', 'g', 'gene', 'cg', 'all'],
                        help="Inference-time missing modality simulation. "
                             "'none' = full modality, 'c' = clinical missing, "
                             "'g' = gene missing, 'cg'/'all' = table side missing.")
    parser.add_argument('--missing_completion', type=str, default='hgs',
                        choices=['hgs', 'placeholder', 'zero'],
                        help='HypABMIL completion strategy when a table-side modality is missing. '
                             "'hgs'=hierarchy-anchor + prototype geodesic averaging (default); "
                             "'placeholder'=learnable token only; 'zero'=zero vector.")
    parser.add_argument('--num_prototypes', type=int, default=8,
                        help='Number of cluster prototypes per table-side modality in HypABMIL.')
    parser.add_argument('--survpath_max_patches', type=int, default=4096,
                        help='Maximum WSI patches sampled by SurvPath. Use <=0 to disable sampling.')

    # ===== WiKG (pure-WSI baseline) =====
    parser.add_argument('--wikg_topk', type=int, default=6,
                        help='Top-k neighbours used in WiKG knowledge-graph attention.')
    parser.add_argument('--wikg_agg', type=str, default='bi-interaction',
                        choices=['gcn', 'sage', 'bi-interaction'],
                        help='Message aggregation type for WiKG.')
    parser.add_argument('--wikg_pool', type=str, default='attn',
                        choices=['attn', 'mean', 'max'],
                        help='Readout pooling for WiKG.')

    # ===== ILRA (pure-WSI baseline) =====
    parser.add_argument('--ilra_topk', type=int, default=4,
                        help='Rank of the latent matrix (num_inds) in ILRA GAB blocks.')
    parser.add_argument('--ilra_layers', type=int, default=2,
                        help='Number of stacked GAB blocks in ILRA.')
    parser.add_argument('--ilra_hidden', type=int, default=256,
                        help='Hidden dimension in ILRA.')
    parser.add_argument('--ilra_heads', type=int, default=8,
                        help='Number of attention heads in ILRA.')

    args = parser.parse_args()

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_c_path')
    else:
        save_path = os.path.join('baseline', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_fulltune')
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'
    

    # seed 
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    if args.model == 'dsmil':
        import dsmil as mil
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,confounder_path=args.c_path).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    elif args.model == 'hyp_d':
        import dsmil as mil
        i_classifier = mil.HypFCLayer(k=1.0, in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,confounder_path=args.c_path).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    elif args.model == 'abmil':
        import abmil as mil
        milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes,confounder_path=args.c_path, \
            confounder_learn=args.c_learn, confounder_dim=args.c_dim, confounder_merge=args.c_merge).cuda()
    elif args.model == 'acmil':
        import abmil as mil
        milnet = mil.ACMIL(in_size=args.feats_size, out_size=args.num_classes).cuda()
    elif args.model == 'hyp_a':
        import abmil as mil
        milnet = mil.HypABMIL(
            input_dim=args.feats_size,
            n_classes=args.num_classes,
            clinical_in_dim=args.table_dim,
            gene_in_dim=args.table_dim,
            fusion_modalities=args.fusion_modalities,
            missing_completion=args.missing_completion,
            hierarchy_weight=args.hierarchy_weight,
            num_prototypes=args.num_prototypes,
            num_gene_pathways=args.num_gene_pathways,
        ).cuda()
    elif args.model == 'concat_mil':
        import abmil as mil
        # Simple Euclidean WSI + clinical-text late-fusion baseline.
        # table_dim = 512 because clinical text features are PLIP/CLIP encoded.
        milnet = mil.ConcatMIL(
            input_dim=args.feats_size,
            table_dim=args.table_dim,
            hidden_dim=256,
            n_classes=args.num_classes,
        ).cuda()
    elif args.model == 'tri_concat':
        import abmil as mil
        # True three-modality concat baseline: WSI, pathway-gene and clinical
        # are each pooled to one vector, then concatenated for classification.
        milnet = mil.TriModalConcatMIL(
            input_dim=args.feats_size,
            table_dim=args.table_dim,
            hidden_dim=256,
            n_classes=args.num_classes,
            num_gene_pathways=args.num_gene_pathways,
        ).cuda()
    elif args.model == 'survpath':
        import abmil as mil
        milnet = mil.SurvPathMIL(
            input_dim=args.feats_size,
            gene_dim=args.table_dim,
            hidden_dim=256,
            n_classes=args.num_classes,
            num_pathways=args.num_gene_pathways,
            max_patches=args.survpath_max_patches,
        ).cuda()
    elif args.model == 'hyp_c':
        import abmil as mil
        milnet = mil.HypACMIL(k=1.0, in_size=args.feats_size, out_size=args.num_classes).cuda()
    elif args.model == 'wikg':
        # Knowledge-graph MIL (CVPR 2024).  Pure-WSI baseline; same input /
        # output contract as ABMIL so it slots into the existing train/test
        # 3-tuple branch.
        import wikg as mil
        milnet = mil.WiKG(
            dim_in=args.feats_size,
            dim_hidden=512,
            topk=args.wikg_topk,
            n_classes=args.num_classes,
            agg_type=args.wikg_agg,
            dropout=0.3,
            pool=args.wikg_pool,
        ).cuda()
    elif args.model == 'ilra':
        # Low-Rank-Property MIL (ICLR 2023).  Pure-WSI baseline.
        import wikg as mil
        milnet = mil.ILRA(
            num_layers=args.ilra_layers,
            feat_dim=args.feats_size,
            n_classes=args.num_classes,
            hidden_feat=args.ilra_hidden,
            num_heads=args.ilra_heads,
            topk=args.ilra_topk,
            ln=False,
        ).cuda()
    elif args.model == 'hyperpath':
        import hyperpath as mil
        # Priority: explicit --hyperpath_task > infer from --dataset > 'NSCLC'.
        ds_up = args.dataset.upper()
        if args.hyperpath_task is not None and args.hyperpath_task != '':
            hp_task = args.hyperpath_task
        elif ds_up.startswith('TCGA') and 'NSCLC' in ds_up or ds_up.startswith('NSCLC') \
                or ds_up.startswith('LUAD') or ds_up.startswith('LUSC'):
            hp_task = 'NSCLC'
        elif ds_up.startswith('CCRCC') or ds_up.startswith('KIRC') or ds_up.startswith('CPTAC'):
            hp_task = 'CCRCC'
        elif ds_up.startswith('C16') or ds_up.startswith('CAMELYON'):
            hp_task = 'C16'
        else:
            # last-resort default
            hp_task = 'NSCLC'
        print(f"[hyperpath] dataset='{args.dataset}' -> task='{hp_task}', "
              f"n_classes={args.num_classes}")
        # CLI > env var > built-in default
        conch_ckpt = os.environ.get('CONCH_CKPT', None) or args.conch_ckpt
        milnet = mil.HyperPathMIL(
            input_dim=args.feats_size,
            n_classes=args.num_classes,
            task=hp_task,
            conch_ckpt=conch_ckpt,
            hf_token=os.environ.get('HF_TOKEN', None),
        ).cuda()
    
    for name, _ in milnet.named_parameters():
        print('Training {}'.format(name))
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n Training {}'.format(name))


        
    if args.bags_csv:
        bags_csv = _resolve_path(args.bags_csv)
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
    elif _dataset_kind(args.dataset) == "tcga":
        mildata_root = _first_existing_dir(
            os.path.join(PROJECT_PARENT, 'MILdata'),
            os.path.join(PROJECT_ROOT, 'MILdata'),
        )
        if args.backbone == 'ctrans':
            bags_csv = os.path.join(mildata_root, 'tcga_ctrans', args.dataset+'.csv') # ImageNet/ctrans
        elif args.backbone == 'conch':
            bags_csv = os.path.join(mildata_root, 'tcga_conch', args.dataset+'.csv') # ImageNet/ctrans
        else :
            bags_csv = os.path.join(mildata_root, 'tcga_ImageNet', args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]

    elif _dataset_kind(args.dataset) == 'ccrcc':
        mildata_root = _first_existing_dir(
            os.path.join(PROJECT_PARENT, 'MILdata'),
            os.path.join(PROJECT_ROOT, 'MILdata'),
        )
        if args.backbone == 'ctrans':
            bags_csv = os.path.join(mildata_root, 'CCRCC_ctrans', args.dataset+'.csv') # ImageNet/ctrans
        elif args.backbone == 'conch':
            bags_csv = os.path.join(mildata_root, 'CCRCC_conch', args.dataset+'.csv')
        else :
            bags_csv = os.path.join(mildata_root, 'CCRCC_r18', args.dataset+'.csv')
        # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]

    elif args.dataset.startswith('C16'):
        if args.backbone == 'ctrans':
            bags_csv = os.path.join('../MILdata/Camelyon16_ctrans/', args.dataset+'.csv') # ImageNet/ctrans
        elif args.backbone == 'conch':
            bags_csv = os.path.join('../MILdata/Camelyon16_conch/', args.dataset+'.csv')
        else :
            bags_csv = os.path.join('../MILdata/Camelyon16_ImageNet/', args.dataset+'.csv')
        # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:270, :]
        test_path = bags_path.iloc[270:, :]
        
    trainset =  BagDataset(train_path, args, apply_test_missing=False)
    train_loader = DataLoader(trainset,1, shuffle=True, num_workers=16)
    testset =  BagDataset(test_path, args, apply_test_missing=True)
    test_loader = DataLoader(testset,1, shuffle=False, num_workers=16)

    # sanity check begins here
    print('*******sanity check *********')
    for k,v in milnet.named_parameters():
        if v.requires_grad == True:
            print(k)

     # loss, optim, schduler
    criterion = nn.BCEWithLogitsLoss() 
    original_params = []
    confounder_parms = []
    for pname, p in milnet.named_parameters():
        if ('confounder' in pname):
            confounder_parms += [p]
            print('confounders:',pname )
        else:
            original_params += [p]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), 
                                lr=args.lr, betas=(0.5, 0.9), 
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    best_score = 0

    # ### inference only
    # if args.test:
    #     epoch = args.num_epochs-1
    #     test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)   
                
    #     train_loss_bag = 0
    #     if args.dataset=='TCGA-lung':
    #         print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
    #               (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    #     else:
    #         print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
    #               (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        
    #     if args.model == 'dsmil':
    #         if  args.agg  == 'tcga':
    #             load_path = 'test/weights/aggregator.pth' 
    #         elif  args.agg  == 'c16':
    #             load_path = 'test-c16/weights/aggregator.pth'   
    #         else:
    #             raise NotImplementedError
                
    #     elif args.model == 'abmil':
    #         if args.agg  == 'tcga':
    #             load_path = 'pretrained_weights/abmil_tcgapretrained.pth' # load c-16 pretrain for adaption
    #         elif args.agg  == 'c16':
    #             load_path = 'pretrained_weights/abmil_c16pretrained.pth'   # load tcga pretrain for adaption
    #         else:
    #             raise NotImplementedError
    #     state_dict_weights = torch.load(load_path)
    #     print('Loading model:{} with {}'.format(args.model, load_path))
    #     with open(log_path,'a+') as log_txt:
    #         log_txt.write('\n loading init from:'+str(load_path))
    #     msg = milnet.load_state_dict(state_dict_weights, strict=False)
    #     print('Missing these:', msg.missing_keys)
    #     test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
    #     if args.dataset=='TCGA-lung':
    #         print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
    #               (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    #     else:
    #         print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
    #               (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
    #     sys.exit()
        
    
    

    
    for epoch in range(1, args.num_epochs):
        start_time = time.time()
        train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, log_path, epoch=epoch-1) # iterate all bags
        print('epoch time:{}'.format(time.time()- start_time))
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
        with open(log_path,'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

        scheduler.step()
        valid_aucs = [a for a in aucs if np.isfinite(a)]
        auc_score = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0
        current_score = (auc_score + avg_score) / 2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            with open(log_path,'a+') as log_txt:
                info = 'Best model saved at: ' + save_name +'\n'
                log_txt.write(info)
                info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                log_txt.write(info)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        if epoch == args.num_epochs-1:
            save_name = os.path.join(save_path, 'last.pth')
            torch.save(milnet.state_dict(), save_name)
    log_txt.close()

if __name__ == '__main__':
    main()
