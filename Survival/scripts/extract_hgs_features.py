"""
Extract per-patient (pathology, omics, clinical) token embeddings from a
trained MHSurv checkpoint for use in HGS theoretical-analysis figures.

This is a visualization-only inference pass: it skips the fusion / classifier
head and just calls the three encoder paths inside MHSurv, then pools each
output to a single 256-D vector per patient.  Results are written to a .npz
with the following arrays:

    P            : [N_patients, 256]   mean-pooled pathology tokens
    O            : [N_patients, 256]   mean-pooled omics tokens
    C            : [N_patients, 256]   mean-pooled clinical tokens
    patient_ids  : [N_patients]        str ids if available, otherwise indices

Usage (single cohort, single fold):

    python scripts/extract_hgs_features.py \
        --ckpt   /path/to/s_0_checkpoint.pt \
        --cohort brca \
        --fold   0 \
        --out    hgs_features_brca_fold0.npz

To pool across cohorts later, call this script multiple times and concatenate
the resulting .npz arrays in numpy.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

# ---- avoid the "Too many open files" crash that happens when
#      DataLoader workers share many small tensors through the default
#      file-descriptor backend.  Same setting used by main.py.
torch.multiprocessing.set_sharing_strategy('file_system')

# ---- make repo root importable when run from /scripts -------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset_survival import SurvivalDatasetFactory   # noqa: E402
from utils.core_utils import (                                  # noqa: E402
    _init_model,
    _load_checkpoint,
    _init_loaders,
    _get_splits,
    _unpack_data,
)
from utils.general_utils import _prepare_for_experiment         # noqa: E402
from utils.process_args import _process_args                    # noqa: E402


# =========================================================================
# Core feature extraction — reuses MHSurv's encoders, skips the fusion head
# =========================================================================
@torch.no_grad()
def _encode_one_patient(model, kwargs, device):
    """Run the three encoder paths of MHSurv and return (P, O, C) tokens.

    The function mirrors the Branch D code path of MHSurv.forward (full
    modality), stopping right before the attention_fusion call.  We pool
    each modality with a simple mean across its token dimension so the
    downstream visualization compares per-patient 256-d embeddings.
    """
    # ---- pathology ----
    x_path = kwargs['x_path']
    pathology_features = model.ffpe_fc(x_path).to(device)         # [1, N_patch, 256]
    path_token = pathology_features.mean(dim=1).squeeze(0)        # [256]

    # ---- omics (genomics + transomics + protein, via the model's helper) ----
    omic_kwargs = {'protein': kwargs.get('protein', None)}
    # copy over the omic-by-omic inputs that process_multiomics expects
    for k, v in kwargs.items():
        if k.startswith('x_genomic') or k.startswith('x_transomic'):
            omic_kwargs[k] = v
    genomics_features, _, _ = model.process_multiomics(omic_kwargs)  # [1, N_omic, 256]
    omic_token = genomics_features.mean(dim=1).squeeze(0)             # [256]

    # ---- clinical ----
    x_table = kwargs['report']
    clinic_features = model.clinic_fc(x_table).to(device)
    if clinic_features.dim() == 3:
        clinic_token = clinic_features.mean(dim=1).squeeze(0)
    elif clinic_features.dim() == 2:
        # [N_c, 256] -> [256]; if a batch dim shows up, mean over it too
        clinic_token = clinic_features.mean(dim=0)
    else:
        clinic_token = clinic_features.view(-1)[:256]

    return path_token, omic_token, clinic_token


@torch.no_grad()
def extract_single_patient_tokens(model, val_loader, device, patient_idx=0):
    """Extract per-token (P, O, C) sequences for ONE specific patient.

    Unlike `extract_features` (which mean-pools each modality to a single
    256-d vector per patient), this dumps the full token bags:

        P : [N_patch,  256]   pathology patch tokens
        O : [N_omic,   256]   omic tokens (genomic + transomic + protein)
        C : [N_c,      256]   clinical attribute tokens

    Intra-patient token variation is much richer than the cross-patient
    variation of the mean-pooled version, so HGS samples driven by these
    tokens do not collapse to a single point in PCA-2D.
    """
    model.eval()
    for idx, batch in enumerate(val_loader):
        if idx != patient_idx:
            continue
        data_WSI, protein, _, _, _, genomics, transomics, \
            table_features, _, patient = _unpack_data('hgnn', device, batch)

        kwargs = {
            'x_path': data_WSI,
            'protein': protein,
            'report': table_features,
            'valid': True,
        }
        for i, g in enumerate(genomics):
            kwargs[f'x_genomic{i+1}'] = g.type(torch.FloatTensor).to(device)
        for i, t in enumerate(transomics):
            kwargs[f'x_transomic{i+1}'] = t.type(torch.FloatTensor).to(device)

        # ---- pathology: keep every patch token ----
        path_tokens = model.ffpe_fc(kwargs['x_path']).to(device)        # [1, N_patch, 256]
        path_tokens = path_tokens.squeeze(0)                            # [N_patch, 256]

        # ---- omics: keep every omic token AND record sub-modality slices ----
        omic_kwargs = {'protein': kwargs.get('protein', None)}
        for k, v in kwargs.items():
            if k.startswith('x_genomic') or k.startswith('x_transomic'):
                omic_kwargs[k] = v
        omic_feat, _, _ = model.process_multiomics(omic_kwargs)         # [1, N_omic, 256]
        omic_tokens = omic_feat.squeeze(0)                              # [N_omic, 256]
        # process_multiomics concatenates  genomics + transomics + protein
        # in that order; recover the sub-bag sizes so plot_hgs_theory can
        # pick a single sub-modality as the imputation target.
        n_g = sum(1 for k in kwargs if k.startswith('x_genomic')
                  and kwargs[k] is not None)                            # ~6
        n_t = sum(1 for k in kwargs if k.startswith('x_transomic')
                  and kwargs[k] is not None)                            # ~331
        # the remaining tokens are the protein bag (if present)
        n_p = omic_tokens.shape[0] - n_g - n_t
        omic_split = np.array([n_g, n_t, n_p], dtype=np.int64)

        # ---- clinical: keep every attribute token ----
        x_table = kwargs['report']
        clinic_feat = model.clinic_fc(x_table).to(device)
        if clinic_feat.dim() == 1:
            clinic_feat = clinic_feat.unsqueeze(0)
        if clinic_feat.dim() == 3:
            clinic_feat = clinic_feat.squeeze(0)                        # [N_c, 256]
        clinic_tokens = clinic_feat

        return (
            path_tokens.detach().cpu().numpy().astype(np.float32),
            omic_tokens.detach().cpu().numpy().astype(np.float32),
            clinic_tokens.detach().cpu().numpy().astype(np.float32),
            str(patient) if patient is not None else f'idx_{idx}',
            omic_split,
        )

    raise IndexError(
        f"patient_idx={patient_idx} out of range; val_loader has "
        f"{len(val_loader)} patients.")


@torch.no_grad()
def extract_features(model, val_loader, device):
    """Iterate the val loader and accumulate per-patient tokens."""
    model.eval()
    P_list, O_list, C_list, pid_list = [], [], [], []

    for idx, batch in enumerate(val_loader):
        data_WSI, protein, _, _, _, genomics, transomics, \
            table_features, _, patient = _unpack_data('hgnn', device, batch)

        kwargs = {
            'x_path': data_WSI,
            'protein': protein,
            'report': table_features,
            'valid': True,
        }
        for i, g in enumerate(genomics):
            kwargs[f'x_genomic{i+1}'] = g.type(torch.FloatTensor).to(device)
        for i, t in enumerate(transomics):
            kwargs[f'x_transomic{i+1}'] = t.type(torch.FloatTensor).to(device)

        try:
            p_tok, o_tok, c_tok = _encode_one_patient(model, kwargs, device)
        except Exception as e:
            # skip patients where any encoder path errors (e.g. missing
            # modality on disk); they are not useful for the analysis.
            print(f"  [skip patient {patient}] {type(e).__name__}: {e}")
            continue

        P_list.append(p_tok.detach().cpu().numpy().astype(np.float32))
        O_list.append(o_tok.detach().cpu().numpy().astype(np.float32))
        C_list.append(c_tok.detach().cpu().numpy().astype(np.float32))
        pid_list.append(str(patient) if patient is not None else f'idx_{idx}')

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1} patients ...")

    return (
        np.stack(P_list, axis=0),
        np.stack(O_list, axis=0),
        np.stack(C_list, axis=0),
        np.array(pid_list, dtype=object),
    )


# =========================================================================
# Args / setup — mirrors main.py so the val loader matches training splits
# =========================================================================
def _build_args(cli):
    """Reuse the project's standard argparser then override paths."""
    # Build a synthetic sys.argv so _process_args() can parse it.
    fake_argv = [
        'extract_hgs_features.py',
        '--study',           f'tcga_{cli.cohort}',
        '--task',            'survival',
        '--split_dir',       cli.split_dir,
        '--which_splits',    cli.which_splits,
        '--type_of_path',    cli.type_of_path,
        '--modality',        'hgnn',
        '--data_root_dir',   os.path.join(cli.data_root_dir, cli.cohort),
        '--label_file',      f'datasets_csv/metadata/tcga_{cli.cohort}.csv',
        '--omics_dir',       f'datasets_csv/raw_rna_data/{cli.type_of_path}/{cli.cohort}',
        '--results_dir',     f'results_{cli.cohort}_extract',
        '--batch_size',      '1',
        '--lr',              '0.0001',
        '--opt',             'radam',
        '--reg',             '0.0001',
        '--alpha_surv',      '0.5',
        '--weighted_sample',
        '--max_epochs',      '1',
        '--encoding_dim',    str(cli.encoding_dim),
        '--label_col',       'survival_months_dss',
        '--k',               '5',
        '--bag_loss',        'nll_surv',
        '--n_classes',       '4',
        '--num_patches',     str(cli.num_patches),
        '--wsi_projection_dim', '256',
        '--fusion',          'concat',
    ]
    saved_argv = sys.argv
    sys.argv = fake_argv
    try:
        args = _process_args()
    finally:
        sys.argv = saved_argv

    args = _prepare_for_experiment(args)
    # extraction is full-modality; do NOT trigger any test-time missing dispatch
    args.test_missing_mode = ''
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed,
        print_info=False,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat='coattn' in args.modality,
        is_survpath=True,
        type_of_pathway=args.type_of_path,
    )
    args.memory_name = 'memory/c_1.h5'
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     required=True, help='Trained MHSurv checkpoint .pt')
    ap.add_argument('--cohort',   required=True,
                    choices=['blca', 'brca', 'coadread', 'hnsc', 'stad'])
    ap.add_argument('--fold',     type=int, default=0,
                    help='Which fold to evaluate the val split of (0..4).')
    ap.add_argument('--out',      type=str, default='hgs_features.npz')
    # data layout (defaults match the standard repo paths used by main.py)
    ap.add_argument('--data_root_dir', default='../WSIdata')
    ap.add_argument('--split_dir',     default='splits')
    ap.add_argument('--which_splits',  default='fxfolds')
    ap.add_argument('--type_of_path',  default='combine')
    ap.add_argument('--encoding_dim',  type=int, default=1024)
    ap.add_argument('--num_patches',   type=int, default=4096)
    ap.add_argument('--single_patient', type=int, default=None,
                    help='If set, dump per-token sequences for the patient at '
                         'this index (0-based) inside the val_loader instead of '
                         'the cohort-wide mean-pooled features. Output npz has '
                         'P, O, C of shapes [N_patch, 256] / [N_omic, 256] / '
                         '[N_c, 256] for that one patient.')
    cli = ap.parse_args()

    args = _build_args(cli)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # ---- val loader for the requested fold -------------------------------
    # NOTE: must be set up BEFORE `_init_model`, because
    #       `args.dataset_factory.return_splits(...)` populates
    #       `args.genomic_sizes` and `args.transomic_sizes` based on the
    #       cohort's signature CSVs.  `_init_model` reads those attributes
    #       to build the MHSurv SNN heads.
    csv_path = '{}/splits_{}.csv'.format(args.split_dir, cli.fold)
    print(f"[data]  using {csv_path}")
    datasets = args.dataset_factory.return_splits(
        args, csv_path=csv_path, fold=cli.fold)
    train_split, val_split = _get_splits(datasets, cli.fold, args)
    _, val_loader = _init_loaders(args, train_split, val_split)
    # Replace val_loader with a single-process variant so the one-shot
    # feature dump does not exhaust the shared-memory file descriptor
    # budget that the default num_workers=8 setting would consume.
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(val_loader, 'collate_fn', None),
    )
    print(f"[data]  val patients: {len(val_loader)}  (single-process loader)")
    print(f"[data]  genomic_sizes={args.genomic_sizes} "
          f"transomic_sizes=(n={len(args.transomic_sizes)})")

    # ---- model + checkpoint ----------------------------------------------
    model = _init_model(args).to(device)
    model = _load_checkpoint(model, cli.ckpt)
    model.eval()
    print(f"[ckpt]  loaded {cli.ckpt}")

    # ---- extract --------------------------------------------------------
    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    if cli.single_patient is not None:
        P, O, C, pid, o_split = extract_single_patient_tokens(
            model, val_loader, device, patient_idx=cli.single_patient)
        n_g, n_t, n_p = o_split.tolist()
        print(f"[out]   patient={pid}  P:{P.shape}  O:{O.shape}  C:{C.shape}")
        print(f"[out]   O breakdown: genomic={n_g}  transomic={n_t}  protein={n_p}")
        np.savez(cli.out, P=P, O=O, C=C, patient_id=pid,
                 cohort=cli.cohort, fold=cli.fold, mode='single_patient',
                 o_split=o_split)
    else:
        P, O, C, pids = extract_features(model, val_loader, device)
        print(f"[out]   P:{P.shape}  O:{O.shape}  C:{C.shape}")
        np.savez(cli.out, P=P, O=O, C=C, patient_ids=pids,
                 cohort=cli.cohort, fold=cli.fold, mode='cohort_pool')
    print(f"[done]  features written to {cli.out}")


if __name__ == '__main__':
    main()
