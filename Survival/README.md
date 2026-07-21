# HuMP Survival Code

This directory contains the survival prediction pipeline. The HuMP model is implemented as `MHSurv` in `models/model_HGNN.py`.

Run from this directory:

```bash
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv
```

Use `--test_missing_mode P/O/C/PO/PC/OC` for inference-time missing-modality evaluation.

More details are in `../docs/survival.md`.

