# LIDC-IDRI Consensus 3D Workflow

This note records the cleaner raw-data pipeline for the public `LIDC-IDRI` benchmark.

## Goal

Use the raw `LIDC-IDRI` DICOM + XML download to build a literature-style benign-vs-malignant benchmark:

1. Build a patient-wise `split_manifest.csv`.
2. Aggregate XML reader annotations into `consensus nodules`.
3. Crop 3D nodule volumes from raw DICOM.
4. Train on a fixed fold with `use_predefined_split`, instead of splitting again inside `train.py`.

## Step 1: Build Consensus Split Manifest

```bash
python build_lidc_idri_split_manifest.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --output-dir outputs/lidc_bvm_manifest_consensus \
  --metadata-source auto \
  --annotation-policy consensus \
  --consensus-min-readers 2 \
  --xy-tolerance-px 15 \
  --z-tolerance-mm 3 \
  --label-policy score12_vs_score45 \
  --split-scheme patient_kfold \
  --n-splits 5 \
  --val-ratio 0.1 \
  --seed 42
```

Expected outputs:

- `nodule_manifest.csv`
- `split_manifest.csv`
- `patient_summary.csv`
- `summary.json`

Recommended first checks:

- open `summary.json`
- confirm `annotation_policy=consensus`
- confirm class distribution is reasonable
- confirm mixed-label patients are much lower than the old reader-level version

## Step 2: Crop Consensus 3D Nodules

```bash
python prepare_lidc_idri_consensus_3d.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --manifest-csv outputs/lidc_bvm_manifest_consensus/nodule_manifest.csv \
  --split-manifest-csv outputs/lidc_bvm_manifest_consensus/split_manifest.csv \
  --split-fold 0 \
  --output-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --depth-size 32 \
  --volume-hw 128 \
  --context-scale 1.5 \
  --min-size-xy 32 \
  --min-size-z 8
```

Expected outputs:

- `benign/*.npy`
- `malignant/*.npy`
- `processed_manifest.csv`
- `processed_split_manifest.csv`
- `preprocess_summary.json`
- `preprocess_failures.csv`

Recommended first checks:

- `processed_split_manifest.csv` contains only one fold
- output file names match `output_stem`
- `preprocess_failures.csv` is empty or only has a few understandable failures

## Step 3: Train the Public Clean Baseline

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --output-dir outputs/lidc_bvm_resnet3d18_fold0 \
  --model resnet3d18 \
  --pretrained \
  --use-3d-input \
  --class-mode binary \
  --binary-task benign_vs_malignant \
  --use-predefined-split \
  --split-manifest-csv /workspace/data-lung/lidc_idri_consensus_3d_fold0/processed_split_manifest.csv \
  --split-fold 0 \
  --split-mode train_val_test \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --selection-metric auroc
```

Notes:

- `split_manifest_csv` now lets `LIDCIDRIDataset` attach fixed `train/val/test` metadata.
- `train.py` will automatically enable `use_predefined_split` when `split_manifest_csv` is provided for `lidc_idri`.
- This is the preferred route for paper-quality reproducibility.

## Step 4: Student-Initialized Transfer Baseline

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --output-dir outputs/lidc_bvm_resnet3d18_student_init_fold0 \
  --model resnet3d18 \
  --use-3d-input \
  --class-mode binary \
  --binary-task benign_vs_malignant \
  --use-predefined-split \
  --split-manifest-csv /workspace/data-lung/lidc_idri_consensus_3d_fold0/processed_split_manifest.csv \
  --split-fold 0 \
  --split-mode train_val_test \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --selection-metric auroc \
  --init-checkpoint outputs/ct_student_kd_v1_tvt/best_model.pt \
  --init-checkpoint-prefix ct_encoder.
```

Compare against the clean baseline on the same fold:

- `AUROC`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `MCC`

## Recommended Reporting

Do not report only one fold. Prefer:

1. run all `5` folds
2. export per-fold metrics
3. summarize `mean ± std`

For paper framing:

- internal real-data task validates the multimodal teacher-student idea
- `LIDC-IDRI` validates CT representation transfer on a public benchmark
- it should be written as `public transfer validation`, not same-task external validation
