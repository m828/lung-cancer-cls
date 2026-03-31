# CT + CNV Multimodal V1

## 1. Goal

This v1 script builds a practical `CT + CNV` teacher baseline for the current binary task:

- `--class-mode binary`
- `--binary-task malignant_vs_normal`

The focus is:

- align CT and CNV on the same patient set
- keep split handling consistent with the CT/CNV single-modal baselines
- provide a first fusion model that is ready for later KD experiments

## 2. Entry Script

- `train_ct_cnv.py`

Core implementation:

- `src/lung_cancer_cls/multimodal_ct_cnv.py`

## 3. Model Design

Current v1 uses late fusion:

- CT encoder: reuse the existing CT backbone registry
- CNV encoder: small MLP
- fusion head: concatenate CT embedding and CNV embedding, then classify

By default:

- CT backbone: `resnet3d18`
- CT input: `3D`

You can also switch to another existing CT backbone, for example:

- `attention3d_cnn`
- `densenet3d_121` if `monai` is installed

## 4. Recommended Command

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_resnet3d18_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

## 5. Recommended Teacher Ablations

Current stage does not recommend opening too many branches at once.

The most practical first batch is:

1. `CT only` reference
2. `CT + CNV` main teacher: `resnet3d18`
3. `CT + CNV` lightweight control: `attention3d_cnn`
4. `CT + CNV` 2D control: `resnet3d18` + `--disable-3d-input`

Suggested naming:

- `outputs/ct_only_resnet3d18_mvn_formal`
- `outputs/ct_cnv_resnet3d18_mvn_v1`
- `outputs/ct_cnv_attention3d_mvn_v1`
- `outputs/ct_cnv_resnet3d18_2d_mvn_v1`

Example commands:

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_attention3d_mvn_v1 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model attention3d_cnn \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_resnet3d18_2d_mvn_v1 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --disable-3d-input \
  --image-size 224 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

## 6. Outputs

The script writes:

- `best_model.pt`
- `metrics.json`
- `train_predictions.csv`
- `val_predictions.csv`
- `test_predictions.csv`

Key metrics to watch:

- `auroc`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `f1`

## 7. Exporting The Ablation Table

After the first teacher batch is finished, export a unified result table:

```bash
python export_experiment_table.py \
  --run ct_only=outputs/ct_only_resnet3d18_mvn_formal \
  --run teacher_main=outputs/ct_cnv_resnet3d18_mvn_v1 \
  --run teacher_light=outputs/ct_cnv_attention3d_mvn_v1 \
  --run teacher_2d=outputs/ct_cnv_resnet3d18_2d_mvn_v1 \
  --output-dir outputs/teacher_ablation_table
```

The exporter writes:

- `experiment_table.csv`
- `experiment_table.md`
- `summary.json`

## 8. Notes

- This is a `v1` fusion baseline, not the final teacher architecture.
- The CT encoder currently reuses the existing backbone registry by treating the backbone output as a learnable CT embedding.
- The next decision point is whether `CT + CNV` actually beats `CT only` on the aligned test set. Only after that is it worth moving on to KD.
