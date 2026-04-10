# 内网 CT DICOM 质控与 NPY 重建说明

本工具用于三分类内网 CT 线的数据重建，目标是把原始 DICOM 重新按同一口径转换成 `.npy`，并生成当前训练入口可直接读取的 manifest。

入口：

- `prepare_intranet_ct_npy.py`
- 核心实现：`src/lung_cancer_cls/intranet_ct_preprocess.py`

默认输出体数据尺寸：

- `128 x 256 x 256`
- 对应当前历史路径里的 `npy_128_256_256`

默认 HU 窗口：

- `[-1000, 400]`
- 输出为 `float32`，范围约为 `[0, 1]`

当前训练时仍然走：

```bash
python train.py \
  --dataset-type intranet_ct \
  --metadata-csv <NEW_MANIFEST_CSV> \
  --ct-root <NPY_OUTPUT_ROOT> \
  --use-3d-input \
  --class-mode multiclass
```

## 1. 先做 QC，不写 NPY

建议先只扫描旧 CSV 里的原始 DICOM 路径，输出层厚、spacing、层数、尺寸等 series 级 QC 表：

```bash
python prepare_intranet_ct_npy.py \
  --source-csv <OLD_MULTI_MODAL_CSV> \
  --source-data-root /home/apulis-dev/userdata/Data \
  --output-root <NPY_OUTPUT_ROOT> \
  --manifest-out outputs/intranet_rebuild_manifest_plan.csv \
  --qc-csv outputs/intranet_rebuild_qc.csv \
  --summary-json outputs/intranet_rebuild_summary.json \
  --scan-only
```

说明：如果旧 CSV 里的 `CT dicom路径` 仍是 `Z:\CT数据 20251120\...`，上传到服务器后不要直接使用旧路径。加上：

```bash
  --source-data-root /home/apulis-dev/userdata/Data
```

工具会按病例目录名把旧路径重建到：

- `/home/apulis-dev/userdata/Data/健康对照_原始/健康对照/<病例目录>`
- `/home/apulis-dev/userdata/Data/良性结节+肺癌_原始/良性结节+肺癌/<病例目录>`

如果实际目录名或层级和上面不一致，可以用显式映射：

```bash
  --source-path-map "Z:\CT数据 20251120\健康对照_find1mm_fix1124\肺窗1mm标准=/home/apulis-dev/userdata/Data/健康对照_原始/健康对照" \
  --source-path-map "Z:\CT数据 20251120\良性结节+肺癌_find1mm_fix1124\肺窗近1mm=/home/apulis-dev/userdata/Data/良性结节+肺癌_原始/良性结节+肺癌"
```

重点先看：

- `intranet_rebuild_qc.csv`
- `intranet_rebuild_summary.json`
- `intranet_rebuild_summary_skipped_cases.csv`

说明：

- 如果你用的是同一份混合 `source_csv`，里面同时包含健康、良性、恶性三类，那么一次运行就会把这三类都一起扫描出来。
- 即使原始 DICOM 实际存放在不同目录，只要 `source_csv` 里有这些行，并且能通过 `--source-data-root` 或 `--source-path-map` 重定位到对应目录，结果仍然会是三类一起进入同一个 QC / manifest。
- 如果你想分开跑，不能只换原始目录位置；要么拆成不同的 CSV 分别运行，要么改用 `--input-root normal=... --input-root benign=... --input-root malignant=...` 只指定某一类目录。

默认筛选条件：

- `min_slices >= 32`
- `rows/columns >= 128`
- `slice_thickness <= 5.0mm`
- `z_spacing <= 5.0mm`
- `xy_spacing <= 2.5mm`
- 排除明显的 localizer/scout/topogram 等非诊断序列

如果想先严格筛薄层，可以把阈值调小，例如：

```bash
  --max-slice-thickness-mm 2.5 \
  --max-z-spacing-mm 2.5
```

## 2. 重建旧 CSV 对应 NPY

确认 QC 后，再真正转换：

```bash
python prepare_intranet_ct_npy.py \
  --source-csv <OLD_MULTI_MODAL_CSV> \
  --source-data-root /home/apulis-dev/userdata/Data \
  --output-root <NPY_OUTPUT_ROOT> \
  --manifest-out outputs/intranet_rebuild_manifest.csv \
  --qc-csv outputs/intranet_rebuild_qc.csv \
  --summary-json outputs/intranet_rebuild_summary.json \
  --overwrite
```

生成的 `outputs/intranet_rebuild_manifest.csv` 会保留原 CSV 行中的主要字段，并更新：

- `CT dicom路径`
- `CT_numpy路径`
- `CT_numpy_cloud路径`
- `样本类型`
- `CT_train_val_split`

转换时如果遇到极少数“读出来不是普通 3D 灰度 CT”的异常序列，例如多通道 `RGB` / `secondary capture` 风格的 DICOM，工具现在会把该病例写入 `*_skipped_cases.csv`，并以 `conversion_failed:...` 记录原因，而不是让整批任务中断。

训练命令示例：

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <NPY_OUTPUT_ROOT> \
  --metadata-csv outputs/intranet_rebuild_manifest.csv \
  --ct-root <NPY_OUTPUT_ROOT> \
  --output-dir outputs/intranet_ct_rebuild_resnet3d18 \
  --class-mode multiclass \
  --model resnet3d18 \
  --use-3d-input \
  --depth-size 128 \
  --volume-hw 256 \
  --epochs 40 \
  --batch-size 1 \
  --lr 3e-4 \
  --aug-profile strong \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num
```

说明：这条示例保持与你当前 86% 左右实验一致的训练输入口径：`128 x 256 x 256`。如果后续为了快速排查数据质量或做消融，也可以临时改成 `--depth-size 32 --volume-hw 128` 或 `--depth-size 64 --volume-hw 160/192`，但那应该作为对照实验，而不是替代当前主线口径。

## 3. 处理新增良性 500 例

新增良性目录建议先单独 QC：

```bash
python prepare_intranet_ct_npy.py \
  --input-root 良性结节=/home/apulis-dev/userdata/Data/良性患者500例 \
  --root-split-mode train_val_test \
  --output-root <BENIGN_NPY_OUTPUT_ROOT> \
  --manifest-out outputs/benign500_manifest_plan.csv \
  --qc-csv outputs/benign500_qc.csv \
  --summary-json outputs/benign500_summary.json \
  --scan-only
```

确认层厚和质量后转换：

```bash
python prepare_intranet_ct_npy.py \
  --input-root 良性结节=/home/apulis-dev/userdata/Data/良性患者500例 \
  --root-split-mode train_val_test \
  --output-root <BENIGN_NPY_OUTPUT_ROOT> \
  --manifest-out outputs/benign500_manifest.csv \
  --qc-csv outputs/benign500_qc.csv \
  --summary-json outputs/benign500_summary.json \
  --overwrite
```

说明：

- `--input-root 良性结节=...` 会把该目录下的每个一级子目录当作一个病例。
- 如果某个病例目录下有多个 DICOM series，默认选择最像肺部诊断序列的一条：优先 `lung/chest/肺/胸`、薄层、层数更多、spacing 更小。
- 如果 SimpleITK 目录级检测提示 `No Series were found`，工具会自动退回逐文件读取 DICOM header 并按 `SeriesInstanceUID` 分组；默认会压掉 SimpleITK/GDCM 的原始 warning 噪声，具体是否扫到、是否合格以 `benign500_qc.csv` 和 `benign500_summary_skipped_cases.csv` 为准。需要调试原始 warning 时再加 `--show-sitk-warnings`。
- `--root-split-mode train_val_test` 会给新增良性病例生成 `train/val/test` 划分；如果后续你决定全量重新随机划分，也可以设成 `blank`，训练时不要加 `--use-predefined-split`。

如果你打算先转换、后续再人工复核，推荐把 `5mm` 边缘样本单独标记，而不是直接删文件：

```bash
python prepare_intranet_ct_review_flags.py \
  --manifest-csv outputs/benign500_manifest.csv \
  --output-csv outputs/benign500_review_flags.csv
```

这会生成一张独立的复核标记表，默认规则是：

- `preprocess_slice_thickness >= 5.0` 或 `preprocess_spacing_z >= 5.0`：标成 `qc_bucket=thick5_borderline`
- `thick5_borderline` 默认写成 `review_status=pending`、`use_for_training=0`
- 其他样本默认写成 `review_status=auto_pass`、`use_for_training=1`

建议人工复核时只改这几列：

- `review_status`：例如改成 `manual_pass` 或 `reject`
- `use_for_training`：通过改成 `1`，排除保持 `0`
- `review_note`：记录原因，例如 `localizer`、`coverage_incomplete`、`wrong_series`

复核完成后，再把 flags 应用回 manifest，导出训练用 CSV：

```bash
python apply_intranet_ct_review_flags.py \
  --manifest-csv outputs/benign500_manifest.csv \
  --flags-csv outputs/benign500_review_flags.csv \
  --output-csv outputs/benign500_manifest_reviewed.csv \
  --audit-csv outputs/benign500_manifest_with_review.csv
```

说明：

- `benign500_manifest_reviewed.csv` 只保留 `use_for_training=1` 的样本，适合直接喂训练。
- `benign500_manifest_with_review.csv` 会保留全部样本并附上复核列，方便留审计记录。
- 这样后续即使发现问题样本，也优先改 `review_flags.csv` 和 reviewed manifest，不必只靠手工删 `.npy` 管理。

## 4. 合并旧数据和新增良性

转换完成后可以用 pandas 或表格工具合并：

- `outputs/intranet_rebuild_manifest.csv`
- `outputs/benign500_manifest.csv`

合并后的 CSV 继续作为 `--metadata-csv`，`--ct-root` 指向统一的 `.npy` 根目录。

更推荐的最终形态是让 `<NPY_OUTPUT_ROOT>` 成为统一根目录，例如：

```text
<NPY_OUTPUT_ROOT>/
  normal/
  benign/
  malignant/
```

这样合并后的 `CT_numpy_cloud路径` 都是相对路径，当前 `IntranetCTDataset` 可以直接读取。

## 5. Bundle 数据

`bundle` 数据目前优先级较低。后续如果要纳入，建议不要直接混用旧 bundle 的 `.npy`，而是尽量回到原始 DICOM，用本工具同一套 HU 窗口、尺寸和 QC 阈值重建后再加入训练。
