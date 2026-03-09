import os
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from monai.transforms import RandFlip, RandRotate, Compose
from transformers import BertTokenizer, BertModel


# 全局配置
FILENAN = "PANDASNAN"
LABEL_DICT = {"健康对照": 0, "良性结节": 1, "肺癌": 2}
ROOT_DIR = '/home/apulis-dev/userdata/Data/CT1500'  # CT数据根目录
MULTI_TABLE_PATH = '/home/apulis-dev/userdata/mmy/ct/多模态统一检索表_CT本地路径_CT划分.csv'
TEXT_HEALTH_PATH = "/home/apulis-dev/userdata/mmy/text/lung0.csv"
TEXT_DISEASE_PATH = "/home/apulis-dev/userdata/mmy/text/lung1.csv"
GENE_DATA_PATH = "/home/apulis-dev/userdata/Data/Gene/final_pca_data.tsv"


def split_dataset_by_label(
    dataset: Dataset,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[Subset, Subset]:
    """按标签分层划分训练/测试集（适配多模态Dataset）"""
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio 必须在(0, 1)之间")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 获取标签并分组
    labels = dataset.get_labels()
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # 分层划分
    train_indices, test_indices = [], []
    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        if shuffle:
            np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_idx])
        test_indices.extend(indices[split_idx:])

    # 验证无重叠
    if set(train_indices) & set(test_indices):
        raise ValueError("训练/测试集存在重叠！")

    print(f"📊 分层划分完成：总样本{len(dataset)} | 训练集{len(train_indices)} | 测试集{len(test_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

class RandomMaskModalTransform:
    def __init__(self, mask_prob=0.1):
        self.mask_prob = mask_prob  # 每个模态被掩盖的概率

    def __call__(self, ct_tensor, text_tensor, gene_tensor, modal_mask):
        # 随机掩盖CT
        if random.random() < self.mask_prob and modal_mask[0]:
            ct_tensor = torch.zeros_like(ct_tensor)
            modal_mask[0] = False

        # 随机掩盖文本
        if random.random() < self.mask_prob and modal_mask[1]:
            text_tensor = torch.zeros_like(text_tensor)
            modal_mask[1] = False

        # 随机掩盖基因
        if random.random() < self.mask_prob and modal_mask[2]:
            gene_tensor = torch.zeros_like(gene_tensor)
            modal_mask[2] = False

        return ct_tensor, text_tensor, gene_tensor, modal_mask

class MultiModalDataset(Dataset):
    def __init__(
        self,
        valid_samples: List[Dict],
        text_emb_dict: Dict[str, np.ndarray],
        gene_feat_dict: Dict[str, np.ndarray],
        transform: Optional[Compose] = None,
        img_size: Tuple[int, int, int] = (256,128,256),
        target_feat_dim: int = 512,  # 统一特征维度
        gene_in_dim: int = 716,
        text_in_dim: int = 768,
        mode: str = "train"
    ):
        self.valid_samples = valid_samples
        self.text_emb_dict = text_emb_dict
        self.gene_feat_dict = gene_feat_dict
        self.transform = transform
        self.img_size = img_size
        self.target_feat_dim = target_feat_dim
        self.gene_in_dim = gene_in_dim
        self.text_in_dim = text_in_dim
        self.mode = mode

        # 标签处理（类别索引）
        self.labels = torch.tensor([s["label"] for s in valid_samples]).long()

        self.modal_mask_transform = RandomMaskModalTransform(mask_prob=0.1) if self.mode == "train" else None

    def __len__(self):
        return len(self.valid_samples)

    def get_labels(self) -> List[int]:
        return [s["label"] for s in self.valid_samples]

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        sample = self.valid_samples[idx]
        modal_present = sample["modal_present"]

        ct_tensor = torch.zeros((1, *self.img_size))  # CT特征维度 (1, D, H, W)
        text_tensor = torch.zeros((self.text_in_dim,))  # 文本嵌入维度
        gene_tensor = torch.zeros((self.gene_in_dim,))  # 基因特征维度
        modal_mask = torch.tensor([modal_present["ct"], modal_present["text"], modal_present["gene"]], dtype=torch.float32).clone()

        if modal_present["ct"] and sample["ct_path"]:
            ct_data = np.load(sample["ct_path"]).astype(np.float32)
            ct_data = np.expand_dims(ct_data, axis=0)  # 增加通道维度
            ct_tensor = ct_data.transpose(0, 3, 1, 2)  # 调整为 (1, D, H, W)
            ct_tensor = np.ascontiguousarray(ct_tensor)
            # ct_tensor = MetaTensor(torch.tensor(ct_tensor), meta={})
            if self.transform:
                ct_tensor = self.transform(ct_tensor)
            if isinstance(ct_tensor, np.ndarray):
                ct_tensor = torch.from_numpy(ct_tensor)
            elif hasattr(ct_tensor, 'as_tensor'):
                ct_tensor = ct_tensor.as_tensor().clone().detach()
            else:
                ct_tensor = torch.as_tensor(ct_tensor)
        ct_tensor = ct_tensor.as_subclass(torch.Tensor).clone().detach()
        # print("3D输入形状ct_tensor",ct_tensor.shape)
        # 加载文本嵌入（如果存在）
        if modal_present["text"] and sample["record_id"]:
            text_emb = self.text_emb_dict[sample["record_id"]].astype(np.float32)
            # 投影到目标维度（如果需要）
            # if text_emb.shape[0] != self.target_feat_dim:
            #     text_emb = np.pad(text_emb, (0, self.target_feat_dim - text_emb.shape[0]), mode='constant')
            text_tensor = torch.from_numpy(text_emb).clone()

        # 加载基因特征（如果存在）
        if modal_present["gene"] and sample["gene_id"]:
            gene_feat = self.gene_feat_dict[sample["gene_id"]].astype(np.float32)
            # 投影到目标维度（如果需要）
            # if gene_feat.shape[0] != self.target_feat_dim:
            #     gene_feat = np.pad(gene_feat, (0, self.target_feat_dim - gene_feat.shape[0]), mode='constant')
            gene_tensor = torch.from_numpy(gene_feat).clone()

        # 加载标签
        label = self.labels[idx].long()

        if self.mode == "train" and self.modal_mask_transform:
            ct_tensor, text_tensor, gene_tensor, modal_mask = self.modal_mask_transform(ct_tensor, text_tensor, gene_tensor, modal_mask)

        return (ct_tensor.contiguous(), text_tensor.contiguous(), gene_tensor.contiguous(), modal_mask.contiguous()), label


def load_text_embeddings() -> Dict[str, np.ndarray]:
    """加载文本数据并提取BERT嵌入（关联record_id）"""
    # 读取并合并文本表
    try:
        df_health = pd.read_csv(TEXT_HEALTH_PATH)
        df_disease = pd.read_csv(TEXT_DISEASE_PATH)
    except FileNotFoundError as e:
        raise Exception(f"文本文件缺失: {e}")

    # 统一列结构 + 合并
    all_cols = df_health.columns.union(df_disease.columns)
    df_health = df_health.reindex(columns=all_cols, fill_value=np.nan)
    df_disease = df_disease.reindex(columns=all_cols, fill_value=np.nan)
    df_text = pd.concat([df_health, df_disease], axis=0, ignore_index=True)

    # 过滤无record_id的样本
    df_text = df_text.dropna(subset=["record_id"]).reset_index(drop=True)

    # 合并文本列（入院记录/病史等）
    text_cols = [col for col in df_text.columns if "入院记录" in col or "史" in col]
    df_text["combined_text"] = df_text[text_cols].fillna("无").apply(
        lambda x: " ".join(x.astype(str)), axis=1
    )

    # 加载BERT（本地预下载，避免网络依赖）
    tokenizer = BertTokenizer.from_pretrained("/home/apulis-dev/userdata/mmy/text/bert")  # 本地路径可替换为'./bert-base-chinese'
    bert_model = BertModel.from_pretrained("/home/apulis-dev/userdata/mmy/text/bert").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = bert_model.to(device)

    # 提取BERT嵌入（CLS token）
    text_emb_dict = {}
    for _, row in df_text.iterrows():
        record_id = row["record_id"]
        text = row["combined_text"]
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        text_emb_dict[record_id] = cls_emb

    print(f"📝 文本嵌入加载完成：共{len(text_emb_dict)}个record_id")
    return text_emb_dict


def load_gene_features() -> Dict[str, np.ndarray]:
    """加载基因数据（关联测序样本ID）"""
    df_gene = pd.read_csv(GENE_DATA_PATH, sep="\t")
    gene_id_col = df_gene.columns[0]  # 假设第一列为测序样本ID（RDLP_xxx）
    label_col = df_gene.columns[1] 
    # 过滤无基因ID的样本
    df_gene = df_gene.dropna(subset=[gene_id_col]).reset_index(drop=True)

    # 构建基因ID -> 特征映射（处理缺失值）
    gene_feat_dict = {}
    for _, row in df_gene.iterrows():
        gene_id = row[gene_id_col]
        # 提取特征列（排除ID列）
        # features = row.drop(gene_id_col).values.astype(np.float32)
        features = row.drop([gene_id_col, label_col]).values.astype(np.float32)
        # 填充缺失值（列均值）
        nan_mask = np.isnan(features)
        if np.any(~nan_mask):
            features[nan_mask] = np.nanmean(features[~nan_mask])
        else:
            features[nan_mask] = 0.0  # 全NaN则填充0
        gene_feat_dict[gene_id] = features

    print(f"🧬 基因特征加载完成：共{len(gene_feat_dict)}个测序样本ID")
    return gene_feat_dict

def filter_valid_samples(
    text_emb_dict: Dict[str, np.ndarray],
    gene_feat_dict: Dict[str, np.ndarray]
) -> List[Dict]:
    """保留有至少1个有效模态的样本，并记录模态存在情况"""
    df_multi = pd.read_csv(MULTI_TABLE_PATH)
    df_multi = df_multi.fillna(FILENAN).reset_index(drop=True)  # 填充缺失值为统一标记

    valid_samples = []
    for _, row in df_multi.iterrows():
        sample = {
            "ct_path": None,
            "record_id": None,
            "gene_id": None,
            "label": None,
            "modal_present": {"ct": False, "text": False, "gene": False},  # 记录模态是否存在
            "split": row["CT_train_val_split"]
        }

        # 检查CT模态
        ct_path = row["CT_numpy_cloud路径"].replace("\\", "/") if row["CT_numpy_cloud路径"] != FILENAN else None
        if ct_path:
            full_ct_path = os.path.join(ROOT_DIR, ct_path)
            if os.path.exists(full_ct_path):
                sample["ct_path"] = full_ct_path
                sample["modal_present"]["ct"] = True

        # 检查文本模态
        record_id = row["record_id"] if row["record_id"] != FILENAN else None
        if record_id and record_id in text_emb_dict:
            sample["record_id"] = record_id
            sample["modal_present"]["text"] = True

        # 检查基因模态
        gene_id = row["SampleID"] if row["SampleID"] != FILENAN else None
        if gene_id and gene_id in gene_feat_dict:
            sample["gene_id"] = gene_id
            sample["modal_present"]["gene"] = True

        # 检查标签有效性
        sample_type = row["样本类型"]
        if sample_type in LABEL_DICT:
            sample["label"] = LABEL_DICT[sample_type]

        # 保留有至少1个模态且标签有效的样本
        if any(sample["modal_present"].values()) and sample["label"] is not None:
            valid_samples.append(sample)

    print(f"✅ 有效样本数（至少1个模态）：{len(valid_samples)}")
    return valid_samples


if __name__ == "__main__":
    # 1. 加载各模态数据
    text_emb_dict = load_text_embeddings()
    gene_feat_dict = load_gene_features()
    valid_samples = filter_valid_samples(text_emb_dict, gene_feat_dict)

    # 2. 构建多模态Dataset
    train_transform = Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=[-20, 20], prob=0.5)
    ])
    dataset = MultiModalDataset(
        valid_samples=valid_samples,
        text_emb_dict=text_emb_dict,
        gene_feat_dict=gene_feat_dict,
        transform=train_transform
    )

    # 3. 划分训练/测试集
    train_subset, test_subset = split_dataset_by_label(dataset, train_ratio=0.8)

    # 4. 创建DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # 测试：打印一个batch的形状
    print("\n📦 测试Batch形状：")
    for (ct_batch, text_batch, gene_batch), label_batch in train_loader:
        print(f"CT: {ct_batch.shape} | 文本嵌入: {text_batch.shape} | 基因特征: {gene_batch.shape} | 标签: {label_batch.shape}")
        break