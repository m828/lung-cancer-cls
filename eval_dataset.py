import os
import pandas as pd
import numpy as np
# from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
import random
from scipy import ndimage
from monai.transforms import RandRotate90, RandFlip, RandRotate, Compose

FILENAN = "PANDASNAN"
# 1. 自定义PyTorch Dataset # -----------------------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, label_column='样本类型（处理）', 
            feature_columns='CT_numpy_cloud路径', 
            transform=None, label_dict={"健康对照":0,"良性结节":1,"肺癌":2},
            mode="train", train_ratio=0.8, seed=42, rand_trainval = False,
            return_sample = False):
        """
        Args:
            csv_path (str): CSV 文件路径
            root_dir (str): 图像/数据文件根目录（path 列的前缀            
            label_column (str): 标签列名
            feature_columns (list or None): 
            transform (callable, optional): 数据增强或预处理变换
        """
        df = pd.read_csv(csv_path)
        self.org_df = df

        self.root_dir = root_dir.replace("\\", "/")
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.label_dict = label_dict
        self.return_sample = return_sample

        self.mode = mode
        self.train_ratio = train_ratio
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rand_trainval = rand_trainval

        self.org_labels, self.paths, self.samples, self.trainval_label = self.df_prepose(df)
        self.train_val_split(self.org_labels, self.rand_trainval)
        self.labels = torch.nn.functional.one_hot(torch.as_tensor(self.org_labels).long()).float()


        # print(self.labels)
        print(f"数据集加载完成，{len(self)} 个样本")
        print(f"标签映射: {self.label_dict}")

    def train_val_split(self, labels, shuffle = False):
        # labels = self.org_labels #直接读取labels
        if self.mode == "all":#使用全部数据，不进行划分
            return list(range(len(labels)))
        if self.mode in ["pre_train", "pre_val"]:#通过之前的标签划            
            print(f"📊 数据集使用之前的标签划分")
            # print(f"📊 目前所有没有添加标注的数据当初测试")
            print(f"📊 目前所有没有添加标注的数据当初训练")
            train_indices = []
            test_indices = []
            for i, name in enumerate(self.trainval_label):
                # if name == "训练集":
                #     train_indices.append(i)
                # else:
                #     test_indices.append(i)
                if name == "测试集":
                    test_indices.append(i)
                else:
                    train_indices.append(i)
        else:
            # label 分组索引
            label_to_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                label_to_indices[label].append(idx)
            # 存储训练和测试的索引
            train_indices = []
            test_indices = []
            if shuffle:
                print(f"📊 数据集随机划分")
            for label, indices in label_to_indices.items():
                indices = np.array(indices)
                # print("indices before shuffle")
                # print(indices)
                if shuffle:
                    np.random.shuffle(indices) #没有随机划分
                # print("indices after shuffle")
                # print(indices)
                # 计算划分                
                n = len(indices)
                split_idx = int(n * self.train_ratio)
                train_idx = indices[:split_idx]
                test_idx = indices[split_idx:]
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
            # 转为列表（PyTorch Subsets 需要）
            train_indices = list(train_indices)
            test_indices = list(test_indices)
        print(f"📊 数据划分完成")
        print(f"   - 总样本数: {self.__len__()}")
        print(f"   - 训练集 {len(train_indices)} ({len(train_indices)/self.__len__()*100:.1f}%)")
        print(f"   - label 分组划分，每个label 保持比例")
        print(f"   - 测试集 {len(test_indices)} ({len(test_indices)/self.__len__()*100:.1f}%)")
        print(f"   - label 分组划分，每个label 保持比例")
        # print(set(train_indices))
        # print(set(test_indices))
        # 检查是否冲突（交集为空)        
        if set(train_indices) & set(test_indices):
            raise ValueError("训练集和测试集存在重叠索引！划分失败")
        self.org_df['CT_train_val_split'] = 'None'  # 默认设为 train
        self.org_df.loc[self.org_df.index.isin([self.index_in_orgdf[k] for k in test_indices]), 'CT_train_val_split'] = 'test'
        self.org_df.loc[self.org_df.index.isin([self.index_in_orgdf[k] for k in train_indices]), 'CT_train_val_split'] = 'train'
        # print(f"📊 保存当此的数据划分到：多模态统一检索表_CT本地路径_CT划分.csv")
        # self.org_df.to_csv('/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_CT划分.csv'  , index=False)
    
        if self.mode in ["train", "pre_train"]:
            # print(f"   - 训练集 {len(train_indices)} ({len(train_indices)/self.__len__()*100:.1f}%)")
            # print(f"   - label 分组划分，每个label 保持比例")
            self.org_labels = [self.org_labels[i] for i in train_indices]
            self.paths = [self.paths[i] for i in train_indices]
            self.samples = [self.samples[i] for i in train_indices]
            return train_indices
        else:
            print(f"   - 测试集 {len(test_indices)} ({len(test_indices)/self.__len__()*100:.1f}%)")
            print(f"   - label 分组划分，每个label 保持比例")
            self.org_labels = [self.org_labels[i] for i in test_indices]
            self.paths = [self.paths[i] for i in test_indices]
            self.samples = [self.samples[i] for i in test_indices]
            return test_indices

    def df_prepose(self, org_df):
        labels = []
        paths = []
        samples = []
        trainval_label = []
        self.index_in_orgdf = []
        org_df = org_df.fillna(FILENAN)#处理其中的nan
        for index, row in org_df.iterrows():
            if row[self.label_column] not in self.label_dict:
                # print(f"文件类型不在处理范围，类型：{row[self.label_column]}，样本名字：{row["样本编号"]}")
                pass
            elif row[self.feature_columns] == FILENAN:
                pass
                # print(f"在原始数据中，文件缺失，样本名字：{row["样本编号"]}")
            # elif "肺窗1mm标准" not in row[self.feature_columns]:
            #     print(f"目前暂时只使用肺部mm标准的数据作为测试集，样本名字：{row["样本编号"]}")
            else:
                abs_path = os.path.join(self.root_dir, row[self.feature_columns].replace("\\", "/"))
                if os.path.exists(abs_path):
                    labels.append(self.label_dict[row[self.label_column]])
                    paths.append(abs_path)
                    # samples.append(row["样本编号"])
                    samples.append(row["测序样本ID"])
                    trainval_label.append(row["CT_train_val_split"]) #现在使用CT划分
                    # gene_trainval_label.append(row[""])
                    self.index_in_orgdf.append(index)
                # else:
                #     print(f"在云上以下文件找不到，可能是传输错误，文件{abs_path}，标签{row[self.label_column]}")
        return labels, paths, samples, trainval_label

    def __len__(self):
        return len(self.paths)
    
    def get_labels(self):
        return self.org_labels
    
    def set_mode(self, mode):
        assert mode in ["tarin", "val"]
        self.mode = mode

    def __getitem__(self, idx):
        # 读取路径
        full_path = self.paths[idx]
        # full_path = os.path.join(self.root_dir, file_path)
        # print(full_path)
        # 使用 numpy 读取数据（假设是 .npy 文件        
        data = np.load(full_path)  # shape: (C, H, W) (D,)
        
        # 修改大小
        # from monai.transforms import Resize
        # # print("3D输入形状old",data.shape)
        # spatial_size = (128,128,128)
        # data = data[None, ...]
        # resizer = Resize(spatial_size=spatial_size)
        # data = resizer(data)
        # data = data[0]
        # print("3D输入形状new",data.shape)
        # 获取标签s
        label = self.labels[idx].long()
        sample = self.samples[idx]

        # 应用变换（可选）
        # if self.transform:
        #     data = self.transform(data)
        if self.mode == "train":
            # data = self.rotate(data)
            data = self.transform(data)
        # 转为 float32（PyTorch 常用）        
        data = data.astype(np.float32)
        ct_tensor = np.expand_dims(data, 0)
        # print("3D输入形状",data.shape)
        text_tensor = torch.zeros((768,))  # 文本嵌入维度
        gene_tensor = torch.zeros((716,))  # 基因特征维度
        modal_mask = torch.tensor([True, False, False], dtype=torch.float32).clone()


        if self.return_sample:
            return (ct_tensor, text_tensor, gene_tensor, modal_mask), label, sample
        else:
            return (ct_tensor, text_tensor, gene_tensor, modal_mask), label

# -----------------------------
# 2. 主函数：读取 CSV，转换为 0,1,2 标签，保存为 numpy
# -----------------------------

def get_ft_dataset():
    csv_path = '/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_CT划分.csv'          # 你的 CSV 文件路径
    root_dir = '/home/apulis-dev/userdata/Data/CT1500'            # 图像/数据文件所在的根目录（path 列中的相对路径）
    output_dir = './output'        # 输出 .npy 文件目录

    # train_transformers = Compose([RandFlip(),RandRotate90()])
    train_transformers = Compose([RandFlip(),RandRotate(range_x=[-20, -10, -5, 5, 10, 20])])
    # 执行数据准备
    train_ds = CSVImageDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        label_column='样本类型',
        feature_columns="CT_numpy_cloud路径",  # 自动处理
        transform=train_transformers,
        label_dict={"健康对照":0,"良性结节":1,"肺癌":2},
        mode="train", 
        train_ratio=0.8, 
        seed=42,
        rand_trainval = True,
    )

    val_ds = CSVImageDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        label_column='样本类型',
        feature_columns="CT_numpy_cloud路径",  # 自动处理
        transform=None,
        label_dict={"健康对照":0,"良性结节":1,"肺癌":2},
        mode="val", 
        train_ratio=0.8, 
        seed=42,
    )
    return train_ds, val_ds
# -----------------------------
# 3. 使用示例
# -----------------------------
if __name__ == "__main__":
    # 配置路径
    # csv_path = '/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径.csv'          # 你的 CSV 文件路径
    csv_path = '/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_CT划分.csv'   
    root_dir = '/home/apulis-dev/userdata/Data/CT1500'            # 图像/数据文件所在的根目录（path 列中的相对路径）
    output_dir = './output'        # 输出 .npy 文件目录

    train_ds, val_ds = get_ft_dataset()
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )