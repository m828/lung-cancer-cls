import argparse
import datetime
import logging
import os
import sys
import random
from contextlib import nullcontext
from tqdm import tqdm

# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data.dataloader as dataloader
import torchvision.utils as vutils

import monai
# import torchio as tio
from monai.losses import DiceCELoss
from monai.apps import download_and_extract
from monai.config import print_config
# from monai.data import DataLoader
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='densenet256_BT_NM_dzh_128128size')
    parser.add_argument('--click_type', type=str, default='random')
    parser.add_argument('--multi_click', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='vit_b_ori')
    parser.add_argument('--checkpoint', type=str, default='C:\\Users\\yinyiyang\\Downloads\\sam_med3d_turbo.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--work_dir', type=str, default='work_dir')

    # train
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--allow_partial_weight', action='store_true', default=False)

    # data
    parser.add_argument('--nm_data_path', type=str, default="/home/apulis-dev/userdata/processed/NM_all.npy")
    parser.add_argument('--bn_data_path', type=str, default="/home/apulis-dev/userdata/processed/BN_all.npy")
    parser.add_argument('--mt_data_path', type=str, default="/home/apulis-dev/userdata/processed/MT_all.npy")
    parser.add_argument('--nm_bn_class', action='store_true', default=False)
    parser.add_argument('--reverse_data', action='store_true', default=False)

    # eval 
    parser.add_argument('--eval_show_steps', type=int, default=20)

    # lr_scheduler
    parser.add_argument('--loss_type', type=str, default='BCELoss')
    parser.add_argument('--lr_scheduler', type=str, default='exponential')
    parser.add_argument('--step_size', type=list, default=[120, 180])
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--port', type=int, default=12361)
    return parser



def get_dataloaders(args):
    from datasets.data_process2 import MultiModalDataset, load_text_embeddings, load_gene_features, filter_valid_samples, split_dataset_by_label
    from monai.transforms import RandFlip, RandRotate, Compose

    if 'cuda' in args.device:
        pin_memory = True
    else:
        pin_memory = False

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
        transform=train_transform,
        mode="all"
    )

    # 3. 划分训练/测试集
    train_subset, val_subset = split_dataset_by_label(dataset, train_ratio=0.8)

    # 设置mode
    train_subset.dataset.mode = "train"
    val_subset.dataset.mode = "val"
    train_subset.dataset.modal_mask_transform = RandomMaskModalTransform(mask_prob=0.1) if train_subset.dataset.mode == "train" else None
    val_subset.dataset.modal_mask_transform == None 

    # 4. 创建DataLoader
    # train_loader = DataLoader(
    #     train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory, collate_fn=dataloader.default_collate
    # )
    # val_loader = DataLoader(
    #     val_subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=pin_memory, collate_fn=dataloader.default_collate
    # )
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    return train_loader, val_loader


def build_model(args):
    from model.base_model import get_MLP_model, CNN3D, MultiModel, RobustMultiModalModel
    from datasets.data_process2 import load_gene_features

    gene_feat_dict = load_gene_features()
    sample_gene_feat = next(iter(gene_feat_dict.values()))
    gene_in_dim = sample_gene_feat.shape[0]
    # print("!!!!!",gene_in_dim)
    model = RobustMultiModalModel(
        gene_in_dim=gene_in_dim, text_dim=768, num_classes=3
    ).to(args.device)
    # model = monai.networks.nets.resnet50(spatial_dims=3,n_input_channels=1, num_classes=3, norm=("GROUP",{"num_groups":32})).to(args.device)
    #  from model.dinov3_2dcnn import Slice3DTo2DClassifier
    # model = Slice3DTo2DClassifier(
    #     num_classes=3, backbone='dinov3', pretrained="/home/apulis-dev/userdata/Model/dinov3-vitb16-pretrain-lvd1689m", 
    #     feature_dim=768, freeze_backbone=True, use_depth_conv=True
    # ).to(args.device)

    # from model.dinov3_cnn import Slice3DTo2DClassifier
    # model = Slice3DTo2DClassifier(
    #     num_classes=3, 
    #     backbone2d='dinov3', 
    #     pretrained2d="/home/apulis-dev/userdata/Model/dinov3-vitb16-pretrain-lvd1689m", 
    #     backbone3d='densenet264', 
    #     pretrained3d="/home/apulis-dev/code/lung_yin/work_dir/densenet256_BT_NM_MT_128_256size_fixNM/model_acc_best.pth", 
    #     feature_dim=768, 
    #     freeze_backbone=True, 
    #     use_depth_conv=False,
    # ).to(args.device)
    # args.checkpoint = "" #防止重复加载
    return model

class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.args = args    
        self.best_loss = np.inf
        self.best_acc = 0.0
        self.step_best_loss = np.inf
        self.step_best_acc = 0.0
        self.losses = []
        self.accs = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        self.train_dataloaders, self.val_dataloaders = dataloaders
        print(f'self.train_dataloaders:{len(self.train_dataloaders)}')
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_out_dir, "runs"))
        # for (ct_batch, text_batch, gene_batch, modal_mask), _ in self.val_dataloaders:
        #     break
        # ct_batch = ct_batch.to(self.args.device)
        # text_batch = text_batch.to(self.args.device)
        # gene_batch = gene_batch.to(self.args.device)
        # modal_mask = modal_mask.to(self.args.device)
        # self.writer.add_graph(self.model, (ct_batch, text_batch, gene_batch, modal_mask),)

        if (args.resume):
            self.init_checkpoint(
                os.path.join(self.args.work_dir, self.args.task_name, 'model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

    def set_loss_fn(self):
        if self.args.loss_type == "BCELoss":
            self.loss = torch.nn.BCELoss()
        elif self.args.loss_type == "BCEWithLogitsLoss":
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.args.loss_type == "CrossEntropyLoss":
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss type wrong")
    
    def set_optimizer(self):
        if self.args.weight_decay == 0:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        elif self.args.lr_scheduler == 'exponential':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if (self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.accs = last_ckpt['accs']
                self.best_loss = last_ckpt['best_loss']
                self.best_acc = last_ckpt['best_acc']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "accs": self.accs,
                "best_loss": self.best_loss,
                "best_acc": self.best_acc,
                "args": self.args,
            }, os.path.join(args.model_save_path, f"model_{describe}.pth"))

    def interaction(self, outputs, labels):
        return_loss = self.loss(outputs, labels)
        return return_loss

    def get_acc_score(self, outputs, labels):
        # if labels.shape[1] > 1: #multi classifa mode
        #     value = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
        #     accurate = value.sum().item() / len(value)
        # else:
        #     a = np.squeeze(outputs.cpu().detach().numpy(), -1)
        #     a = a < 0.5
        #     b = np.squeeze(labels.cpu().detach().numpy(), -1) == 0
        #     c = np.sum(a == b)
        #     accurate = c / len(a)
        # return accurate
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels)

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()

        if not self.args.multi_gpu:
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.train_dataloaders)
        else:
            tbar = self.train_dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_acc = 0
        for step, ((ct_batch, text_batch, gene_batch, modal_mask), labels) in enumerate(tbar):
            # import pdb; pdb.set_trace()
            # my_context = self.model.no_sync if self.args.rank != - \
            #     1 and step % self.args.accumulation_steps != 0 else nullcontext
            # with my_context():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ct_batch = ct_batch.to(self.args.device)
                text_batch = text_batch.to(self.args.device)
                gene_batch = gene_batch.to(self.args.device)
                modal_mask = modal_mask.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = self.model(ct_batch, text_batch, gene_batch, modal_mask)
                # loss = self.interaction(outputs, labels.float())
                loss = self.interaction(outputs, labels)

                epoch_loss += loss.item()
                epoch_acc += self.get_acc_score(outputs, labels)
                cur_loss = loss.item()
                loss /= self.args.accumulation_steps
                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                print_loss = step_loss / self.args.accumulation_steps
                step_loss = cur_loss #这里修复了一个错误，设置为0会跳过一次loss
                print_acc = self.get_acc_score(outputs, labels)
            else:
                step_loss += cur_loss
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    if print_acc > self.step_best_acc:
                        logger.info(f'Best step accuracy. Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Acc: {print_acc}')
                        self.step_best_acc = print_acc
                        if print_acc > 0.6:
                            self.save_checkpoint(epoch,
                                                 self.model.state_dict(),
                                                 describe=f'{epoch}_step_acc:{print_acc}_best')
                    else:
                        print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Acc: {print_acc}')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_acc /= step + 1

        return epoch_loss, epoch_iou, epoch_acc

    def eval_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_acc = 0 #一个一个计算acc的方式
        # epoch_label_list = [] #一起计算acc的方式
        epoch_output_list = []

        self.model.eval()
        tbar = tqdm(self.val_dataloaders)
        with torch.no_grad():
            for step, ((ct_batch, text_batch, gene_batch, modal_mask), labels) in enumerate(tbar):
                ct_batch = ct_batch.to(self.args.device)
                text_batch = text_batch.to(self.args.device)
                gene_batch = gene_batch.to(self.args.device)
                labels = labels.to(self.args.device)
                modal_mask = modal_mask.to(self.args.device)
                outputs = self.model(ct_batch, text_batch, gene_batch, modal_mask)

                # loss = self.interaction(outputs, labels.float())
                loss = self.interaction(outputs, labels)
                epoch_loss += loss.item()
                epoch_acc += self.get_acc_score(outputs, labels) #一个一个计算acc的方式
                # epoch_label_list.append(labels)
                # epoch_output_list.append(outputs)
            epoch_loss /= step + 1
            epoch_acc /= step + 1 #一个一个计算acc的方式
            # epoch_acc = self.get_acc_score(torch.cat(epoch_output_list), torch.cat(epoch_label_list))
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch}, All eval data num: {len(self.val_dataloaders)}, Eval Loss: {epoch_loss}, Eval Acc: {epoch_acc}')
                
        return epoch_loss, epoch_iou, epoch_acc


    # def plot_result(self, plot_data, description, save_name):
    #     plt.plot(plot_data)
    #     plt.title(description)
    #     plt.xlabel('Epoch')
    #     plt.ylabel(f'{save_name}')
    #     plt.savefig(os.path.join(self.args.model_save_path, f'{save_name}.png'))
    #     plt.close()

    def train(self):
        self.scaler = torch.amp.GradScaler(self.args.device)
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.train_dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_acc= self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.writer.add_scalar('train/Loss', epoch_loss, epoch)
                self.writer.add_scalar('train/Accuracy', epoch_acc, epoch)
                self.losses.append(epoch_loss)
                self.accs.append(epoch_acc)
                print(f'EPOCH: {epoch}, Train Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Train Accuracy: {epoch_acc}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, acc: {epoch_acc}')

                # if self.args.multi_gpu:
                #     state_dict = self.model.module.state_dict()
                # else:
                #     state_dict = self.model.state_dict()
                state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                epoch_loss, epoch_iou, epoch_acc= self.eval_epoch(epoch, num_clicks)
                self.writer.add_scalar('eval/Loss', epoch_loss, epoch)
                self.writer.add_scalar('eval/Accuracy', epoch_acc, epoch)

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')

                # save train acc best checkpoint
                if epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.save_checkpoint(epoch, state_dict, describe='acc_best')

                # self.plot_result(self.losses, 'Accuracy + Cross Entropy Loss', 'Loss')
                # self.plot_result(self.accs, 'Acc', 'Acc')
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.writer.close()
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best acc: {self.best_acc}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total acc: {self.accs}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def save_config(args): #设置保存路径的config
    args.log_out_dir = os.path.join(args.work_dir, args.task_name)
    args.model_save_path = os.path.join(args.work_dir, args.task_name)
    os.makedirs(args.model_save_path, exist_ok=True)


def device_config(args):
    #Use CPU, GPU or NPU, 如果可用进行调用，不可用设置为CPU
    if args.device == 'cuda':
        if  torch.cuda.is_available():
            logging.info("CUDA is available")
            logging.info("Use GPU" + str(args.gpu_ids))
            # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
        else:
            logging.warning("CUDA is not available, use CPU as device")
            args.device = 'cpu'
    elif args.device == 'npu':
        try:
            import torch_npu
            assert torch_npu.npu.is_available()
            logging.info("NPU is available")
        except:
            logging.warning("Warning!! NPU is not available, use CPU as device")
            args.device = 'cpu'
        
    try: #多卡设置
        if not args.multi_gpu:
            # Single GPU
            if args.device != 'mps':
                args.device = f"{args.device}:{args.gpu_ids[0]}" #兼容GPU和NPU
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main_worker(rank, args):
    setup(rank, args.world_size, args)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN,
                        filemode='w',
                        filename=os.path.join(args.log_out_dir, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size, args):
    if 'cuda' in args.device:
        # initialize the process group
        dist.init_process_group(backend='nccl',
                                init_method=f'tcp://127.0.0.1:{args.port}',
                                world_size=world_size,
                                rank=rank)
    elif 'npu' in args.device:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.port #用户自行设置
        dist.init_process_group(backend='hccl',
                                world_size=world_size,
                                rank=rank)



def cleanup():
    dist.destroy_process_group()


def main(args):#主函数，在这里设置单机多卡和多级多卡的训练
    save_config(args)
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.basicConfig(format='[%(asctime)s] - %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.INFO,
                            filemode='w',
                            filename=os.path.join(args.log_out_dir, f'output_{cur_time}.log'))
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)