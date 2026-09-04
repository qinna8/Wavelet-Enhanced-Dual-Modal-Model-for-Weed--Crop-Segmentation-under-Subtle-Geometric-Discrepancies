import os
import argparse
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BPlantSeg import get_multimodal_datasets
from Bmodel3DiWT2319_decoder import MultiModalCKD2D
from utils import mkdir, save_best_model, AverageMeter, save_checkpoint
from torch.backends import cudnn
import matplotlib

matplotlib.use('Agg')
import cv2
from datetime import datetime
import csv

parser = argparse.ArgumentParser(description='rdn')

parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--dataset-folder', default="../datasetSelectOBO", type=str)

parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--end-epoch', default=100, type=int)

parser.add_argument('--lr', default=0.0008, type=float)
parser.add_argument('--devices', default=0, type=int)

parser.add_argument('--seed', default=4, type=int)
parser.add_argument('--val', default=1, type=int)
parser.add_argument('--num-classes', default=3, type=int)

parser.add_argument('--exp-name', default="WT2i256OBO319", type=str)  # 0.8927
# parser.add_argument('--exp-name', default="WT2i256OBO319200", type=str)  #
parser.add_argument('--img-size', default=256, type=int)
parser.add_argument('--embed-dim', default=64, type=int)
parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 4])
parser.add_argument('--num-heads', nargs='+', type=int, default=[2, 4, 8])
parser.add_argument('--window-size', default=4, type=int)
parser.add_argument('--batch-size', default=8, type=int)

CLASS_NAMES = ['Background', 'Sugar_Beet', 'Weed']

import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""

    def __init__(self, num_classes=3, smooth=1e-5, weight=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight  # 类别权重

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] 模型输出（logits）
            targets: [B, H, W] 标签
        """
        # 转换为概率
        inputs = F.softmax(inputs, dim=1)

        # 将targets转为one-hot编码
        targets_one_hot = F.one_hot(targets.long(), self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dice_loss = 0.0
        for cls in range(self.num_classes):
            input_cls = inputs[:, cls, :, :]
            target_cls = targets_one_hot[:, cls, :, :]

            intersection = (input_cls * target_cls).sum()
            union = input_cls.sum() + target_cls.sum()

            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            if self.weight is not None:
                dice_loss += (1 - dice) * self.weight[cls]
            else:
                dice_loss += (1 - dice)

        return dice_loss / self.num_classes


# ==================== Focal Loss ====================
class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] 模型输出（logits）
            targets: [B, H, W] 标签
        """
        # 计算交叉熵（不进行reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 计算pt (预测正确类别的概率)
        pt = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== Combined Dice + Focal Loss ====================
class DiceFocalLoss(nn.Module):

    def __init__(self, num_classes=3, dice_weight=0.5, focal_weight=0.5,
                 gamma=2.0, alpha=None, smooth=1e-5, weight=None):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes

        self.dice_loss = DiceLoss(num_classes=num_classes, smooth=smooth, weight=weight)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        total_loss = self.dice_weight * dice + self.focal_weight * focal

        return total_loss

    def get_losses(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return dice, focal


def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def init_folder(args):
    args.base_folder = mkdir(os.path.dirname(os.path.realpath(__file__)))
    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))
    args.best_folder = mkdir(f"{args.base_folder}/best_model/{args.exp_name}")
    args.writer_folder = mkdir(f"{args.base_folder}/writer/{args.exp_name}")
    args.checkpoint_folder = mkdir(f"{args.base_folder}/checkpoint/{args.exp_name}")


def main(args):
    writer = SummaryWriter(args.writer_folder)

    model = MultiModalCKD2D(
        in_channels_rgb=3,
        in_channels_nir=1,
        num_classes=args.num_classes,
        img_size=(args.img_size, args.img_size),
        patch_size=(4, 4),
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=(args.window_size, args.window_size),
        mlp_ratio=4.
    ).cuda()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = DiceFocalLoss(
        num_classes=args.num_classes,
        dice_weight=0.6,  # Dice Loss 权重
        focal_weight=0.4,  # Focal Loss 权重
        gamma=2.5,  # Focal Loss gamma参数
        alpha=torch.tensor([0.01, 0.2, 0.8]).cuda(),  # FocalLoss类别权重
        smooth=1e-5,  # Dice Loss 平滑参数
        weight=torch.tensor([0.01, 0.2, 0.8]).cuda()  # DiceLoss类别权重
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.mode == "train":
        train_dataset = get_multimodal_datasets(
            args.dataset_folder, "train",
            img_size=(args.img_size, args.img_size),
            num_classes=args.num_classes
        )
        val_dataset = get_multimodal_datasets(
            args.dataset_folder, "train_val",
            img_size=(args.img_size, args.img_size),
            num_classes=args.num_classes
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers,
            pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, num_workers=args.workers,
            pin_memory=True
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        train_manager(args, train_loader, val_loader, model, criterion, optimizer, writer)

    elif args.mode == "test":
        print("Starting test...")
        model.load_state_dict(torch.load(os.path.join(args.best_folder, "best_model.pkl")))
        model.eval()
        test_dataset = get_multimodal_datasets(
            args.dataset_folder, "test",
            img_size=(args.img_size, args.img_size),
            num_classes=args.num_classes
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, num_workers=args.workers,
            pin_memory=True
        )
        test(args, test_loader, model, writer)


# ==================== 新增：每个类别的Loss计算函数 ====================
def calculate_per_class_loss(outputs, labels, num_classes=3):
    criterion_none = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    pixel_losses = criterion_none(outputs, labels)  # [B, H, W]

    per_class_losses = []
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.sum() > 0:
            class_loss = pixel_losses[mask].mean().item()
        else:
            class_loss = 0.0
        per_class_losses.append(class_loss)

    return per_class_losses


def calculate_hd95(pred, target, num_classes=3):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 如果是batch，取第一个
    if pred.ndim == 3:
        pred = pred[0]
        target = target[0]

    hd95_per_class = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.uint8)
        target_cls = (target == cls).astype(np.uint8)

        # 如果预测或真实中该类别不存在
        if pred_cls.sum() == 0 and target_cls.sum() == 0:
            hd95_per_class.append(0.0)
            continue
        elif pred_cls.sum() == 0 or target_cls.sum() == 0:
            # 一方为空时，返回最大对角线距离
            max_dist = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
            hd95_per_class.append(max_dist)
            continue

        # 获取边界点
        pred_boundary = get_boundary_points(pred_cls)
        target_boundary = get_boundary_points(target_cls)

        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            hd95_per_class.append(0.0)
            continue

        # 计算双向距离
        distances_pred_to_target = compute_surface_distances(pred_boundary, target_boundary)
        distances_target_to_pred = compute_surface_distances(target_boundary, pred_boundary)

        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        hd95 = np.percentile(all_distances, 95)
        hd95_per_class.append(hd95)

    mean_hd95 = np.mean(hd95_per_class)
    return hd95_per_class, mean_hd95


def get_boundary_points(binary_mask):
    from scipy import ndimage
    # 使用形态学梯度获取边界
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = binary_mask ^ eroded
    points = np.argwhere(boundary)
    return points


def compute_surface_distances(points1, points2):
    from scipy.spatial import cKDTree
    tree = cKDTree(points2)
    distances, _ = tree.query(points1)
    return distances


def calculate_all_per_class_metrics(pred, target, num_classes=3):
    smooth = 1e-6
    metrics = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'hd95': []  # HD95 per class
    }

    # 计算HD95
    hd95_per_class, _ = calculate_hd95(pred, target, num_classes)

    for cls in range(num_classes):
        if isinstance(pred, torch.Tensor):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()

            # True Positives, False Positives, False Negatives, True Negatives
            TP = (pred_cls * target_cls).sum().item()
            FP = (pred_cls * (1 - target_cls)).sum().item()
            FN = ((1 - pred_cls) * target_cls).sum().item()
            TN = ((1 - pred_cls) * (1 - target_cls)).sum().item()
        else:
            pred_cls = (pred == cls).astype(np.float32)
            target_cls = (target == cls).astype(np.float32)

            TP = (pred_cls * target_cls).sum()
            FP = (pred_cls * (1 - target_cls)).sum()
            FN = ((1 - pred_cls) * target_cls).sum()
            TN = ((1 - pred_cls) * (1 - target_cls)).sum()

        # IoU
        intersection = TP
        union = TP + FP + FN
        iou = (intersection + smooth) / (union + smooth)
        metrics['iou'].append(iou)

        # Dice
        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        metrics['dice'].append(dice)

        # Precision
        precision = (TP + smooth) / (TP + FP + smooth)
        metrics['precision'].append(precision)

        # Recall
        recall = (TP + smooth) / (TP + FN + smooth)
        metrics['recall'].append(recall)

        # Accuracy (per class)
        accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
        metrics['accuracy'].append(accuracy)

        # HD95
        metrics['hd95'].append(hd95_per_class[cls])

    return metrics


def train(data_loader, model, criterion, optimizer, epoch, writer, num_classes=3):
    # 总体指标
    losses = AverageMeter('Loss', ':.4e')
    dice_losses = AverageMeter('DiceLoss', ':.4e')
    focal_losses = AverageMeter('FocalLoss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4f')
    ious = AverageMeter('IoU', ':.4f')

    # 每个类别的指标累积器
    per_class_losses_avg = [AverageMeter(f'Loss_{CLASS_NAMES[i]}', ':.4e') for i in range(num_classes)]
    per_class_ious_avg = [AverageMeter(f'IoU_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_dices_avg = [AverageMeter(f'Dice_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_precisions_avg = [AverageMeter(f'Prec_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_recalls_avg = [AverageMeter(f'Recall_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]

    for i, data in enumerate(data_loader):
        rgb = data["rgb"].cuda()
        nir = data["nir"].cuda()
        labels = data["label"].cuda()

        if epoch == 0 and i == 0:
            print(f"\nTraining data info:")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  NIR shape: {nir.shape}")
            print(f"  Label shape: {labels.shape}")
            print(f"  Label unique values: {torch.unique(labels)}")

        outputs = model({'rgb': rgb, 'nir': nir})
        loss = criterion(outputs, labels)

        # 获取单独的dice和focal loss用于监控
        if hasattr(criterion, 'get_losses'):
            with torch.no_grad():
                dice_l, focal_l = criterion.get_losses(outputs, labels)
                dice_losses.update(dice_l.item(), rgb.size(0))
                focal_losses.update(focal_l.item(), rgb.size(0))

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        iou = calculate_iou(preds, labels, num_classes=num_classes)

        # 计算每个类别的Loss (使用CrossEntropy来分析每类损失)
        per_class_losses = calculate_per_class_loss(outputs, labels, num_classes)

        # 计算每个类别的完整指标
        per_class_metrics = calculate_all_per_class_metrics(preds, labels, num_classes)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 更新总体指标
        losses.update(loss.item(), rgb.size(0))
        accuracies.update(acc.item(), rgb.size(0))
        ious.update(iou, rgb.size(0))

        # 更新每个类别的指标
        for cls in range(num_classes):
            per_class_losses_avg[cls].update(per_class_losses[cls], rgb.size(0))
            per_class_ious_avg[cls].update(per_class_metrics['iou'][cls], rgb.size(0))
            per_class_dices_avg[cls].update(per_class_metrics['dice'][cls], rgb.size(0))
            per_class_precisions_avg[cls].update(per_class_metrics['precision'][cls], rgb.size(0))
            per_class_recalls_avg[cls].update(per_class_metrics['recall'][cls], rgb.size(0))

        if (i + 1) % 50 == 0:
            print(f"  Epoch [{epoch + 1}] Batch [{i + 1}/{len(data_loader)}] "
                  f"Loss: {losses.avg:.4f} (Dice: {dice_losses.avg:.4f}, Focal: {focal_losses.avg:.4f}), "
                  f"Acc: {accuracies.avg:.4f}, IoU: {ious.avg:.4f}")

    # TensorBoard记录
    writer.add_scalar("loss/train", losses.avg, epoch)
    writer.add_scalar("loss/train_dice", dice_losses.avg, epoch)  # 新增
    writer.add_scalar("loss/train_focal", focal_losses.avg, epoch)  # 新增
    writer.add_scalar("acc/train", accuracies.avg, epoch)
    writer.add_scalar("iou/train", ious.avg, epoch)

    # 记录每个类别的指标到TensorBoard
    for cls in range(num_classes):
        writer.add_scalar(f"loss_train/{CLASS_NAMES[cls]}", per_class_losses_avg[cls].avg, epoch)
        writer.add_scalar(f"iou_train/{CLASS_NAMES[cls]}", per_class_ious_avg[cls].avg, epoch)
        writer.add_scalar(f"dice_train/{CLASS_NAMES[cls]}", per_class_dices_avg[cls].avg, epoch)

    # 构建返回的指标字典
    train_metrics = {
        'loss': losses.avg,
        'dice_loss': dice_losses.avg,  # 新增
        'focal_loss': focal_losses.avg,  # 新增
        'acc': accuracies.avg,
        'iou': ious.avg,
    }

    for cls in range(num_classes):
        class_name = CLASS_NAMES[cls]
        train_metrics[f'loss_{class_name}'] = per_class_losses_avg[cls].avg
        train_metrics[f'iou_{class_name}'] = per_class_ious_avg[cls].avg
        train_metrics[f'dice_{class_name}'] = per_class_dices_avg[cls].avg
        train_metrics[f'precision_{class_name}'] = per_class_precisions_avg[cls].avg
        train_metrics[f'recall_{class_name}'] = per_class_recalls_avg[cls].avg

    return train_metrics


# ==================== 修改：validate函数 ====================
def validate(data_loader, model, criterion, epoch, writer, num_classes=3):
    # 总体指标
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4f')
    ious = AverageMeter('IoU', ':.4f')
    dices = AverageMeter('Dice', ':.4f')

    # 每个类别的指标累积器
    per_class_losses_avg = [AverageMeter(f'Loss_{CLASS_NAMES[i]}', ':.4e') for i in range(num_classes)]
    per_class_ious_avg = [AverageMeter(f'IoU_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_dices_avg = [AverageMeter(f'Dice_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_precisions_avg = [AverageMeter(f'Prec_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]
    per_class_recalls_avg = [AverageMeter(f'Recall_{CLASS_NAMES[i]}', ':.4f') for i in range(num_classes)]

    for i, data in enumerate(data_loader):
        rgb = data["rgb"].cuda()
        nir = data["nir"].cuda()
        labels = data["label"].cuda()

        with torch.no_grad():
            outputs = model({'rgb': rgb, 'nir': nir})
            loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        iou = calculate_iou(preds, labels, num_classes=num_classes)
        dice = calculate_dice(preds, labels, num_classes=num_classes)

        # 计算每个类别的Loss
        per_class_losses = calculate_per_class_loss(outputs, labels, num_classes)

        # 计算每个类别的完整指标
        per_class_metrics = calculate_all_per_class_metrics(preds, labels, num_classes)

        # 更新总体指标
        losses.update(loss.item(), rgb.size(0))
        accuracies.update(acc.item(), rgb.size(0))
        ious.update(iou, rgb.size(0))
        dices.update(dice, rgb.size(0))

        # 更新每个类别的指标
        for cls in range(num_classes):
            per_class_losses_avg[cls].update(per_class_losses[cls], rgb.size(0))
            per_class_ious_avg[cls].update(per_class_metrics['iou'][cls], rgb.size(0))
            per_class_dices_avg[cls].update(per_class_metrics['dice'][cls], rgb.size(0))
            per_class_precisions_avg[cls].update(per_class_metrics['precision'][cls], rgb.size(0))
            per_class_recalls_avg[cls].update(per_class_metrics['recall'][cls], rgb.size(0))

    # 打印每类指标
    print(f"\n  Per-class Validation Metrics:")
    print(f"  {'Class':<15} {'Loss':<10} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10}")
    print(f"  {'-' * 65}")
    for cls in range(num_classes):
        print(f"  {CLASS_NAMES[cls]:<15} "
              f"{per_class_losses_avg[cls].avg:<10.4f} "
              f"{per_class_ious_avg[cls].avg:<10.4f} "
              f"{per_class_dices_avg[cls].avg:<10.4f} "
              f"{per_class_precisions_avg[cls].avg:<10.4f} "
              f"{per_class_recalls_avg[cls].avg:<10.4f}")

    # TensorBoard记录
    writer.add_scalar("loss/val", losses.avg, epoch)
    writer.add_scalar("acc/val", accuracies.avg, epoch)
    writer.add_scalar("iou/val", ious.avg, epoch)
    writer.add_scalar("dice/val", dices.avg, epoch)

    for cls in range(num_classes):
        writer.add_scalar(f"loss_val/{CLASS_NAMES[cls]}", per_class_losses_avg[cls].avg, epoch)
        writer.add_scalar(f"iou_val/{CLASS_NAMES[cls]}", per_class_ious_avg[cls].avg, epoch)
        writer.add_scalar(f"dice_val/{CLASS_NAMES[cls]}", per_class_dices_avg[cls].avg, epoch)

    # 构建返回的指标字典
    val_metrics = {
        'loss': losses.avg,
        'acc': accuracies.avg,
        'iou': ious.avg,
        'dice': dices.avg,
    }

    for cls in range(num_classes):
        class_name = CLASS_NAMES[cls]
        val_metrics[f'loss_{class_name}'] = per_class_losses_avg[cls].avg
        val_metrics[f'iou_{class_name}'] = per_class_ious_avg[cls].avg
        val_metrics[f'dice_{class_name}'] = per_class_dices_avg[cls].avg
        val_metrics[f'precision_{class_name}'] = per_class_precisions_avg[cls].avg
        val_metrics[f'recall_{class_name}'] = per_class_recalls_avg[cls].avg

    return val_metrics


# ==================== 修改：test函数 ====================
def test(args, data_loader, model, writer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = mkdir(f"{args.base_folder}/test_results/{args.exp_name}_{timestamp}")
    pred_dir = mkdir(os.path.join(save_dir, "predictions"))
    overlay_dir = mkdir(os.path.join(save_dir, "overlays"))
    comparison_dir = mkdir(os.path.join(save_dir, "comparisons"))
    modality_dir = mkdir(os.path.join(save_dir, "modalities"))

    # 总体指标列表
    all_results = []

    # 每个类别的指标累积 - 添加accuracy
    per_class_metrics_all = {
        cls: {
            'iou': [], 'dice': [], 'precision': [], 'recall': [],
            'loss': [], 'hd95': [], 'accuracy': []
        }
        for cls in range(args.num_classes)
    }

    # 总体指标累积
    all_hd95 = []
    all_precision = []
    all_recall = []

    print(f"Saving results to: {save_dir}")

    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i, data in enumerate(data_loader):
        patient_id = data["patient_id"][0]
        rgb = data["rgb"].cuda()
        nir = data["nir"].cuda()
        labels = data["label"].cuda()

        with torch.no_grad():
            outputs = model({'rgb': rgb, 'nir': nir})
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            acc = (preds == labels).float().mean().item()
            iou = calculate_iou(preds, labels, num_classes=args.num_classes)
            dice = calculate_dice(preds, labels, num_classes=args.num_classes)

            # 计算每个类别的Loss
            per_class_losses = calculate_per_class_loss(outputs, labels, args.num_classes)

            # 计算每个类别的完整指标（包含新增的HD95）
            per_class_metrics = calculate_all_per_class_metrics(preds, labels, args.num_classes)

            # 计算总体HD95
            hd95_per_class, mean_hd95 = calculate_hd95(preds, labels, args.num_classes)
            all_hd95.append(mean_hd95)

            # 计算总体Precision、Recall（各类别平均）
            mean_precision = np.mean(per_class_metrics['precision'])
            mean_recall = np.mean(per_class_metrics['recall'])
            all_precision.append(mean_precision)
            all_recall.append(mean_recall)

            # 保存可视化结果
            save_multimodal_predictions_rgb(
                rgb, nir, labels, preds, patient_id,
                pred_dir, overlay_dir, comparison_dir, modality_dir, args
            )

            # 构建单个样本的结果字典 - 添加新指标
            sample_result = {
                'patient_id': patient_id,
                'accuracy': acc,
                'iou_mean': iou,
                'dice_mean': dice,
                'precision_mean': mean_precision,
                'recall_mean': mean_recall,
                'hd95_mean': mean_hd95,
                'loss': loss.item(),
            }

            # 添加每个类别的指标
            for cls in range(args.num_classes):
                class_name = CLASS_NAMES[cls]
                sample_result[f'loss_{class_name}'] = per_class_losses[cls]
                sample_result[f'iou_{class_name}'] = per_class_metrics['iou'][cls]
                sample_result[f'dice_{class_name}'] = per_class_metrics['dice'][cls]
                sample_result[f'precision_{class_name}'] = per_class_metrics['precision'][cls]
                sample_result[f'recall_{class_name}'] = per_class_metrics['recall'][cls]
                sample_result[f'accuracy_{class_name}'] = per_class_metrics['accuracy'][cls]
                sample_result[f'hd95_{class_name}'] = per_class_metrics['hd95'][cls]

                # 累积每类指标
                per_class_metrics_all[cls]['loss'].append(per_class_losses[cls])
                per_class_metrics_all[cls]['iou'].append(per_class_metrics['iou'][cls])
                per_class_metrics_all[cls]['dice'].append(per_class_metrics['dice'][cls])
                per_class_metrics_all[cls]['precision'].append(per_class_metrics['precision'][cls])
                per_class_metrics_all[cls]['recall'].append(per_class_metrics['recall'][cls])
                per_class_metrics_all[cls]['accuracy'].append(per_class_metrics['accuracy'][cls])
                per_class_metrics_all[cls]['hd95'].append(per_class_metrics['hd95'][cls])

            all_results.append(sample_result)

            print(f"[{i + 1}/{len(data_loader)}] Sample {patient_id}: "
                  f"Acc={acc:.4f}, IoU={iou:.4f}, Dice={dice:.4f}, "
                  f"Prec={mean_precision:.4f}, Rec={mean_recall:.4f}, "
                  f"HD95={mean_hd95:.2f}")

    # 打印汇总结果
    print(f"\n{'=' * 120}")
    print("Test Results Summary")
    print(f"{'=' * 120}")
    print(f"\nOverall Metrics:")
    print(
        f"  Mean Accuracy:  {np.mean([r['accuracy'] for r in all_results]):.4f} ± {np.std([r['accuracy'] for r in all_results]):.4f}")
    print(
        f"  Mean IoU:       {np.mean([r['iou_mean'] for r in all_results]):.4f} ± {np.std([r['iou_mean'] for r in all_results]):.4f}")
    print(
        f"  Mean Dice:      {np.mean([r['dice_mean'] for r in all_results]):.4f} ± {np.std([r['dice_mean'] for r in all_results]):.4f}")
    print(f"  Mean Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"  Mean Recall:    {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"  Mean HD95:      {np.mean(all_hd95):.2f} ± {np.std(all_hd95):.2f}")

    print(f"\nPer-Class Metrics:")
    header = f"  {'Class':<12} {'Acc':<10} {'IoU':<10} {'Dice':<10} {'Prec':<10} {'Recall':<10} {'HD95':<10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for cls in range(args.num_classes):
        m = per_class_metrics_all[cls]
        print(f"  {CLASS_NAMES[cls]:<12} "
              f"{np.mean(m['accuracy']):.4f}    "
              f"{np.mean(m['iou']):.4f}    "
              f"{np.mean(m['dice']):.4f}    "
              f"{np.mean(m['precision']):.4f}    "
              f"{np.mean(m['recall']):.4f}    "
              f"{np.mean(m['hd95']):.2f}")

    # 打印标准差表格
    print(f"\nPer-Class Metrics (Std):")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for cls in range(args.num_classes):
        m = per_class_metrics_all[cls]
        print(f"  {CLASS_NAMES[cls]:<12} "
              f"{np.std(m['accuracy']):.4f}    "
              f"{np.std(m['iou']):.4f}    "
              f"{np.std(m['dice']):.4f}    "
              f"{np.std(m['precision']):.4f}    "
              f"{np.std(m['recall']):.4f}    "
              f"{np.std(m['hd95']):.2f}")


# ==================== 修改：train_manager函数 ====================
def train_manager(args, train_loader, val_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    best_iou = 0.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, eta_min=1e-6
    )

    for epoch in range(args.end_epoch):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("lr", current_lr, epoch)

        # 训练并获取训练指标
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, writer, num_classes=args.num_classes)
        scheduler.step()

        # 初始化验证指标
        val_metrics = {
            'val_loss': np.nan,
            'val_acc': np.nan,
            'val_iou': np.nan,
            'val_dice': np.nan,
        }

        if (epoch + 1) % args.val == 0:
            model.eval()
            with torch.no_grad():
                val_metrics_result = validate(val_loader, model, criterion, epoch, writer, num_classes=args.num_classes)

                # 重新映射验证指标
                val_metrics = {
                    'val_loss': val_metrics_result['loss'],
                    'val_acc': val_metrics_result['acc'],
                    'val_iou': val_metrics_result['iou'],
                    'val_dice': val_metrics_result['dice'],
                }

                # 基于IoU保存最佳模型
                if val_metrics['val_iou'] > best_iou:
                    best_iou = val_metrics['val_iou']
                    best_loss = val_metrics['val_loss']
                    save_best_model(args, model)
                    print(f"  >>> New best model saved! IoU: {val_metrics['val_iou']:.4f}")

        # 保存检查点
        save_checkpoint(args, {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_iou': best_iou,
            'best_loss': best_loss
        })

        # 打印简要信息
        print(f"Epoch [{epoch + 1}/{args.end_epoch}]: "
              f"train_loss={train_metrics['loss']:.4f}, train_iou={train_metrics['iou']:.4f}, "
              f"val_loss={val_metrics.get('val_loss', np.nan):.4f}, val_iou={val_metrics.get('val_iou', np.nan):.4f}, "
              f"lr={current_lr:.6f}")


def save_multimodal_predictions_rgb(rgb, nir, labels, preds, patient_id, pred_dir, overlay_dir, comparison_dir,
                                    modality_dir, args):
    """保存双模态预测结果的可视化（RGB + NIR）"""

    rgb_tensor = rgb[0].cpu()
    nir_tensor = nir[0].cpu()[0]
    label = labels[0].cpu().numpy()
    pred = preds[0].cpu().numpy()

    # 反归一化RGB
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    rgb_np = rgb_tensor.numpy()
    rgb_np = rgb_np * std + mean
    rgb_np = np.clip(rgb_np, 0, 1)
    rgb_np = (rgb_np * 255).astype(np.uint8)
    rgb_vis = np.transpose(rgb_np, (1, 2, 0))

    # 反归一化NIR
    nir_np = nir_tensor.numpy()
    nir_np = (nir_np * 0.5 + 0.5)
    nir_np = np.clip(nir_np, 0, 1)
    nir_vis = (nir_np * 255).astype(np.uint8)

    # 保存各个模态
    cv2.imwrite(os.path.join(modality_dir, f"{patient_id}_rgb.png"),
                cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(modality_dir, f"{patient_id}_nir.png"), nir_vis)

    # 保存预测结果
    pred_mask = (pred * 127).astype(np.uint8)  # 0, 127, 254 for 3 classes
    cv2.imwrite(os.path.join(pred_dir, f"{patient_id}_pred.png"), pred_mask)

    pred_colored = colorize_mask(pred, args.num_classes)
    cv2.imwrite(os.path.join(pred_dir, f"{patient_id}_pred_colored.png"),
                cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))

    overlay = create_overlay(rgb_vis, pred, alpha=0.6, num_classes=args.num_classes)
    cv2.imwrite(os.path.join(overlay_dir, f"{patient_id}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    comparison = create_multimodal_comparison_rgb(
        rgb_vis, nir_vis, label, pred, args.num_classes
    )
    cv2.imwrite(os.path.join(comparison_dir, f"{patient_id}_comparison.png"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def calculate_iou(pred, target, num_classes=3):
    """计算IoU (Intersection over Union)"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(1.0 if intersection == 0 else 0.0)
    return np.mean(ious)


def calculate_dice(pred, target, num_classes=3):
    """计算Dice系数"""
    dices = []
    smooth = 1e-5
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dice = (2. * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        dices.append(dice.item())
    return np.mean(dices)


def colorize_mask(mask, num_classes):
    """将分割mask转换为彩色图像"""
    if num_classes == 2:
        palette = np.array([
            [0, 0, 0],
            [0, 255, 0],
        ], dtype=np.uint8)
    elif num_classes == 3:
        palette = np.array([
            [0, 0, 0],  # 类别0：背景 - 黑色
            [0, 255, 0],  # 类别1：甜菜(sugar beet) - 绿色
            [255, 0, 0],  # 类别2：杂草(weed) - 红色
        ], dtype=np.uint8)
    else:
        palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]

    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(min(num_classes, len(palette))):
        colored_mask[mask == class_id] = palette[class_id]

    return colored_mask


# def create_overlay(image, mask, alpha=0.6, num_classes=3):
#     """创建原图和分割结果的叠加图"""
#     if len(image.shape) == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     colored_mask = colorize_mask(mask, num_classes)
#     overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
#     return overlay


def create_overlay(image, mask, alpha=0.5, num_classes=3):
    """创建原图和分割结果的叠加图（只叠加像素值为1和2的部分）"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colored_mask = colorize_mask(mask, num_classes)

    # 创建一个掩码，只包含像素值为1和2的区域
    foreground_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)

    # 扩展掩码到3通道
    foreground_mask_3ch = np.stack([foreground_mask] * 3, axis=-1)

    # 只在前景区域进行叠加
    overlay = image.copy()
    overlay = np.where(
        foreground_mask_3ch > 0,
        cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0),
        image
    )

    return overlay


def create_multimodal_comparison_rgb(rgb, nir, gt_mask, pred_mask, num_classes):
    """创建对比图（RGB + NIR版本）"""
    h, w = rgb.shape[:2]

    nir_3ch = cv2.cvtColor(nir, cv2.COLOR_GRAY2RGB)

    gt_colored = colorize_mask(gt_mask, num_classes)
    pred_colored = colorize_mask(pred_mask, num_classes)
    overlay = create_overlay(rgb, pred_mask, alpha=0.6, num_classes=num_classes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    def add_label(img, text):
        img_labeled = img.copy()
        cv2.putText(img_labeled, text, (5, 20), font, font_scale, (255, 255, 255), thickness)
        return img_labeled

    rgb_labeled = add_label(rgb, 'RGB')
    nir_labeled = add_label(nir_3ch, 'NIR')
    gt_labeled = add_label(gt_colored, 'Ground Truth')
    pred_labeled = add_label(pred_colored, 'Prediction')
    overlay_labeled = add_label(overlay, 'Overlay')

    top_row = np.hstack([rgb_labeled, nir_labeled, gt_labeled])
    blank = np.zeros_like(rgb_labeled)
    bottom_row = np.hstack([pred_labeled, overlay_labeled, blank])
    comparison = np.vstack([top_row, bottom_row])

    return comparison


def save_test_statistics(save_dir, results_list, accuracies, ious, dices):
    """保存测试统计信息"""
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'accuracy', 'iou', 'dice'])
        writer.writeheader()
        writer.writerows(results_list)

    stats_path = os.path.join(save_dir, 'statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Test Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(results_list)}\n\n")
        f.write(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
        f.write(f"Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")


if __name__ == '__main__':
    args = parser.parse_args()
    print("\n" + "=" * 50)
    print("Multi-Modal Plant Segmentation with RGB + NIR (Dual Modality)")
    print("=" * 50)
    print("\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50 + "\n")

    init_random(args.seed)
    init_folder(args)
    torch.cuda.set_device(args.devices)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(args.devices)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.devices).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("WARNING: CUDA not available, using CPU")

    main(args)