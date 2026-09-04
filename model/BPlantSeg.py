import torch
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class MultiModalPlantSeg2D_RGB(Dataset):
    """双模态植物分割数据集加载器 - RGB + NIR"""

    def __init__(self, root_dir, mode='train', img_size=(128, 128), num_classes=3):
        super(MultiModalPlantSeg2D_RGB, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.num_classes = num_classes  # ✅ 新增：明确类别数

        self.rgb_dir = os.path.join(root_dir, 'RGB', mode)
        self.nir_dir = os.path.join(root_dir, 'NIR', mode)
        self.ann_dir = os.path.join(root_dir, 'Seg', mode)

        self.images = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])

        self._validate_dataset()

        self.rgb_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.gray_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # ✅ 修复：使用 PIL.Image.NEAREST 替代
        self.label_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
        ])

    def _validate_dataset(self):
        """验证两个模态是否都有相应的文件"""
        for img_name in self.images:
            nir_path = os.path.join(self.nir_dir, img_name)
            label_path = os.path.join(self.ann_dir, img_name)

            if not os.path.exists(nir_path):
                raise FileNotFoundError(f"NIR image not found: {nir_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label not found: {label_path}")

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = img_name.replace('.png', '')

        # 加载RGB图像
        rgb_path = os.path.join(self.rgb_dir, img_name)
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.rgb_transform(rgb_image)  # [3, H, W]

        # 加载NIR图像
        nir_path = os.path.join(self.nir_dir, img_name)
        nir_image = Image.open(nir_path).convert('L')
        nir_tensor = self.gray_transform(nir_image)  # [1, H, W]

        # ✅ 修复：正确处理标签
        label_path = os.path.join(self.ann_dir, img_name)
        label = Image.open(label_path).convert('L')
        label = self.label_transform(label)
        label = np.array(label, dtype=np.int64)

        # ✅ 修复：移除二值化操作，保留原始标签值 [0, 1, 2]
        # label = (label > 0).astype(np.int64)  # ❌ 删除这行！

        # ✅ 新增：验证标签值范围
        unique_labels = np.unique(label)
        if unique_labels.max() >= self.num_classes:
            print(f"Warning: {img_name} has invalid labels {unique_labels}, max should be {self.num_classes - 1}")
            label = np.clip(label, 0, self.num_classes - 1)

        label = torch.from_numpy(label).long()

        return {
            'patient_id': base_name,
            'rgb': rgb_tensor,  # [3, H, W]
            'nir': nir_tensor,  # [1, H, W]
            'label': label,  # [H, W]，值为 0/1/2
        }

    def __len__(self):
        return len(self.images)


def get_multimodal_datasets(dataset_folder, mode, img_size=(128, 128), num_classes=3):
    """
    获取双模态数据集（RGB + NIR）

    Args:
        num_classes: 类别数，默认3（背景、sugar beet、weed）
    """
    assert mode in ["train", "train_val", "test"]
    if mode == "train_val":
        mode = "val"

    return MultiModalPlantSeg2D_RGB(dataset_folder, mode=mode, img_size=img_size, num_classes=num_classes)