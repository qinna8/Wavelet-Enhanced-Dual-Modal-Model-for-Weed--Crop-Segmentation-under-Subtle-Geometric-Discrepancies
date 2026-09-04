import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from get_dataset_folder import get_brats_folder
import pprint
from medpy.metric import binary


def mkdir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def save_best_model(args, model, filename="best_model.pkl"):
    path = os.path.join(args.best_folder, filename)
    torch.save(model.state_dict(), path)
    print(f"Best model saved to {path}")

def save_checkpoint(args, state, name="checkpont"):
    torch.save(state, f"{args.checkpoint_folder}/{name}.pth.tar")

def save_seg_csv(args, mode, csv):
    try:
        val_metrics = pd.DataFrame.from_records(csv)
        columns = ['id', 'et_dice', 'tc_dice', 'wt_dice', 'et_hd', 'tc_hd', 'wt_hd', 'et_sens', 'tc_sens', 'wt_sens', 'et_spec', 'tc_spec', 'wt_spec']
        val_metrics.to_csv(f'{str(args.csv_folder)}/metrics.csv', index=False, columns=columns)
    except KeyboardInterrupt:
        print("Save CSV File Error!")

def load_nii(path):
    nii_file = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return nii_file

def listdir(path):
    files_list = os.listdir(path)
    files_list.sort()
    return files_list
    
def save_test_label(args, mode, patient_id, predict):

    save_dir = mkdir(os.path.join(args.pred_folder, mode))
    save_path = os.path.join(save_dir, f"{patient_id}.npy")
    np.save(save_path, predict.astype(np.uint8))
    print(f"Saved prediction for {patient_id} to {save_path}")

class AverageMeter(object):
    def __init__(self, name, fmt):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.count += n
            self.sum += val * n
            self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_crop_slice(target_size,dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return (left, dim - right)
    else:
        return (0, dim)

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    else:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right
        
def pad_image_and_label(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    pad_todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    pad_list = [0, 0]
    for to_pad in pad_todos:
        if to_pad[0]:
            pad_list.insert(0, to_pad[1])
            pad_list.insert(0, to_pad[2])
        else:
            pad_list.insert(0, 0)
            pad_list.insert(0, 0)
    if np.sum(pad_list) != 0:
        image = F.pad(image, pad_list, 'constant')
    if seg is not None:
        if np.sum(pad_list) != 0:
            seg = F.pad(seg, pad_list,'constant')
        return image, seg, pad_list
    return image, seg, pad_list

def pad_or_crop_image(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    crop_list = [z_slice, y_slice, x_slice]
    image = image[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    if seg is not None:
        seg = seg[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    image, seg, pad_list = pad_image_and_label(image, seg)
    return image, seg, pad_list, crop_list

def normalize(image):
    min_ = torch.min(image)
    max_ = torch.max(image)
    scale_ = max_ - min_
    image = (image - min_) / scale_
    return image

def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image>0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    image = normalize(image)
    return image
    
def cal_confuse(preds, targets, patient):
    assert preds.shape == targets.shape
    results = []
    total_pixels = targets[0].numel()
    
    for i in range(3):
        p = preds[i].bool()
        t = targets[i].bool()

        tp = (p & t).sum().item()
        tn = (~p & ~t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()
        

        sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        
        results.append([sens, spec])
    return results


def cal_dice(preds, targets, patient, tta=False):
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"
    labels = ["ET", "TC", "WT"]
    metrics_list = []

    for i, label in enumerate(labels):
        metrics = {"patient_id": patient, "label": label, "tta": tta}


        if torch.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            dice = 1 if torch.sum(preds[i]) == 0 else 0
        else:

            tp = torch.sum(torch.logical_and(preds[i], targets[i])).item()
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i]))).item()
            fn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), targets[i])).item()
            dice = 2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) != 0 else 0
        

        if torch.sum(preds[i]) > 0 and torch.sum(targets[i]) > 0:

            hd95 = binary.hd95(
                preds[i].cpu().numpy().astype(bool), 
                targets[i].cpu().numpy().astype(bool)
            )
        else:
            hd95 = 0
        
        metrics["DICE"] = dice
        metrics["HAUSSDORF"] = hd95
        metrics_list.append(metrics)
    
    return metrics_list
