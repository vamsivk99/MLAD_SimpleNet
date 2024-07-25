import os
import json
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from data import DATA
from util.data import get_img_loader

class MVTecDataset(Dataset):
    def __init__(self, root, meta_file, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.loader = get_img_loader("pil")
        self.loader_target = get_img_loader("pil")

        with open(os.path.join(root, meta_file), 'r') as f:
            meta_info = json.load(f)

        if "mvtec" in root or "visa" in root:
            meta_info = meta_info['train' if train else 'test']
            self.cls_names = list(meta_info.keys())
        elif "mvtec_loco" in root:
            if train:
                meta_info, meta_info_val = meta_info['train'], meta_info['validation']
                for k in meta_info.keys():
                    meta_info[k].extend(meta_info_val[k])
            else:
                meta_info = meta_info['test']
            self.cls_names = list(meta_info.keys())

        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])

        if self.train:
            random.shuffle(self.data_all)

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, _, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
        img_path = os.path.join(self.root, img_path)
        img = self.loader(img_path)

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            img_mask = np.array(self.loader_target(os.path.join(self.root, mask_path))) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        if self.transform:
            img = self.transform(img)
        if self.target_transform and img_mask is not None:
            img_mask = self.target_transform(img_mask)
        img_mask = [] if img_mask is None else img_mask

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}

def get_transforms(size, isize):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

if __name__ == '__main__':
    from argparse import Namespace as _Namespace

    cfg = _Namespace()
    data = _Namespace()
    
    # Define dataset-specific configurations
    # Uncomment the appropriate dataset configuration
    
    # For MVTec dataset
    # data.root = 'path_to_mvtec_dataset'
    # data.meta = 'meta.json'
    # data.cls_names = []

    # For MVTec Loco dataset
    # data.root = 'path_to_mvtec_loco_dataset'
    # data.meta = 'meta.json'
    # data.cls_names = []

    # For VISA dataset
    data.root = 'path_to_visa_dataset'
    data.meta = 'meta.json'
    data.cls_names = []

    data.loader_type = 'pil'
    data.loader_type_target = 'pil'
    cfg.data = data

    size = 256  # Example size, adjust as necessary
    isize = 256  # Example isize, adjust as necessary

    train_transforms, target_transforms = get_transforms(size, isize)
    dataset = MVTecDataset(root=cfg.data.root, meta_file=cfg.data.meta, train=True, transform=train_transforms, target_transform=target_transforms)

    for idx, data in enumerate(dataset):
        print(data)
        if idx > 10:  # Print first 10 samples
            break
