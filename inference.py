from argparse import ArgumentParser
import sys
import os
import cv2 as cv
import numpy as np
import glob
import tqdm
from mmengine.config import Config

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchmetrics import MetricCollection, JaccardIndex, F1Score, ClasswiseWrapper
from lightning.pytorch import seed_everything
from get_elements import get_metrics, get_model

parser = ArgumentParser()
parser.add_argument("--config", default='configs.eval', type=str, help="config file path (default: None)")
parser.add_argument('--devices', type=lambda s: [int(item) for item in s.split(',')], default=[0])
parser.add_argument('--project', type=str, default="mFoV")
parser.add_argument('--name', type=str, default="test_sam_prompt")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--save_postfix', type=str, default='segmented_medsam')
args = parser.parse_args()

module = __import__(args.config, globals(), locals(), ['cfg'])
cfg = module.cfg
#cfg = Config.fromfile('../config/BCSS.py')

cfg["project"] = args.project
cfg["devices"] = args.devices
cfg["name"] = args.name
cfg["seed"] = args.seed
seed_everything(cfg["seed"])

print('-----------------------------------------------------------------------------------')
print(cfg)
print('-----------------------------------------------------------------------------------')
# main(cfg)

metrics_calculator = get_metrics(cfg=cfg)

sam_model = get_model(cfg)

ckpt = torch.load(
    './ckpts/model.ckpt', map_location='cpu'
)

updated_state_dict = {k[6:]: v for k, v in ckpt['state_dict'].items() if k[6:] in sam_model.state_dict()}

sam_model.load_state_dict(updated_state_dict)
sam_model.eval()


class ImageMaskDataset(Dataset):
    def __init__(self, root_dir):
        dataset = 'BCSS'
        mode = 'test'
        #with open(f'../datasets/{dataset}/{mode}_files.txt', 'r') as f:
        #    self.img_paths = f.read().splitlines()

        self.dataset = dataset
        #self.transform = A.Compose(
        #    [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform]
        #    + [ToTensorV2()], p=1)
        mean = cfg.dataset.dataset_mean
        std = cfg.dataset.dataset_std

        transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),  
            ToTensorV2() 
        ])
        self.transform = transform
        self.img_paths = glob.glob(f"{root_dir}/**/*.png", recursive=True)
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        index = index % len(self)
        # img_path = '../' + self.img_paths[index]
        img_path = self.img_paths[index]

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


        ret = self.transform(image=image)
        return img_path, ret["image"]

if __name__ == '__main__':

    test_dataset = ImageMaskDataset(root_dir=args.root_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False
    )

    device = 'cuda:1'
    metrics_calculator = metrics_calculator.to(device)

    ignore_index = 0
    num_classes = 6
    epoch_iterator = tqdm.tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                            dynamic_ncols=True)
    epoch = 0
    sam_model.to(device)

    for data_iter_step, (img_paths, images) in enumerate(epoch_iterator):
        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))
        images = images.float()
        images = images.to(device)
        
        pred_masks = sam_model(images)[0]
        pred_masks = torch.stack(pred_masks, dim=0)
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        
        for img_path, pred_mask in zip(img_paths, pred_masks):
            pred_mask = pred_mask.cpu()
            pred_mask = pred_mask.to(torch.uint8)
            pred_mask = torchvision.transforms.functional.to_pil_image(pred_mask)
            save_path = img_path.replace(args.root_dir, f"{args.root_dir}_{args.save_postfix}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pred_mask.save(save_path)