# import torch
# import numpy as np
# import os
from functools import partial
from timm.data.loader import _worker_init
# from timm.data.distributed_sampler import OrderedDistributedSampler
# from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
# from timm.loss import BinaryCrossEntropy
# from ignite.distributed import DistributedProxySampler
from torch.utils.data import BatchSampler
from losses.cr_loss import CrLoss
# import sys

try:
    from datasets.transforms import *
except:
    from .transforms import *

import cv2
import os
import glob


class AbdomenCTDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.size = args.img_size

        self.mode = args.mode
        self.is_training = is_training
        self.patch_num = args.patch_num
        self.cut_crop_size = args.cut_crop_size

        # phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase',
        #               'C-pre', 'C+A', 'C+V', 'C+Delay']
        # if self.args.return_glb:
        #     glb_img_list = []
        #     phase_list_glb = ['T2WI_glb', 'DWI_glb', 'In Phase_glb', 'Out Phase_glb',
        #                       'C-pre_glb', 'C+A_glb', 'C+V_glb', 'C+Delay_glb']

        if is_training:
            self.data_dir = args.data_dir
            self.seg_dir = args.seg_dir
        else:
            self.data_dir = args.pred_data_dir
            self.seg_dir = args.pred_seg_dir
        # 读取数据图片路径
        img_name_list = os.listdir(self.data_dir)
        seg_name_list = os.listdir(self.seg_dir)
        # 读取标签csv文件
        label_csv = args.csvfile
        lab_list = np.loadtxt(open(label_csv, "rb"), delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))

        self.img_list = img_name_list
        self.lab_list = lab_list
        self.seg_list = seg_name_list

    def get_labels(self, ):
        return self.lab_list

    def __getitem__(self, index):
        args = self.args
        label = self.lab_list[index]
        image = self.load_nii_image(os.path.join(self.data_dir,self.img_list[index]))
        image = self.transforms(image, args.train_transform_list)
        seg = self.load_nii_image(os.path.join(self.seg_dir, self.seg_list[index]))
        seg = self.transforms(seg, args.train_transform_list)
        return (image, seg, label)

    def mixup(self, image, label):
        # 类内mixup
        alpha = 1.0
        all_target_ind = [i for i, x in enumerate(self.lab_list) if int(x) == label]
        index = random.choice(all_target_ind)
        image_bal = self.load_mp_images(self.img_list[index])
        lam = np.random.beta(alpha, alpha)
        image = lam * image + (1 - lam) * image_bal

        return image

    def load_mp_images(self, mp_img_list):

        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            # image = self.get_z_roi(image,16)
            image = resize3D(image, self.size, self.mode)
            image = image_normalization(image)
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image
    def load_nii_image(self, img_path):
        image = load_nii_file(img_path)
        image = cut_3Dimage(image, self.patch_num)
        image = resizePatch3D(image, self.cut_crop_size, self.mode)
        image = image_normalization(image)

        return image

    def get_z_roi(self, image, z_num):
        Z, H, W = image.shape

        if Z > z_num:
            Z_mid = int(Z / 2)
            image_mid = image[Z_mid - int(z_num / 2):Z_mid + int(z_num / 2), :]
            return image_mid
        else:
            return image

    def transforms(self, mp_image, transform_list):
        args = self.args

        seed_diff = random.random()
        if seed_diff > 0.8:
            if 'diff_aug' in transform_list:
                diff_image = diffframe(mp_image)
                T, Z, H, W = mp_image.shape
                noise = np.random.normal(0, 0.8, [T, Z, H, W]).astype(np.float32)
                diff_image *= noise
                mp_image = mp_image + diff_image

        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)

        if 'autoaugment' in transform_list:
            mp_image = image_net_autoaugment(mp_image)
            return mp_image

        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)

        seed = random.random()
        if seed > 0.9:
            if 'edge' in transform_list:
                mp_image = edge(mp_image)
        elif seed > 0.8:
            if 'emboss' in transform_list:
                mp_image = emboss(mp_image)
        elif seed > 0.4:
            if 'filter' in transform_list:
                seed2 = random.random()
                if seed2 > 0.8:
                    mp_image = blur(mp_image)
                elif seed2 > 0.6:
                    mp_image = sharpen(mp_image)
                elif seed2 > 0.5:
                    mp_image = mask(mp_image)

        return mp_image

    def __len__(self):
        return len(self.img_list)


def create_loader_CT(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
        mode='instance',
):
    sampler = None
    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader


if __name__ == "__main__":
    import yaml
    import parser
    import argparse
    from tqdm import tqdm

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='F:/oss/nnUNet/nnunetv2/DATASET/nnUNet_raw/Dataset022_FLARE22/imagesTr', type=str)
    parser.add_argument(
        '--train_anno_file', default='data/classification_dataset/labels/train_fold1.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='data/classification_dataset/labels/val_fold1.txt', type=str)
    parser.add_argument('--train_transform_list', default=[
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation', ],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['random_crop'], nargs='+', type=str)
    parser.add_argument(
        '--mode', default='trilinear', type=str)
    parser.add_argument(
        '--csvfile', default='E:/Git/gitproject/miccai2023/data/test.csv', type=str)
    parser.add_argument('--img_size', default=(48, 256, 256),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(96, 512, 512),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)


    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text


    args, args_text = _parse_args()
    args_text = yaml.load(args_text, Loader=yaml.FullLoader)
    args_text['img_size'] = 'xxx'
    print(args_text)

    args.distributed = False
    args.batch_size = 2

    dataset = AbdomenCTDataset(args, is_training=True)
    data_loader = create_loader_CT(dataset, batch_size=3, is_training=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    output = torch.tensor([[[0.1,0.9],[0.3,0.7],[0.1,0.9],[0.3,0.7]],[[0.1,0.9],[0.3,0.7],[0.1,0.9],[0.3,0.7]],[[0.1,0.9],[0.3,0.7],[0.1,0.9],[0.3,0.7]]])

    loss_fn = CrLoss()

    for images, labels in data_loader:
        loss_fn(output,labels)
        print(images.shape)
        print(labels)


    # val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    # val_data_loader = create_loader(val_dataset, batch_size=10, is_training=False)
    # for images, labels in val_data_loader:
    #     print(images.shape)
    #     print(labels)