#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5"

PORT=${PORT:-69500}

python3 -m torch.distributed.launch --master_port=$PORT --nproc_per_node=2 train.py \
  --data_dir E:\LLD-MMRI2023\main\data\classification_dataset\images \
  --train_anno_file E:\Git\gitproject\miccai2023\data\trainval_labels\train_fold1.txt \
  --val_anno_file E:\Git\gitproject\miccai2023\data\trainval_labels\val_fold1.txt \
  --batch-size 4 \
  --path_num 2 \
  --cut_crop_size 48 256 256 \
  --model UniverSeg3D \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --epochs 300 \
  --output output/ \
  --train_transform_list random_crop z_flip x_flip y_flip rotation edge emboss filter \
  --crop_size 14 112 112 \
  --pretrained \
  --sampling sqrt \
  --mixup \
  --cb_loss \
  --smoothing 0.1 \
  --img_size 96 512 512 \
  --drop-path 0.1 \
  --eval-metric f1 kappa \
  --mode trilinear \
  # --train_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels.txt \
  # --val_anno_file /home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/output/labels.txt \