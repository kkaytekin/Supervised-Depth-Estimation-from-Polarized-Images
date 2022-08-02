#!/usr/bin/env bash

python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
  --data_path /media/jungo/Research/Datasets/HAMMER/train/ \
  --data_path_val /media/jungo/Research/Datasets/HAMMER/test_unseen/ \
  --log_dir /media/jungo/Research/Experiments/AT3DCV/EXPERIMENTS \
  --num_epochs 50 --scheduler_step_size 15 --freeze_teacher_epoch 50 --learning_rate 1e-4 \
  --width 480 --height 320 \
  --dataset HAMMER --split HAMMER --eval_split HAMMER_unseen \
  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-3 \
  --offset 10 \
  --no_matching_augmentation \
  --depth_supervision_only True \
  --depth_supervision True \
  --modality polarization \
  --depth_modality _gt \
  --model_name Letsmakeitfinallywork_RGB \
#  --augment_xolp \
#  --augment_normals \
#  --overfit True