#!/usr/bin/env bash

python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
  --data_path /media/jungo/Research/Datasets/ \
  --data_path_val /media/jungo/Research/Datasets/ \
  --log_dir /media/jungo/Research/Experiments/depth_eccv2022/ \
  --num_epochs 20 --scheduler_step_size 15 --freeze_teacher_epoch 20 --learning_rate 1e-4 \
  --width 320 --height 192 \
  --dataset eccv_depth --split eccv_depth --eval_split eccv_depth \
  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-3 \
  --offset 10 \
  --no_matching_augmentation \
  --depth_supervision True \
  --model_name supervised_GT_plus_ss_4