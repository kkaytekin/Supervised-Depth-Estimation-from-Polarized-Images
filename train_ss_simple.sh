#!/usr/bin/env bash

python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 1 --batch_size 8 \
  --data_path /media/jungo/Research/Datasets/ \
  --data_path_val /media/jungo/Research/Datasets/ \
  --log_dir /media/jungo/Research/Experiments/depth_eccv2022/ \
  --num_epochs 80 --scheduler_step_size 60 --freeze_teacher_epoch 40 --learning_rate 1e-4 \
  --width 320 --height 160 \
  --dataset eccv_depth --split eccv_depth --eval_split eccv_depth \
  --disparity_smoothness 5e-2 \
  --offset 10 \
  --no_matching_augmentation \
  --modality polarization \
  --min_depth 0.1 --max_depth 2.0 \
  --model_name eccv_depth_ss_7

#   --use_future_frame