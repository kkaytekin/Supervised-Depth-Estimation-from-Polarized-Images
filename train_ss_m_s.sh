#!/usr/bin/env bash

python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 4 --batch_size 10 \
  --data_path /media/jungo/Research/Datasets/_depth_dataset_new/ \
  --data_path_val /media/jungo/Research/Datasets/_depth_dataset_new/ \
  --log_dir /media/jungo/Research/Experiments/depth_eccv2022/ \
  --num_epochs 200 --scheduler_step_size 50 --freeze_teacher_epoch 180 --learning_rate 1e-4 \
  --width 480 --height 320 \
  --dataset eccv_depth --split eccv_depth --eval_split eccv_depth \
  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-2 \
  --offset 10 \
  --no_matching_augmentation --use_future_frame \
  --use_stereo True \
  --modality d435 \
  --model_name eccv_depth_ss_m_s_1