#!/usr/bin/env bash

python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
  --data_path /media/jungo/Research/Datasets/HAMMER/train/ \
  --data_path_val /media/jungo/Research/Datasets/HAMMER/test_unseen/ \
  --log_dir /media/jungo/Research/Experiments/AT3DCV/ \
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
  --model_name Letsmakeitworkfinally \
  --overfit True

#python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
#  --data_path /media/jungo/Research/Datasets/HAMMER/train/ \
#  --data_path_val /media/jungo/Research/Datasets/HAMMER/test_unseen/ \
#  --log_dir /media/jungo/Research/Experiments/AT3DCV/ \
#  --num_epochs 50 --scheduler_step_size 15 --freeze_teacher_epoch 50 --learning_rate 1e-4 \
#  --width 480 --height 320 \  # to get faster results its small (480-320 is lowest resoltion for resnet encoder)
#  --dataset HAMMER --split HAMMER --eval_split HAMMER_unseen \
#  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-3 \
#  --offset 10 \  # checks if frames +/- 10 is there and start using it when both are there
#  --no_matching_augmentation \
#  --depth_supervision_only True \
#  --depth_supervision True \
#  --modality polarization \  # searching for all png in the list
#  --depth_modality _gt \  # for supervision  (or depth_d435/l515 folders)
#  --model_name RN18_supervised_GT_RGB_only


#python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
#  --data_path /media/jungo/Research/Datasets/HAMMER/train/ \
#  --data_path_val /media/jungo/Research/Datasets/HAMMER/test_unseen/ \
#  --log_dir /media/jungo/Research/Experiments/AT3DCV/ \
#  --num_epochs 50 --scheduler_step_size 15 --freeze_teacher_epoch 50 --learning_rate 1e-4 \
#  --width 480 --height 320 \
#  --dataset HAMMER --split HAMMER --eval_split HAMMER_unseen \
#  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-3 \
#  --offset 10 \
#  --no_matching_augmentation \
#  --depth_supervision_only True \
#  --depth_supervision True \
#  --modality polarization \
#  --depth_modality _gt \
#  --train_dpt True \
#  --scales 0 \
#  --model_name DPT_supervised_GT_RGB_only
#
#
#python3 -m manydepth.train --png --num_depth_bins 96  --num_workers 8 --batch_size 12 \
#  --data_path /media/jungo/Research/Datasets/HAMMER/train/ \
#  --data_path_val /media/jungo/Research/Datasets/HAMMER/test_unseen/ \
#  --log_dir /media/jungo/Research/Experiments/AT3DCV/ \
#  --num_epochs 50 --scheduler_step_size 15 --freeze_teacher_epoch 50 --learning_rate 1e-4 \
#  --width 480 --height 320 \
#  --dataset HAMMER --split HAMMER --eval_split HAMMER_unseen \
#  --min_depth 0.1 --max_depth 2.0 --disparity_smoothness 1e-3 \
#  --offset 10 \
#  --no_matching_augmentation \
#  --depth_supervision_only True \
#  --depth_supervision True \
#  --modality polarization \
#  --depth_modality _gt \
#  --train_dpt True \
#  --midas True \
#  --scales 0 \
#  --model_name MIDAS_supervised_GT_RGB_only
