#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=fpu trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 1., 0., 0.]" &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 1., 1., 1.]" &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=none & 
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/scheduler=1cycle &

# python3 ../src/train.py experiment=fpu trainer.devices=[1] module.optimizer.lr=1e-4 &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module.optimizer.lr=5e-4 &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] trainer.max_epochs=2000 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024649_76cpvy/checkpoints/last.ckpt &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] trainer.max_epochs=5000 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &

python3 ../src/eval.py ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240303-171207_tfjemg/checkpoints/epoch999-val_loss9.118e-06.ckpt & 
# python3 ../src/eval.py output_dir_suffix=_freeze ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240222-180507_rjfvfv/checkpoints/epoch999-val_loss1.850e-06.ckpt &
# python3 ../src/eval.py output_dir_suffix=_2000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024649_76cpvy/checkpoints/epoch1999-val_loss8.837e-07.ckpt &
# python3 ../src/eval.py output_dir_suffix=_5000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &
# python3 ../src/eval.py output_dir_suffix=_10000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240215-211116_uffl43/checkpoints/epoch9999-val_loss1.610e-07.ckpt &

wait