#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=/workspace/projects_rui/learnsolnmap/data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt1e-3

# python3 ../src/train.py debug=runtime &

python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 0., 0., 0.]" &
python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 1., 0., 0.]" &
python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 1., 1., 0.]" &
python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 1., 1., 1.]" &

# python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] trainer.max_epochs=400 module.optimizer.lr=2.5e-5 init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240205-223702_dvzfsx/checkpoints/epoch948-val_loss3.126e-05.ckpt &

# python3 ../src/eval.py output_dir_suffix=_resnet ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240201-191037_8b1nxq/checkpoints/last.ckpt &
# python3 ../src/eval.py output_dir_suffix=_1000epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240207-012638_zlxnfb/checkpoints/epoch999-val_loss8.814e-06.ckpt &
# python3 ../src/eval.py output_dir_suffix=_2000epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240208-013017_zxe0r4/checkpoints/epoch1999-val_loss4.148e-06.ckpt &
# python3 ../src/eval.py output_dir_suffix=_nd_anchoredenergy ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240201-193028_d73lac/checkpoints/last.ckpt
wait