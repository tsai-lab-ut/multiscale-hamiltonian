#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
DATA_DIR=$PROJECT_ROOT/data1/fpu/omega=300/version0
DIR_INPUTS=$DATA_DIR/inputs/
DIR_TARGETS=$DATA_DIR/Dt=1e0_v1/


# generate inputs 
julia src/data/generate_inputs.jl configs/fpu/inputs.toml --output_dir $DIR_INPUTS

# generate targets
mkdir -p $DIR_TARGETS
cp $DIR_INPUTS/U0.csv $DIR_TARGETS
julia src/data/generate_targets.jl configs/fpu/targets.toml --output_dir $DIR_TARGETS

# split into train/test sets
python3 deep_learning/src/split_data.py --data_dir $DIR_TARGETS