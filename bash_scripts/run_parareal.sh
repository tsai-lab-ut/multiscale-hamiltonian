#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
OUTPUT_DIR=$PROJECT_ROOT/out/fpu/omega=300/202403072132

julia src/parareal/run_parareal.jl configs/fpu/parareal_sequential.toml --output_dir $OUTPUT_DIR/ref
julia src/parareal/run_parareal.jl configs/fpu/parareal_plain.toml --output_dir $OUTPUT_DIR/plain
julia src/parareal/run_parareal.jl configs/fpu/parareal_procrustes.toml --output_dir $OUTPUT_DIR/procrustes
julia src/parareal/run_parareal.jl configs/fpu/parareal_interpolative.toml --output_dir $OUTPUT_DIR/interpolative