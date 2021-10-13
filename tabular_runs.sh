#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset power
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5  --dataset gas
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset hepmass
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset miniboone

CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset power --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=1
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset gas --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=1
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset hepmass --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=1
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset miniboone --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=1

CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset power --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=10
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset gas --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=10
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset hepmass --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=10
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset miniboone --config log_jacobian_method=hutch_with_cg --config hutchinson_samples=10

CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset power --baseline
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset gas --baseline
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset hepmass --baseline
CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --num-seeds 5 --dataset miniboone --baseline

./tabular_evaluate.py
