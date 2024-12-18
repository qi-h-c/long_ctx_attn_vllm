#!/bin/bash -x
export LD_LIBRARY_PATH=/home/hyf/anaconda3/envs/llm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
pip uninstall vllm -y
# 查看当前版本
git rev-parse HEAD
export VLLM_COMMIT=a1c02058baf47be1a91ee743378a340ee1b10416
# 安装wheel
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install -e .
python examples/offline_inference.py 