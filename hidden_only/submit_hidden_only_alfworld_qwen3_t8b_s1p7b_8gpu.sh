#!/bin/bash

set -e

vc submit \
  --type pytorch \
  --cluster D14 \
  --partition pdgpu-aispeech-ai \
  --project sds \
  --image docker.v2.aispeech.com/hpc/ai_sds-verl-megatron-ray:v1.0 \
  --job alfworld-oprd-bridge-hidden-only-qwen3-t8b-s1.7b \
  --num-task 1 \
  --cpu-per-task 84 \
  --mem-per-task 720G \
  --gpu-per-task 8 \
  --dir "/hpc_stor01/home/yong.tang_sx/agents/feedback-oprd" \
  --cmd "bash hidden_only/run_formal_qwen3_t8b_s1p7b_8gpu.sh"
