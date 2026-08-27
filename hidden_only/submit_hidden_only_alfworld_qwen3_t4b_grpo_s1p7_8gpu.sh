#!/bin/bash

set -e

vc submit \
  --type pytorch \
  --cluster D14 \
  --partition pdgpu-aispeech-ai \
  --project sds \
  --image docker.v2.aispeech.com/hpc/ai_sds-verl-megatron-ray:v1.0 \
  --job alfworld-hidden-only-qwen3-t4b-grpo-s1.7b \
  --num-task 1 \
  --cpu-per-task 56 \
  --mem-per-task 560G \
  --gpu-per-task 8 \
  --dir "/hpc_stor01/home/yong.tang_sx/agents/feedback-oprd" \
  --cmd "bash hidden_only/run_alfworld_hidden_only_s1p7b_t4b_grpo_8gpu.sh"
