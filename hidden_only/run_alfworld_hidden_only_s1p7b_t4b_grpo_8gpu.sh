#!/bin/bash

set -ex

ENGINE=${ENGINE:-vllm}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate atod

REPO_ROOT="/hpc_stor01/home/yong.tang_sx/agents/feedback-oprd"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# ============================================================
# Logging
# ============================================================
LOG_DIR="$REPO_ROOT/outputs/vc_logs"
RUN_NAME="qwen3_t4b_s1.7b_alfworld_oprd_hidden_only"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

export PYTHONUNBUFFERED=1

# Keep output visible in vc logs while also writing a local log.
exec > >(tee -a "$LOG_FILE") 2>&1

# Always point to the latest run.
ln -sfn "$LOG_FILE" "$LOG_DIR/${RUN_NAME}_latest.log"

echo "============================================================"
echo "RUN_NAME=$RUN_NAME"
echo "START_TIME=$(date)"
echo "HOSTNAME=$(hostname)"
echo "LOG_FILE=$LOG_FILE"
echo "============================================================"

# vLLM memory-pool compatibility.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

# Force Triton to use its bundled CUDA 12.8 ptxas.
unset TRITON_PTXAS_PATH
export TRITON_PTXAS_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"

test -x "$TRITON_PTXAS_PATH"

echo "===== Triton / CUDA environment ====="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"

"$TRITON_PTXAS_PATH" --version

python - <<'PY'
import torch
import triton
from triton.backends.nvidia.compiler import get_ptxas

tool = get_ptxas()

print("Python-selected ptxas:", tool.path)
print("Python-selected ptxas version:", tool.version)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("triton:", triton.__version__)

assert tool.version == "12.8", (
    f"Expected Triton ptxas 12.8, got {tool.version}"
)
PY

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

num_cpus_per_env_worker=0.1

# =====================================================
# OPRD-Bridge hidden-only configuration
# =====================================================
use_external_teacher=true
sod_mode="uniform"
opd_coef=1.0
opd_only=true

# Model paths
# Student: Qwen3-1.7B
# Teacher: Qwen3-4B-GRPO-ALFWorld
student_model_path=/hpc_stor01/home/yong.tang_sx/models/qwen/Qwen3-1.7B
teacher_model_path=/hpc_stor01/home/yong.tang_sx/models/Qwen3-4B-GRPO-ALFWorld-step150-SR73p44

bridge_bank_path=/hpc_stor01/home/yong.tang_sx/agents/feedback-oprd/artifacts/bridge_bank/bank_alfworld_1p7b_4bgrpo_r64.pt

# =====================================================
# Training configuration
# =====================================================
train_data_size=16
val_data_size=128
group_size=8
experiment_name="oprd_bridge_hidden_only_alfworld_1p7b_to_4bgrpo_tp1_step150"
export ALFWORLD_DATA=$HOME/data/alfworld
export HF_ENDPOINT=https://hf-mirror.com

set +x
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -x

python3 -m verl.trainer.main_sod_oprd_bridge_backbone_cached \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.use_oprd_bridge_hidden_loss=True \
    +actor_rollout_ref.actor.oprd_bridge_checkpoint_path=$bridge_bank_path \
    +actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef=1.0 \
    +actor_rollout_ref.actor.oprd_bridge_hidden_only_update=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sod.use_external_teacher=$use_external_teacher \
    +algorithm.sod.mode=$sod_mode \
    +algorithm.sod.opd_coef=$opd_coef \
    +algorithm.sod.epsilon=1e-6 \
    +algorithm.sod.delta=0.2 \
    +algorithm.sod.opd_only=$opd_only \
    +algorithm.sod.skills_dir=skills/alfworld \
    +algorithm.sod.skill_all=false \
    +algorithm.sod.hidden_signal.enabled=true \
    +algorithm.sod.hidden_signal.bridge_checkpoint_path=$bridge_bank_path \
    +algorithm.sod.hidden_signal.loss_coef=1.0 \
    +algorithm.sod.hidden_signal.micro_batch_size=32 \
    +algorithm.sod.hidden_signal.response_last_k=512 \
    +algorithm.sod.hidden_signal.disable_logprob_opd=true \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "$@"
