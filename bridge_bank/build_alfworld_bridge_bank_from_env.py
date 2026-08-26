"""Build an OPRD-Bridge bank from freshly sampled ALFWorld turns.

This builder is for model pairs where no old TCOD/rollout buffer exists.  It
collects turn-level ALFWorld prompts from the live text environment, pairs each
prompt with a simple no-think action response, replays the token sequences
through student and teacher models, and saves the same ps_bank.pt format used by
the OPRD-Bridge hidden loss.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)


@dataclass(frozen=True)
class TokenExperience:
    token_ids: list[int]
    prompt_length: int
    response_length: int
    traj_uid: str
    turn_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=8192)
    parser.add_argument("--max-total-response-rows", type=int, default=262144)
    parser.add_argument("--max-pca-rows", type=int, default=32768)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--env-batch-size", type=int, default=16)
    parser.add_argument("--env-max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-policy", default="random", choices=["random", "first"])
    parser.add_argument("--response-style", default="brief", choices=["action_only", "brief"])
    parser.add_argument(
        "--response-source",
        default="fixed",
        choices=["fixed", "student"],
        help="Use a fixed admissible-action response or generate a real student response.",
    )
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--generation-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alfworld-eval-dataset", default="eval_in_distribution")
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def make_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "data": {
                "train_batch_size": args.env_batch_size,
                "val_batch_size": 1,
                "max_prompt_length": args.max_model_len,
                "truncation": "error",
                "apply_chat_template_kwargs": {"enable_thinking": False},
            },
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "seed": args.seed,
                "history_length": args.history_length,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
                "rollout": {"n": 1},
                "alfworld": {"eval_dataset": args.alfworld_eval_dataset},
            },
        }
    )


def normalize_actions(actions: list[str]) -> list[str]:
    cleaned = []
    for action in actions:
        text = str(action).strip()
        if text and text != "help":
            cleaned.append(text)
    return cleaned


def choose_action(actions: list[str], rng: random.Random, policy: str) -> str:
    candidates = normalize_actions(actions)
    if not candidates:
        return "pass"
    if policy == "first":
        return candidates[0]
    return rng.choice(candidates)


def make_response(action: str, style: str) -> str:
    if style == "action_only":
        return f"<action>{action}</action>"
    return (
        "I will choose an admissible action that fits the current observation.\n"
        f"<action>{action}</action>"
    )


def prompt_ids_from_text(tokenizer, prompt_text: str) -> list[int]:
    chat = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def response_ids_from_text(tokenizer, response: str) -> list[int]:
    return tokenizer.encode(response, add_special_tokens=False)


def check_tokenizers(student_path: str, teacher_path: str) -> None:
    student = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if student.get_vocab() != teacher.get_vocab():
        raise RuntimeError(
            "Student and teacher tokenizers differ; shared input_ids cannot be used."
        )
    print("tokenizer_check=ok", flush=True)


def collect_experiences(args: argparse.Namespace) -> list[TokenExperience]:
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    student_model = None
    if args.response_source == "student":
        dtype = resolve_dtype(args.dtype)
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(torch.device(args.device))
        student_model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    config = make_config(args)
    repo_root = Path(__file__).resolve().parent
    while repo_root.name != "ATOD" and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    alf_config_path = (
        repo_root
        / "agent_system"
        / "environments"
        / "env_package"
        / "alfworld"
        / "configs"
        / "config_tw.yaml"
    )
    if not alf_config_path.exists():
        raise RuntimeError(f"Cannot find ALFWorld config: {alf_config_path}")

    if not ray.is_initialized():
        ray.init(num_cpus=args.ray_num_cpus, ignore_reinit_error=True)

    raw_envs = build_alfworld_envs(
        str(alf_config_path),
        seed=args.seed,
        env_num=args.env_batch_size,
        group_n=1,
        resources_per_worker=dict(config.env.resources_per_worker),
        is_train=True,
        env_kwargs={"eval_dataset": args.alfworld_eval_dataset},
    )
    envs = AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)

    experiences: list[TokenExperience] = []
    total_response_rows = 0
    skipped = {"too_long": 0, "empty_response": 0}
    episode_index = 0
    try:
        obs, _ = envs.reset({})
        dones = [False] * args.env_batch_size
        turn_step = 0
        while (
            len(experiences) < args.max_pairs
            and total_response_rows < args.max_total_response_rows
        ):
            prompts = obs["text"]
            admissible = envs.envs.get_admissible_commands
            active_indices = [index for index, done in enumerate(dones) if not done]
            responses = ["pass" for _ in prompts]
            generated_by_index = {}
            if args.response_source == "student" and active_indices:
                prompt_texts = [prompts[index] for index in active_indices]
                prompt_strings = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                    for prompt_text in prompt_texts
                ]
                encoded = tokenizer(
                    prompt_strings,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_model_len - args.max_response_tokens,
                ).to(student_model.device)
                with torch.inference_mode():
                    generated = student_model.generate(
                        **encoded,
                        max_new_tokens=args.max_response_tokens,
                        do_sample=args.generation_do_sample,
                        temperature=args.generation_temperature
                        if args.generation_do_sample
                        else None,
                        top_p=args.generation_top_p if args.generation_do_sample else None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                padded_prompt_len = encoded.input_ids.shape[1]
                for row, index in enumerate(active_indices):
                    response_ids = generated[row, padded_prompt_len:].detach().cpu().tolist()
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    generated_by_index[index] = response
                del encoded, generated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for index in active_indices:
                prompt_text = prompts[index]
                if args.response_source == "student":
                    response = generated_by_index.get(index, "")
                else:
                    action = choose_action(admissible[index], rng, args.action_policy)
                    response = make_response(action, args.response_style)
                responses[index] = response
                prompt_ids = prompt_ids_from_text(tokenizer, prompt_text)
                response_ids = response_ids_from_text(tokenizer, response)
                if not response_ids:
                    skipped["empty_response"] += 1
                    continue
                if len(response_ids) > args.max_response_tokens:
                    response_ids = response_ids[: args.max_response_tokens]
                if len(prompt_ids) + len(response_ids) > args.max_model_len:
                    skipped["too_long"] += 1
                    continue
                experiences.append(
                    TokenExperience(
                        token_ids=prompt_ids + response_ids,
                        prompt_length=len(prompt_ids),
                        response_length=len(response_ids),
                        traj_uid=f"episode{episode_index}_env{index}",
                        turn_step=turn_step,
                    )
                )
                total_response_rows += len(response_ids)
                if (
                    len(experiences) >= args.max_pairs
                    or total_response_rows >= args.max_total_response_rows
                ):
                    break
            if (
                len(experiences) >= args.max_pairs
                or total_response_rows >= args.max_total_response_rows
            ):
                break
            obs, _, dones_arr, _ = envs.step(responses)
            dones = [bool(x) for x in dones_arr]
            turn_step += 1
            if any(dones) or turn_step >= args.env_max_steps:
                episode_index += 1
                obs, _ = envs.reset({})
                dones = [False] * args.env_batch_size
                turn_step = 0
    finally:
        del student_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            envs.envs.close()
        finally:
            if ray.is_initialized():
                ray.shutdown()

    if not experiences:
        raise RuntimeError(f"No usable ALFWorld token experiences collected: {skipped}")
    print(
        json.dumps(
            {
                "experiences": len(experiences),
                "trajectories": len({experience.traj_uid for experience in experiences}),
                "response_tokens": total_response_rows,
                "skipped": skipped,
                "response_source": args.response_source,
                "response_style": args.response_style,
                "action_policy": args.action_policy,
            },
            indent=2,
        ),
        flush=True,
    )
    return experiences


def split_by_trajectory(
    experiences: list[TokenExperience], val_fraction: float, seed: int
) -> tuple[list[TokenExperience], list[TokenExperience]]:
    by_traj: dict[str, list[TokenExperience]] = {}
    for experience in experiences:
        by_traj.setdefault(experience.traj_uid, []).append(experience)
    traj_ids = list(by_traj)
    random.Random(seed).shuffle(traj_ids)
    val_count = max(1, int(round(len(traj_ids) * val_fraction)))
    val_ids = set(traj_ids[:val_count])
    train = [x for x in experiences if x.traj_uid not in val_ids]
    val = [x for x in experiences if x.traj_uid in val_ids]
    if not train or not val:
        raise RuntimeError("Trajectory split produced an empty train or validation set")
    return train, val


def select_offsets(
    experiences: list[TokenExperience], max_rows: int, seed: int
) -> list[list[int]]:
    all_positions = [
        (exp_index, offset)
        for exp_index, experience in enumerate(experiences)
        for offset in range(experience.response_length)
    ]
    if len(all_positions) > max_rows:
        rng = random.Random(seed)
        all_positions = rng.sample(all_positions, max_rows)
    selected = [[] for _ in experiences]
    for exp_index, offset in all_positions:
        selected[exp_index].append(offset)
    return selected


@torch.inference_mode()
def collect_model_rows(
    experiences: list[TokenExperience],
    selected_offsets: list[list[int]],
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    rows = []
    for experience, offsets in tqdm(
        zip(experiences, selected_offsets),
        total=len(experiences),
        desc=desc,
    ):
        if not offsets:
            continue
        input_ids = torch.tensor([experience.token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        positions = torch.tensor(
            [experience.prompt_length + offset for offset in offsets],
            dtype=torch.long,
            device=device,
        )
        layer_rows = [
            hidden[0, positions, :].detach().cpu()
            for hidden in outputs.hidden_states[1:]
        ]
        rows.append(torch.stack(layer_rows, dim=1))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError(f"No hidden rows collected for {model_path}")
    return torch.cat(rows, dim=0)


def proportional_layer_pairs(student_layers: int, teacher_layers: int) -> list[tuple[int, int]]:
    teacher_indices = torch.linspace(0, teacher_layers - 1, steps=student_layers)
    return [
        (index, int(round(float(teacher_indices[index]))))
        for index in range(student_layers)
    ]


@torch.no_grad()
def fit_teacher_pca(rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = rows.float()
    mean = rows.mean(dim=0)
    centered = rows - mean
    fitted_rank = min(rank, max(rows.shape[0] - 1, 1), rows.shape[1])
    components = torch.zeros((rank, rows.shape[1]), dtype=torch.float32)
    if rows.shape[0] <= rows.shape[1]:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        components[:fitted_rank] = vh[:fitted_rank]
    else:
        covariance = (centered.T @ centered) / max(rows.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        components[:fitted_rank] = eigenvectors[:, order].T[:fitted_rank]
    return components, mean, fitted_rank


def train_projector(
    student_train: torch.Tensor,
    teacher_train: torch.Tensor,
    student_val: torch.Tensor,
    teacher_val: torch.Tensor,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, dict]:
    teacher_weights, teacher_mean, fitted_rank = fit_teacher_pca(
        teacher_train[: args.max_pca_rows], rank
    )
    z_train = (teacher_train.float() - teacher_mean) @ teacher_weights.T
    z_val = (teacher_val.float() - teacher_mean) @ teacher_weights.T
    projector = nn.Linear(student_train.shape[-1], rank, bias=False, device=device)
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(student_train.float(), z_train.float()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        projector.train()
        losses = []
        for h_student, z_teacher in loader:
            h_student = h_student.to(device)
            z_teacher = z_teacher.to(device)
            projected = F.normalize(projector(h_student), dim=-1)
            target = F.normalize(z_teacher, dim=-1)
            loss = F.mse_loss(projected, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        projector.eval()
        with torch.no_grad():
            val_projected = projector(student_val.float().to(device)).cpu()
        val_projected_normalized = F.normalize(val_projected, dim=-1)
        z_val_normalized = F.normalize(z_val.float(), dim=-1)
        val_cosine = F.cosine_similarity(
            val_projected_normalized, z_val_normalized, dim=-1
        ).mean().item()
        val_mse = F.mse_loss(val_projected_normalized, z_val_normalized).item()
        history.append(
            {
                "epoch": epoch,
                "train_epoch_mse": float(sum(losses) / max(len(losses), 1)),
                "val_mse": float(val_mse),
                "val_cosine": float(val_cosine),
            }
        )
    return projector, teacher_weights, teacher_mean, {
        "fitted_rank": int(fitted_rank),
        "val_mse": float(history[-1]["val_mse"]),
        "val_cosine": float(history[-1]["val_cosine"]),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    check_tokenizers(args.student_model_path, args.teacher_model_path)
    experiences = collect_experiences(args)
    train, val = split_by_trajectory(experiences, args.val_fraction, args.seed)
    train_offsets = select_offsets(train, args.max_total_response_rows, args.seed)
    val_offsets = select_offsets(
        val, max(args.max_total_response_rows // 10, args.rank * 16), args.seed + 1
    )

    student_train = collect_model_rows(
        train, train_offsets, args.student_model_path, dtype, device, "student train"
    )
    student_val = collect_model_rows(
        val, val_offsets, args.student_model_path, dtype, device, "student val"
    )
    teacher_train = collect_model_rows(
        train, train_offsets, args.teacher_model_path, dtype, device, "teacher train"
    )
    teacher_val = collect_model_rows(
        val, val_offsets, args.teacher_model_path, dtype, device, "teacher val"
    )
    if student_train.shape[0] != teacher_train.shape[0] or student_val.shape[0] != teacher_val.shape[0]:
        raise RuntimeError("Student/teacher hidden row counts do not align")

    layer_pairs = proportional_layer_pairs(student_train.shape[1], teacher_train.shape[1])
    payload = {
        "subspace_mode": "full",
        "projector_type": "linear",
        "rank": int(args.rank),
        "layer_pairs": [
            {"student_layer": int(student), "teacher_layer": int(teacher)}
            for student, teacher in layer_pairs
        ],
        "state_dict": {},
        "frozen_pt_weights": {},
        "frozen_pt_means": {},
    }
    pair_metrics = {}
    for pair_index, (student_layer, teacher_layer) in enumerate(layer_pairs):
        print(
            f"training pair {pair_index + 1}/{len(layer_pairs)}: "
            f"s{student_layer}_t{teacher_layer}",
            flush=True,
        )
        projector, teacher_weights, teacher_mean, metrics = train_projector(
            student_train[:, student_layer, :],
            teacher_train[:, teacher_layer, :],
            student_val[:, student_layer, :],
            teacher_val[:, teacher_layer, :],
            args.rank,
            args,
            device,
        )
        key = f"s{student_layer}_t{teacher_layer}"
        payload["state_dict"][f"projectors.{key}.weight"] = (
            projector.weight.detach().float().cpu()
        )
        payload["frozen_pt_weights"][key] = teacher_weights.cpu()
        payload["frozen_pt_means"][key] = teacher_mean.cpu()
        pair_metrics[key] = metrics

    output_dir = Path(args.output_dir)
    rank_dir = output_dir / f"rank_{args.rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    payload["metrics"] = {
        "num_pairs": len(experiences),
        "num_trajectories": len({experience.traj_uid for experience in experiences}),
        "train_turns": len(train),
        "val_turns": len(val),
        "train_rows": int(student_train.shape[0]),
        "val_rows": int(student_val.shape[0]),
        "student_dim": int(student_train.shape[-1]),
        "teacher_dim": int(teacher_train.shape[-1]),
        "student_layers": int(student_train.shape[1]),
        "teacher_layers": int(teacher_train.shape[1]),
        "rank": int(args.rank),
        "projector_loss": "normalized_mse",
        "response_style": args.response_style,
        "action_policy": args.action_policy,
        "pair_metrics": pair_metrics,
    }
    torch.save(payload, rank_dir / "ps_bank.pt")
    with (rank_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["metrics"], handle, indent=2)
    print(f"saved {rank_dir / 'ps_bank.pt'}", flush=True)
    print(json.dumps(payload["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    # ALFWorld expects this variable; make the failure explicit.
    if "ALFWORLD_DATA" not in os.environ:
        raise RuntimeError("Set ALFWORLD_DATA before running this builder.")
    main()
