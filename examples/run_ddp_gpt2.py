"""
Standard PyTorch DDP baseline for GPT-2 on the tiny-Shakespeare dataset.

Launch with torchrun:
    torchrun --nproc_per_node <NUM_GPUS> run_ddp_gpt2.py [OPTIONS]

For a fair comparison with the Oobleck run_gpt2.py script, match
--global_batch_size, --num_epoch, and --model_name_or_path.
"""

import functools
import os

import click
import datasets
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import Adam


def tokenize_batch_for_pretrain(
    batch, tokenizer: PreTrainedTokenizer, max_length: int = 1024
):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


@click.command()
@click.option(
    "--model_name_or_path", type=str, default="gpt2", help="Model name or path."
)
@click.option("--global_batch_size", type=int, default=96, help="Global batch size.")
@click.option("--microbatch_size", type=int, default=2, help="Per-GPU batch size.")
@click.option("--num_epoch", type=int, default=3, help="Number of epochs.")
@click.option("--warmup_fraction", type=float, default=0.1, help="Warmup fraction.")
def main(
    model_name_or_path: str,
    global_batch_size: int,
    microbatch_size: int,
    num_epoch: int,
    warmup_fraction: float,
):
    # ── Distributed setup ────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    is_main = local_rank == 0

    # ── Model ─────────────────────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)
    model.gradient_checkpointing_enable()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # ── Data ──────────────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset("karpathy/tiny_shakespeare")["train"]

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        sampler=sampler,
        collate_fn=functools.partial(
            tokenize_batch_for_pretrain,
            tokenizer=tokenizer,
            max_length=config.max_position_embeddings,
        ),
        pin_memory=True,
    )

    # ── Optimizer & LR schedule ───────────────────────────────────────────────
    optimizer = Adam(model.parameters())

    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # Gradient accumulation steps so effective batch size matches global_batch_size
    grad_accum_steps = max(1, global_batch_size // (microbatch_size * world_size))

    # ── Mixed precision ───────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(num_epoch):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()

        with tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
            disable=not is_main,
        ) as pbar:
            for step, batch in pbar:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if is_main:
                    pbar.set_postfix(loss=loss.item() * grad_accum_steps)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
