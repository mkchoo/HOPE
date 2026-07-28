"""Train the released HOPE model with the K1 -> K4 -> K8 curriculum."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from adam_atan2_pytorch import AdamAtan2
from dataloader import DEFAULT_DATA_DIR, RolloutDataset
from exp_config.config import Config
from models.hope import HOPE
from models.utils import weighted_masked_mean


PROJECT_DIR = Path(__file__).resolve().parent
EMA_DECAY = 0.999
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SAVE_EVERY = 5
VARIABLE_WEIGHTS = {
    "siconc": 1.0,
    "sithick": 2.0,
    "zos": 1.5,
    "thetao": 5.0,
    "so": 10.0,
    "uo": 1.0,
    "vo": 1.0,
}


@dataclass(frozen=True)
class Stage:
    rollout_days: int
    warmup_days: int
    epochs: int
    learning_rate: float
    batch_size: int
    cache_days: int


STAGES = {
    "K1": Stage(rollout_days=1, warmup_days=0, epochs=250, learning_rate=2e-4, batch_size=6, cache_days=1),
    "K4": Stage(rollout_days=4, warmup_days=1, epochs=100, learning_rate=1e-4, batch_size=2, cache_days=5),
    "K8": Stage(rollout_days=8, warmup_days=2, epochs=100, learning_rate=1e-4, batch_size=1, cache_days=10),
}


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


class ModelEMA:
    def __init__(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model).eval()
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            new_value = source[name].detach().to(value.device)
            if torch.is_floating_point(value):
                value.mul_(EMA_DECAY).add_(new_value.to(value.dtype), alpha=1.0 - EMA_DECAY)
            else:
                value.copy_(new_value)


def observed_warmup(model, inputs: torch.Tensor, device: torch.device):
    if inputs.shape[1] == 0:
        return None
    state = None
    with torch.no_grad():
        for day in range(inputs.shape[1]):
            state = model.spinup_state(inputs[:, day].to(device, non_blocking=True), state)
            state = tuple(value.detach() for value in state)
    return state


def variable_weighted_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset: RolloutDataset,
    valid_mask: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).abs()
    total = error.new_tensor(0.0)
    denominator = 0.0
    for name, channel_slice in dataset.y_var_slices.items():
        weight = VARIABLE_WEIGHTS[name]
        value = weighted_masked_mean(
            error[:, channel_slice],
            mask=valid_mask[:, channel_slice],
            spatial_weight=latitude_weights[:, channel_slice],
        )
        total = total + weight * value
        denominator += weight
    return total / denominator


def rollout_loss(
    model,
    batch,
    dataset: RolloutDataset,
    device: torch.device,
    valid_mask: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    warmup = batch["warmup_inputs"]
    x = batch["input"].to(device, non_blocking=True)
    auxiliary = batch["aux_future"].to(device, non_blocking=True)
    forcing = batch["forcing_future"].to(device, non_blocking=True)
    targets = batch["y_future"].to(device, non_blocking=True)

    state = observed_warmup(model, warmup, device)
    loss = x.new_tensor(0.0)
    for lead in range(dataset.rollout_days):
        state, prediction = model(x, state)
        loss = loss + variable_weighted_mae(
            prediction, targets[:, lead], dataset, valid_mask, latitude_weights
        )
        if lead + 1 < dataset.rollout_days:
            prediction = prediction * valid_mask.to(prediction.dtype)
            x = torch.cat([prediction, auxiliary[:, lead], forcing[:, lead]], dim=1)
    return loss / dataset.rollout_days


def loader(dataset, batch_size: int, workers: int, *, shuffle: bool) -> DataLoader:
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def checkpoint_payload(epoch, model, optimizer, ema, config):
    return {
        "epoch": int(epoch),
        "model": unwrap(model).state_dict(),
        "ema_model": ema.model.state_dict(),
        "ema_decay": EMA_DECAY,
        "ema_num_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "optimizer_name": "AdamAtan2",
        "config": config.to_dict(),
    }


def load_weights(model, state_dict, label: str) -> None:
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {label} weights ({len(state_dict)} tensors)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HOPE curriculum training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_root", default=str(PROJECT_DIR / "runs"))
    parser.add_argument("--init_ckpt", help="Previous curriculum-stage checkpoint; loads ema_model only")
    parser.add_argument("--resume_ckpt", help="Same-stage checkpoint; restores model, EMA, optimizer, and epoch")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    stage = STAGES[args.stage]
    set_seed(42)

    run_dir = Path(args.out_root) / f"hope_{args.stage}"
    checkpoint_dir = run_dir / "checkpoints"
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "tb")),
    )
    accelerator.init_trackers(
        project_name=f"hope_{args.stage}",
        config={
            "stage": args.stage,
            **asdict(stage),
            "ema_decay": EMA_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "variable_weights": json.dumps(VARIABLE_WEIGHTS),
        },
    )

    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": args.stage,
                    "data_dir": args.data_dir,
                    "out_root": args.out_root,
                    "init_ckpt": args.init_ckpt,
                    "resume_ckpt": args.resume_ckpt,
                    "num_workers": args.num_workers,
                    **asdict(stage),
                },
                handle,
                indent=2,
            )

    train_set = RolloutDataset(
        args.data_dir,
        stage.rollout_days,
        stage.warmup_days,
        start_date="199301010000",
        end_date="201712310000",
        cache_days=stage.cache_days,
    )
    val_set = RolloutDataset(
        args.data_dir,
        stage.rollout_days,
        stage.warmup_days,
        start_date="201801010000",
        end_date="201912310000",
        cache_days=stage.cache_days,
    )
    train_loader = loader(train_set, stage.batch_size, args.num_workers, shuffle=True)
    val_loader = loader(val_set, stage.batch_size, args.num_workers, shuffle=False)

    config = Config(
        name="rollout",
        height=len(train_set.lat_values),
        width=len(train_set.lon_values),
        in_chans=train_set.n_state_channels + 2 + 8,
        out_chans=train_set.n_state_channels,
    ).sync_dims()
    model = HOPE(config)
    optimizer = AdamAtan2(model.parameters(), lr=stage.learning_rate, weight_decay=WEIGHT_DECAY)

    resume = None
    if args.resume_ckpt:
        resume = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        load_weights(model, resume["model"], "resume model")
    elif args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        load_weights(model, init["ema_model"], "previous-stage EMA")
    elif args.stage != "K1":
        raise ValueError(f"{args.stage} requires --init_ckpt or --resume_ckpt")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    ema = ModelEMA(unwrap(model))
    start_epoch = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        ema.model.load_state_dict(resume["ema_model"], strict=True)
        ema.updates = int(resume.get("ema_num_updates", 0))
        start_epoch = int(resume["epoch"])

    valid_mask = train_set.valid_mask_tensor(accelerator.device)
    latitude_weights = train_set.latitude_weights_tensor().to(accelerator.device)
    val_mask = val_set.valid_mask_tensor(accelerator.device)
    val_latitude_weights = val_set.latitude_weights_tensor().to(accelerator.device)

    for epoch in range(start_epoch + 1, stage.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        progress = tqdm(
            train_loader,
            desc=f"{args.stage} epoch {epoch}/{stage.epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss = rollout_loss(
                    model, batch, train_set, accelerator.device, valid_mask, latitude_weights
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(unwrap(model))

            batch_size = int(batch["input"].shape[0])
            train_sum += float(loss.detach()) * batch_size
            train_count += batch_size
            progress.set_postfix(loss=float(loss.detach()))

        train_stats = torch.tensor([train_sum, train_count], device=accelerator.device)
        train_stats = accelerator.reduce(train_stats, reduction="sum")
        train_loss = float((train_stats[0] / train_stats[1]).item())
        if accelerator.is_main_process:
            accelerator.log({"loss/train": train_loss}, step=epoch)

        if epoch % SAVE_EVERY:
            continue

        model.eval()
        raw_sum = ema_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in tqdm(
                val_loader, desc="validation", leave=False, disable=not accelerator.is_local_main_process
            ):
                batch_size = int(batch["input"].shape[0])
                with accelerator.autocast():
                    raw_loss = rollout_loss(
                        unwrap(model), batch, val_set, accelerator.device, val_mask, val_latitude_weights
                    )
                    ema_loss = rollout_loss(
                        ema.model, batch, val_set, accelerator.device, val_mask, val_latitude_weights
                    )
                raw_sum += float(raw_loss) * batch_size
                ema_sum += float(ema_loss) * batch_size
                val_count += batch_size

        val_stats = torch.tensor([raw_sum, ema_sum, val_count], device=accelerator.device)
        val_stats = accelerator.reduce(val_stats, reduction="sum")
        raw_val = float((val_stats[0] / val_stats[2]).item())
        ema_val = float((val_stats[1] / val_stats[2]).item())

        if accelerator.is_main_process:
            accelerator.log({"loss/val": raw_val, "loss/ema_val": ema_val}, step=epoch)
            torch.save(
                checkpoint_payload(epoch, model, optimizer, ema, config),
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
