"""Autoregressive HOPE hindcast driven by future ERA5 atmosphere."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DEFAULT_DATA_DIR, RolloutDataset
from inference_utils import load_ema_model, parse_time, rollout, save_netcdf, to_physical


PROJECT_DIR = Path(__file__).resolve().parent


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HOPE ERA5-forced rollout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=os.environ.get("HOPE_CKPT"), required=False)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--start_date", default="202001010000")
    parser.add_argument("--end_date", default="202512310000")
    parser.add_argument("--start_day", type=int, default=1, help="Initialization day of month; 0 keeps every day")
    parser.add_argument("--rollout_days", type=int, default=215)
    parser.add_argument("--warmup_days", type=int, default=2)
    parser.add_argument("--out_dir", default=str(PROJECT_DIR / "outputs/era5"))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def select_initializations(dataset: RolloutDataset, start: str, end: str, day: int) -> None:
    start_time = parse_time(start)
    end_time = parse_time(end)
    selected = []
    for index in dataset.valid_start_indices:
        init_time = parse_time(dataset.file_dates[index])
        lead1 = parse_time(dataset.file_dates[index + 1])
        leadk = parse_time(dataset.file_dates[index + dataset.rollout_days])
        if lead1 < start_time or leadk > end_time:
            continue
        if day > 0 and init_time.day != day:
            continue
        selected.append(index)
    if not selected:
        raise ValueError("No ERA5 initializations match the requested dates")
    dataset.valid_start_indices = selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if not args.ckpt:
        raise ValueError("Set --ckpt or HOPE_CKPT")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = RolloutDataset(
        args.data_dir,
        args.rollout_days,
        args.warmup_days,
        cache_days=args.rollout_days + args.warmup_days,
    )
    select_initializations(dataset, args.start_date, args.end_date, args.start_day)
    model = load_ema_model(args.ckpt, dataset, device)
    valid_mask = dataset.valid_mask_tensor(device)

    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=1)
    data_loader = DataLoader(dataset, **loader_options)

    with (output / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.ckpt,
                "checkpoint_state": "ema_model",
                "forcing": "ERA5",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "start_day": args.start_day,
                "rollout_days": args.rollout_days,
                "warmup_days": args.warmup_days,
            },
            handle,
            indent=2,
        )

    saved = skipped = 0
    for batch in tqdm(data_loader, desc="ERA5 rollout"):
        start_index = int(batch["start_index"].item())
        init_time = dataset.file_dates[start_index]
        path = output / f"{init_time}.nc"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue

        prediction = rollout(
            model,
            batch["input"],
            batch["warmup_inputs"],
            batch["aux_future"],
            batch["forcing_future"],
            valid_mask,
        )[0].float().cpu().numpy()
        truth = batch["y_future"][0].numpy()
        save_netcdf(
            path,
            dataset,
            to_physical(dataset, prediction),
            truth=to_physical(dataset, truth),
            init_time=init_time,
        )
        saved += 1

    print(f"ERA5 forecasts: saved={saved}, skipped_existing={skipped}, output={output}")


if __name__ == "__main__":
    main()
