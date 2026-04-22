"""BC-Z training entrypoint.

Usage:
    uv run python main.py --config configs/default.yaml
    uv run python main.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trackio
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import BCZDataset
from src.models.policy import BCZPolicy
from src.training.trainer import Trainer


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(
    cfg: dict[str, Any],
    path_key: str,
    mode: str,
    max_samples_key: str,
) -> BCZDataset | Subset | None:
    path = cfg["data"].get(path_key)
    if not path:
        return None
    ds = BCZDataset(
        data_path=path,
        image_size=tuple(cfg["data"]["image_size"]),
        num_waypoints=cfg["data"]["num_waypoints"],
        mode=mode,
    )
    cap = cfg["data"].get(max_samples_key)
    if cap is not None:
        ds = Subset(ds, range(min(cap, len(ds))))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    train_ds = build_dataset(cfg, "train_path", "train", "max_train_samples")
    val_ds = build_dataset(cfg, "val_path", "val", "max_val_samples")
    if train_ds is None:
        raise ValueError("config.data.train_path is required")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
        )
        if val_ds is not None
        else None
    )

    policy = BCZPolicy(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        embedding_dim=cfg["model"]["embedding_dim"],
        state_dim=cfg["model"]["state_dim"],
        num_waypoints=cfg["data"]["num_waypoints"],
        hidden_dim=cfg["model"]["hidden_dim"],
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["training"]["lr"])

    trackio.init(
        project=cfg["trackio"]["project"],
        name=cfg["trackio"].get("run_name"),
        config=cfg,
    )

    def log_fn(
        record: dict[str, Any],
    ) -> None:
        trackio.log(record)
        msg = " | ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
        )
        print(msg)

    trainer = Trainer(
        model=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=cfg["training"]["device"],
        loss_weights=cfg["training"]["loss_weights"],
        embedding_noise_std=cfg["training"]["embedding_noise_std"],
        log_fn=log_fn,
    )

    try:
        trainer.fit(num_epochs=cfg["training"]["num_epochs"])
    finally:
        trackio.finish()


if __name__ == "__main__":
    main()
