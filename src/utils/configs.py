from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    checkpoint_dir: Path = Path("./checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "mps"
    use_wandb: bool = False
