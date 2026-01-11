# BCZDataset Usage Guide

## Overview

`BCZDataset` loads BC-Z robotic manipulation data from TFRecord files into PyTorch tensors.

## Data Format

The dataset expects TFRecord files with the following structure:

```
data/
└── bcz-21task_v9.0.1.tfrecord/
    └── bcz-21task_v9.0.1.tfrecord/
        ├── train-00000-of-00100
        ├── train-00001-of-00100
        └── ...
```

### Features per sample:

- `present/image/encoded`: JPEG encoded image
- `sentence_embedding`: (512,) task description embedding
- `present/xyz`: (3,) current end-effector position
- `present/axis_angle`: (3,) current orientation (axis-angle representation)
- `present/sensed_close`: (1,) current gripper state
- `future/xyz_residual`: (30,) residual xyz actions for 10 waypoints
- `future/axis_angle_residual`: (30,) residual orientation actions
- `future/target_close`: (10,) target gripper states

## Basic Usage

```python
from pathlib import Path
from src.data.dataset import BCZDataset

# Create dataset
dataset = BCZDataset(
    data_path=Path("data/bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord"),
    image_size=(100, 100),
    mode="train",
    num_waypoints=10,
)

# Access a sample
sample = dataset[0]
print(sample.keys())
# Output: dict_keys(['image', 'sentence_embedding', 'present_xyz',
#                    'present_axis_angle', 'present_gripper',
#                    'future_xyz_residual', 'future_axis_angle_residual',
#                    'future_target_close'])
```

## With DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

for batch in dataloader:
    images = batch['image']  # (B, 3, H, W)
    actions = batch['future_xyz_residual']  # (B, num_waypoints, 3)
    # ... training code
```

## Output Shapes

For batch_size=32, num_waypoints=10, image_size=(100, 100):

| Key | Shape | Description |
|-----|-------|-------------|
| `image` | (32, 3, 100, 100) | RGB images |
| `sentence_embedding` | (32, 512) | Task embeddings |
| `present_xyz` | (32, 3) | Current position |
| `present_axis_angle` | (32, 3) | Current orientation |
| `present_gripper` | (32, 1) | Current gripper state |
| `future_xyz_residual` | (32, 10, 3) | XYZ residual actions |
| `future_axis_angle_residual` | (32, 10, 3) | Orientation residuals |
| `future_target_close` | (32, 10, 1) | Gripper targets |

## Custom Transforms

```python
def custom_transform(image_np):
    """Custom image augmentation."""
    # image_np: (H, W, 3) numpy array in [0, 1]
    # Apply augmentations...
    return image_np

dataset = BCZDataset(
    data_path=data_path,
    transform=custom_transform,
)
```

## Notes

- Dataset loads all samples into memory for fast random access
- Images are automatically decoded from JPEG and normalized to [0, 1]
- Residual actions are relative to current state
- 10 waypoints = predictions for next 10 timesteps
