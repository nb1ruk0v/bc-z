"""Example of using BCZDataset with PyTorch DataLoader."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import BCZDataset

# Configuration - use absolute path from project root
DATA_PATH = Path("data") / "bcz-21task_v9.0.1.tfrecord" / "bcz-21task_v9.0.1.tfrecord"
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to > 0 for parallel loading
IMAGE_SIZE = (100, 100)
NUM_WAYPOINTS = 10


def main():
    """Example: Load BC-Z dataset and iterate through batches."""
    print("=" * 60)
    print("Example: BCZDataset + DataLoader Usage")
    print("=" * 60)

    # Step 1: Create dataset
    print(f"\n1. Creating dataset from: {DATA_PATH}")
    dataset = BCZDataset(
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
        mode="train",
        num_waypoints=NUM_WAYPOINTS,
    )
    print(f"   Dataset size: {len(dataset)} samples")

    # Step 2: Create DataLoader
    print(f"\n2. Creating DataLoader with batch_size={BATCH_SIZE}")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"   Number of batches: {len(dataloader)}")

    # Step 3: Iterate through batches
    print("\n3. Loading a few batches to show structure:")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:
            break

        print(f"\n   Batch {batch_idx}:")
        print(f"      image: {batch['image'].shape}")
        print(f"      sentence_embedding: {batch['sentence_embedding'].shape}")
        print(f"      present_xyz: {batch['present_xyz'].shape}")
        print(f"      present_axis_angle: {batch['present_axis_angle'].shape}")
        print(f"      present_gripper: {batch['present_gripper'].shape}")
        print(f"      future_xyz_residual: {batch['future_xyz_residual'].shape}")
        print(f"      future_axis_angle_residual: {batch['future_axis_angle_residual'].shape}")
        print(f"      future_target_close: {batch['future_target_close'].shape}")

        if batch_idx == 0:
            # Show data ranges for first batch
            print("\n   Data value ranges (batch 0):")
            print(f"      image: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(
                f"      present_xyz: "
                f"[{batch['present_xyz'].min():.3f}, {batch['present_xyz'].max():.3f}]"
            )
            print(
                f"      present_gripper: "
                f"[{batch['present_gripper'].min():.3f}, "
                f"{batch['present_gripper'].max():.3f}]"
            )

    # Step 4: Show typical usage pattern
    print("\n4. Typical training loop pattern:")
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch['image']
            actions = batch['future_xyz_residual']

            # Forward pass
            predictions = model(images, batch['sentence_embedding'])

            # Compute loss
            loss = criterion(predictions, actions)

            # Backward pass
            loss.backward()
            optimizer.step()
    """)

    print("=" * 60)
    print("Done! Ready to use in training loop.")
    print("=" * 60)


if __name__ == "__main__":
    main()
