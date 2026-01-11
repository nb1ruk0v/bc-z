"""Quick test script to verify BCZDataset loads correctly."""

from pathlib import Path

from src.data.dataset import BCZDataset

# Path to data
data_path = Path("data/bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord")

print(f"Loading dataset from: {data_path}")
print(f"Exists: {data_path.exists()}")

# Create dataset
dataset = BCZDataset(
    data_path=data_path,
    image_size=(100, 100),
    mode="train",
    num_waypoints=10,
)

print("\nDataset loaded!")
print(f"Total samples: {len(dataset)}")

# Test loading a sample
if len(dataset) > 0:
    sample = dataset[0]

    print("\nSample 0:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    # Test a few more samples
    print("\nLoading samples 1-5...")
    for i in range(1, min(6, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: image shape={sample['image'].shape}")

print("\n✅ Dataset test passed!")
