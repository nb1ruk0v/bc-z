"""Create small test TFRecord dataset from real data."""

import sys
from pathlib import Path

import tensorflow as tf

# Paths
REAL_DATA_PATH = (
    Path("data")
    / "bcz-21task_v9.0.1.tfrecord"
    / "bcz-21task_v9.0.1.tfrecord"
    / "train-00000-of-00100"
)
TEST_DATA_PATH = Path(__file__).parent / "test_data.tfrecord"

NUM_SAMPLES = 5  # Only take 5 samples for testing


def create_test_tfrecord():
    """Extract 5 samples from real data and save to test TFRecord."""
    if not REAL_DATA_PATH.exists():
        print(f"Error: Real data not found at {REAL_DATA_PATH}")
        print("Please ensure the BC-Z dataset is available.")
        sys.exit(1)

    print(f"Reading samples from: {REAL_DATA_PATH}")

    # Define feature spec
    # num_waypoints = 10
    # example features = {
    #     "present/image/encoded": tf.io.FixedLenFeature([], tf.string),
    #     "sentence_embedding": tf.io.FixedLenFeature([512], tf.float32),
    #     "present/xyz": tf.io.FixedLenFeature([3], tf.float32),
    #     "present/axis_angle": tf.io.FixedLenFeature([3], tf.float32),
    #     "present/sensed_close": tf.io.FixedLenFeature([1], tf.float32),
    #     "future/xyz_residual": tf.io.FixedLenFeature([3 * num_waypoints], tf.float32),
    #     "future/axis_angle_residual": tf.io.FixedLenFeature([3 * num_waypoints], tf.float32),
    #     "future/target_close": tf.io.FixedLenFeature([1 * num_waypoints], tf.float32),
    # }

    # Read samples
    dataset = tf.data.TFRecordDataset(str(REAL_DATA_PATH))
    samples = []

    for i, serialized_example in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        samples.append(serialized_example.numpy())
        print(f"  Extracted sample {i + 1}/{NUM_SAMPLES}")

    # Write to test file
    print(f"\nWriting test data to: {TEST_DATA_PATH}")
    with tf.io.TFRecordWriter(str(TEST_DATA_PATH)) as writer:
        for sample in samples:
            writer.write(sample)

    print(f"✅ Created test TFRecord with {len(samples)} samples")
    print(f"   File size: {TEST_DATA_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    create_test_tfrecord()
