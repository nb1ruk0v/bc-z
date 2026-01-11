"""Dataset class for BC-Z training data stored in TFRecord format."""

import glob
import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class BCZDataset(Dataset):
    """
    Dataset for BC-Z robotic manipulation data stored in TFRecord format.

    Data structure (from TFRecord):
        - present/image/encoded: JPEG encoded image
        - sentence_embedding: (512,) task embedding
        - present/xyz: (3,) current position
        - present/axis_angle: (3,) current orientation
        - present/sensed_close: (1,) current gripper state
        - future/xyz_residual: (3*num_waypoints,) xyz residual actions
        - future/axis_angle_residual: (3*num_waypoints,) orientation residual
        - future/target_close: (1*num_waypoints,) gripper targets
    """

    def __init__(
        self,
        data_path: Path | str,
        image_size: tuple[int, int] = (100, 100),
        transform: Callable | None = None,
        mode: str = "train",
        num_waypoints: int = 10,
    ):
        """
        Initialize BCZ dataset.

        Args:
            data_path: Path to directory containing TFRecord files
            image_size: Target image size (H, W) for resizing
            transform: Optional transform to apply to images
            mode: Dataset mode ('train', 'val', 'test')
            num_waypoints: Number of future waypoints to predict

        Raises:
            FileNotFoundError: If data_path does not exist
            ValueError: If no TFRecord files found
            ImportError: If tensorflow is not installed
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.num_waypoints = num_waypoints

        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Import tensorflow for TFRecord reading
        try:
            import tensorflow as tf

            self.tf = tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for reading TFRecord files. Install with: uv add tensorflow"
            ) from e

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()

        # Load dataset into memory (for indexing)
        self.samples = []
        self._load_dataset()

        print(f"Loaded {len(self.samples)} samples from {len(self.tfrecord_files)} files")

    def _find_tfrecord_files(self) -> list[str]:
        """
        Find TFRecord files in the data path.

        Returns:
            List of TFRecord file paths

        Raises:
            ValueError: If no TFRecord files found
        """
        # Handle direct path to a file
        if self.data_path.is_file():
            if self.data_path.suffix == ".tfrecord" or self.data_path.name.startswith("train-"):
                return [str(self.data_path)]
            else:
                raise ValueError(f"Expected TFRecord file, got: {self.data_path}")

        # Handle directory - search for TFRecord files
        # Search patterns
        patterns = [
            str(self.data_path / f"{self.mode}-*"),
            str(self.data_path / "**" / f"{self.mode}-*"),
            str(self.data_path / "*.tfrecord"),
            str(self.data_path / "**" / "*.tfrecord"),
        ]

        tfrecord_files = []
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            tfrecord_files.extend(files)

        # Remove duplicates
        tfrecord_files = sorted(list(set(tfrecord_files)))

        if len(tfrecord_files) == 0:
            raise ValueError(f"No TFRecord files found in {self.data_path} with mode='{self.mode}'")

        return tfrecord_files

    def _load_dataset(self) -> None:
        """
        Load all samples from TFRecord files into memory.

        This enables random access via indexing.
        """
        # Define TFRecord feature spec
        features = {
            "present/image/encoded": self.tf.io.FixedLenFeature([], self.tf.string),
            "sentence_embedding": self.tf.io.FixedLenFeature([512], self.tf.float32),
            "present/xyz": self.tf.io.FixedLenFeature([3], self.tf.float32),
            "present/axis_angle": self.tf.io.FixedLenFeature([3], self.tf.float32),
            "present/sensed_close": self.tf.io.FixedLenFeature([1], self.tf.float32),
            "future/xyz_residual": self.tf.io.FixedLenFeature(
                [3 * self.num_waypoints], self.tf.float32
            ),
            "future/axis_angle_residual": self.tf.io.FixedLenFeature(
                [3 * self.num_waypoints], self.tf.float32
            ),
            "future/target_close": self.tf.io.FixedLenFeature(
                [1 * self.num_waypoints], self.tf.float32
            ),
        }

        # Parse TFRecord files
        for tfrecord_file in self.tfrecord_files:
            dataset = self.tf.data.TFRecordDataset(tfrecord_file)

            for serialized_example in dataset:
                example = self.tf.io.parse_single_example(serialized_example, features)

                # Convert to numpy and store
                sample = {
                    "image_encoded": example["present/image/encoded"].numpy(),
                    "sentence_embedding": example["sentence_embedding"].numpy(),
                    "present_xyz": example["present/xyz"].numpy(),
                    "present_axis_angle": example["present/axis_angle"].numpy(),
                    "present_gripper": example["present/sensed_close"].numpy(),
                    "future_xyz_residual": example["future/xyz_residual"].numpy(),
                    "future_axis_angle_residual": example["future/axis_angle_residual"].numpy(),
                    "future_target_close": example["future/target_close"].numpy(),
                }

                self.samples.append(sample)

    def __len__(self) -> int:
        """
        Get total number of samples.

        Returns:
            Number of samples in dataset
        """
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'image': Tensor (3, H, W) - preprocessed image
                - 'sentence_embedding': Tensor (512,) - task embedding
                - 'present_xyz': Tensor (3,) - current xyz position
                - 'present_axis_angle': Tensor (3,) - current orientation
                - 'present_gripper': Tensor (1,) - current gripper state
                - 'future_xyz_residual': Tensor (num_waypoints, 3) - xyz residual
                - 'future_axis_angle_residual': Tensor (num_waypoints, 3) - orientation residual
                - 'future_target_close': Tensor (num_waypoints, 1) - gripper targets

        Raises:
            IndexError: If index is out of range
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.samples)})")

        sample = self.samples[idx]

        # Decode and preprocess image
        image = self._decode_image(sample["image_encoded"])

        # Convert numpy arrays to tensors
        sentence_embedding = torch.from_numpy(sample["sentence_embedding"]).float()
        present_xyz = torch.from_numpy(sample["present_xyz"]).float()
        present_axis_angle = torch.from_numpy(sample["present_axis_angle"]).float()
        present_gripper = torch.from_numpy(sample["present_gripper"]).float()

        # Reshape future actions to (num_waypoints, dim)
        future_xyz_residual = (
            torch.from_numpy(sample["future_xyz_residual"]).float().reshape(self.num_waypoints, 3)
        )
        future_axis_angle_residual = (
            torch.from_numpy(sample["future_axis_angle_residual"])
            .float()
            .reshape(self.num_waypoints, 3)
        )
        future_target_close = (
            torch.from_numpy(sample["future_target_close"]).float().reshape(self.num_waypoints, 1)
        )

        return {
            "image": image,
            "sentence_embedding": sentence_embedding,
            "present_xyz": present_xyz,
            "present_axis_angle": present_axis_angle,
            "present_gripper": present_gripper,
            "future_xyz_residual": future_xyz_residual,
            "future_axis_angle_residual": future_axis_angle_residual,
            "future_target_close": future_target_close,
        }

    def _decode_image(
        self,
        image_encoded: bytes,
    ) -> Tensor:
        """
        Decode JPEG image and preprocess.

        Args:
            image_encoded: JPEG encoded image bytes

        Returns:
            Preprocessed image tensor (3, H, W)
        """
        # Decode JPEG
        pil_image = Image.open(io.BytesIO(image_encoded))

        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Resize
        pil_image = pil_image.resize(
            (self.image_size[1], self.image_size[0]),  # PIL uses (W, H)
            Image.BILINEAR,
        )

        # Convert to numpy array
        image_np = np.array(pil_image).astype(np.float32) / 255.0

        # Apply custom transforms if provided
        if self.transform is not None:
            image_np = self.transform(image_np)

        # Convert to tensor (H, W, 3) -> (3, H, W)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        return image_tensor
