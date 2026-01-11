"""Unit tests for BCZDataset."""

from pathlib import Path

import pytest
import torch

from src.data.dataset import BCZDataset

# Path to test fixtures
TEST_DATA_PATH = Path(__file__).parent / "fixtures" / "test_data.tfrecord"


@pytest.fixture
def test_data_path():
    """Get path to test TFRecord file."""
    if not TEST_DATA_PATH.exists():
        pytest.fail(
            f"Test data not found at {TEST_DATA_PATH}. "
            f"Run: python tests/fixtures/create_test_data.py"
        )
    return TEST_DATA_PATH


class TestBCZDatasetInitialization:
    """Test BCZDataset initialization and validation."""

    def test_dataset_with_valid_path(self, test_data_path):
        """Test dataset initializes with valid data path."""
        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            num_waypoints=10,
        )

        assert len(dataset) == 5  # We created 5 test samples
        assert dataset.image_size == (100, 100)
        assert dataset.num_waypoints == 10
        assert dataset.mode == "train"

    def test_dataset_with_invalid_path(self):
        """Test dataset raises FileNotFoundError with invalid path."""
        with pytest.raises(FileNotFoundError, match="Data path not found"):
            BCZDataset(
                data_path=Path("/nonexistent/path"),
                image_size=(100, 100),
                mode="train",
            )

    def test_dataset_with_no_tfrecord_files(self, tmp_path):
        """Test dataset raises ValueError when no TFRecord files found."""
        with pytest.raises(ValueError, match="No TFRecord files found"):
            BCZDataset(
                data_path=tmp_path,
                image_size=(100, 100),
                mode="train",
            )


class TestBCZDatasetSampleStructure:
    """Test structure and types of dataset samples."""

    @pytest.fixture
    def dataset(self, test_data_path):
        """Create dataset fixture."""
        return BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            num_waypoints=10,
        )

    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) == 5

    def test_sample_has_all_keys(self, dataset):
        """Test sample contains all required keys."""
        sample = dataset[0]

        expected_keys = {
            "image",
            "sentence_embedding",
            "present_xyz",
            "present_axis_angle",
            "present_gripper",
            "future_xyz_residual",
            "future_axis_angle_residual",
            "future_target_close",
        }

        assert set(sample.keys()) == expected_keys

    def test_sample_image_shape(self, dataset):
        """Test image has correct shape."""
        sample = dataset[0]
        assert sample["image"].shape == (3, 100, 100)
        assert sample["image"].dtype == torch.float32

    def test_sample_image_range(self, dataset):
        """Test image values are in [0, 1] range."""
        sample = dataset[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_sample_sentence_embedding_shape(self, dataset):
        """Test sentence embedding has correct shape."""
        sample = dataset[0]
        assert sample["sentence_embedding"].shape == (512,)
        assert sample["sentence_embedding"].dtype == torch.float32

    def test_sample_present_xyz_shape(self, dataset):
        """Test present xyz has correct shape."""
        sample = dataset[0]
        assert sample["present_xyz"].shape == (3,)
        assert sample["present_xyz"].dtype == torch.float32

    def test_sample_present_axis_angle_shape(self, dataset):
        """Test present axis angle has correct shape."""
        sample = dataset[0]
        assert sample["present_axis_angle"].shape == (3,)
        assert sample["present_axis_angle"].dtype == torch.float32

    def test_sample_present_gripper_shape(self, dataset):
        """Test present gripper has correct shape."""
        sample = dataset[0]
        assert sample["present_gripper"].shape == (1,)
        assert sample["present_gripper"].dtype == torch.float32

    def test_sample_future_xyz_residual_shape(self, dataset):
        """Test future xyz residual has correct shape."""
        sample = dataset[0]
        assert sample["future_xyz_residual"].shape == (10, 3)
        assert sample["future_xyz_residual"].dtype == torch.float32

    def test_sample_future_axis_angle_residual_shape(self, dataset):
        """Test future axis angle residual has correct shape."""
        sample = dataset[0]
        assert sample["future_axis_angle_residual"].shape == (10, 3)
        assert sample["future_axis_angle_residual"].dtype == torch.float32

    def test_sample_future_target_close_shape(self, dataset):
        """Test future target close has correct shape."""
        sample = dataset[0]
        assert sample["future_target_close"].shape == (10, 1)
        assert sample["future_target_close"].dtype == torch.float32

    def test_all_samples_have_same_structure(self, dataset):
        """Test all samples have consistent structure."""
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["image"].shape == (3, 100, 100)
            assert sample["sentence_embedding"].shape == (512,)
            assert sample["future_xyz_residual"].shape == (10, 3)


class TestBCZDatasetIndexing:
    """Test dataset indexing behavior."""

    @pytest.fixture
    def dataset(self, test_data_path):
        """Create dataset fixture."""
        return BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            num_waypoints=10,
        )

    def test_valid_indices(self, dataset):
        """Test accessing all valid indices."""
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            assert "image" in sample

    def test_negative_index(self, dataset):
        """Test negative indices raise IndexError."""
        with pytest.raises(IndexError, match="Index .* out of range"):
            _ = dataset[-1]

    def test_out_of_bounds_index(self, dataset):
        """Test out of bounds indices raise IndexError."""
        with pytest.raises(IndexError, match="Index .* out of range"):
            _ = dataset[len(dataset)]

        with pytest.raises(IndexError, match="Index .* out of range"):
            _ = dataset[100]

    def test_multiple_accesses_same_index(self, dataset):
        """Test multiple accesses to same index return consistent data."""
        sample_1 = dataset[0]
        sample_2 = dataset[0]

        # Should return same data
        assert torch.allclose(sample_1["image"], sample_2["image"])
        assert torch.allclose(
            sample_1["sentence_embedding"],
            sample_2["sentence_embedding"],
        )
        assert torch.allclose(sample_1["present_xyz"], sample_2["present_xyz"])


class TestBCZDatasetImageSize:
    """Test dataset with different image sizes."""

    def test_custom_image_size_50x50(self, test_data_path):
        """Test dataset with 50x50 image size."""
        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(50, 50),
            mode="train",
            num_waypoints=10,
        )

        sample = dataset[0]
        assert sample["image"].shape == (3, 50, 50)

    def test_custom_image_size_224x224(self, test_data_path):
        """Test dataset with 224x224 image size."""
        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(224, 224),
            mode="train",
            num_waypoints=10,
        )

        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)

    def test_different_image_sizes_dont_affect_other_fields(self, test_data_path):
        """Test that changing image size doesn't affect other tensor shapes."""
        dataset_small = BCZDataset(
            data_path=test_data_path,
            image_size=(50, 50),
        )
        dataset_large = BCZDataset(
            data_path=test_data_path,
            image_size=(200, 200),
        )

        sample_small = dataset_small[0]
        sample_large = dataset_large[0]

        # Images are different sizes
        assert sample_small["image"].shape != sample_large["image"].shape

        # But other fields are the same
        assert sample_small["present_xyz"].shape == sample_large["present_xyz"].shape
        assert sample_small["sentence_embedding"].shape == sample_large["sentence_embedding"].shape


class TestBCZDatasetWaypoints:
    """Test dataset with different numbers of waypoints.

    Note: TFRecord files have fixed-size features, so num_waypoints must match
    the value used when creating the TFRecord. Our test data was created with
    num_waypoints=10, so we can only test that value.
    """

    def test_default_waypoints(self, test_data_path):
        """Test dataset with default 10 waypoints (matches test data)."""
        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            num_waypoints=10,
        )

        sample = dataset[0]
        assert sample["future_xyz_residual"].shape == (10, 3)
        assert sample["future_axis_angle_residual"].shape == (10, 3)
        assert sample["future_target_close"].shape == (10, 1)

    def test_waypoints_consistent_across_samples(self, test_data_path):
        """Test that all samples have same waypoint dimension."""
        dataset = BCZDataset(
            data_path=test_data_path,
            num_waypoints=10,
        )

        # Check all samples have consistent waypoint shape
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["future_xyz_residual"].shape[0] == 10
            assert sample["future_axis_angle_residual"].shape[0] == 10
            assert sample["future_target_close"].shape[0] == 10


class TestBCZDatasetTransforms:
    """Test dataset with custom transforms."""

    def test_custom_transform_applied(self, test_data_path):
        """Test custom transform is applied to images."""

        def invert_transform(image):
            """Simple transform: invert colors."""
            return 1.0 - image

        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            transform=invert_transform,
        )

        sample = dataset[0]

        # After inversion, values should still be in [0, 1]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_no_transform(self, test_data_path):
        """Test dataset works without transforms."""
        dataset = BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            transform=None,
        )

        sample = dataset[0]
        assert sample["image"] is not None
        assert sample["image"].shape == (3, 100, 100)


class TestBCZDatasetIntegration:
    """Integration tests with PyTorch DataLoader."""

    @pytest.fixture
    def dataset(self, test_data_path):
        """Create dataset fixture."""
        return BCZDataset(
            data_path=test_data_path,
            image_size=(100, 100),
            mode="train",
            num_waypoints=10,
        )

    def test_dataloader_batch_shapes(self, dataset):
        """Test dataset works with DataLoader and produces correct batch shapes."""
        from torch.utils.data import DataLoader

        batch_size = 2
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Get first batch
        batch = next(iter(dataloader))

        assert batch["image"].shape == (batch_size, 3, 100, 100)
        assert batch["sentence_embedding"].shape == (batch_size, 512)
        assert batch["present_xyz"].shape == (batch_size, 3)
        assert batch["future_xyz_residual"].shape == (batch_size, 10, 3)

    def test_dataloader_iterates_all_samples(self, dataset):
        """Test DataLoader iterates through all samples."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        total_samples = 0
        for batch in dataloader:
            total_samples += batch["image"].shape[0]

        assert total_samples == len(dataset)

    def test_dataloader_with_shuffle(self, dataset):
        """Test dataset works with shuffled DataLoader."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        # Should be able to iterate
        batch = next(iter(dataloader))
        assert batch["image"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
