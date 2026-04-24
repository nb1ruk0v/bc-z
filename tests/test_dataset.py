"""Unit tests for BCZDataset."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BCZDataset

TEST_DATA_PATH = Path(__file__).parent / "fixtures" / "test_data.tfrecord"


@pytest.fixture(scope="module")
def test_data_path():
    if not TEST_DATA_PATH.exists():
        pytest.fail(
            f"Test data not found at {TEST_DATA_PATH}. "
            f"Run: python tests/fixtures/create_test_data.py"
        )
    return TEST_DATA_PATH


@pytest.fixture
def dataset(test_data_path):
    return BCZDataset(
        data_path=test_data_path,
        image_size=(100, 100),
        mode="train",
        num_waypoints=10,
    )


# Expected per-sample shapes/dtypes. Keyed for spec-driven shape tests.
_EXPECTED = {
    "image": ((3, 100, 100), torch.float32),
    "sentence_embedding": ((512,), torch.float32),
    "present_xyz": ((3,), torch.float32),
    "present_axis_angle": ((3,), torch.float32),
    "present_gripper": ((1,), torch.float32),
    "future_xyz_residual": ((10, 3), torch.float32),
    "future_axis_angle_residual": ((10, 3), torch.float32),
    "future_target_close": ((10, 1), torch.float32),
}


# ---------- Initialization errors ----------


def test_invalid_path_raises():
    with pytest.raises(FileNotFoundError, match="Data path not found"):
        BCZDataset(data_path=Path("/nonexistent/path"))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No TFRecord files found"):
        BCZDataset(data_path=tmp_path)


# ---------- Sample contract ----------


def test_sample_has_expected_shapes_and_dtypes(dataset):
    sample = dataset[0]
    assert set(sample.keys()) == set(_EXPECTED.keys())
    for key, (shape, dtype) in _EXPECTED.items():
        assert sample[key].shape == shape, f"{key}: {sample[key].shape} != {shape}"
        assert sample[key].dtype == dtype, f"{key}: {sample[key].dtype} != {dtype}"


def test_image_values_normalized_to_unit_range(dataset):
    # Catches regression if /255 normalization is skipped.
    sample = dataset[0]
    assert 0.0 <= sample["image"].min() <= sample["image"].max() <= 1.0


# ---------- Indexing ----------


def test_out_of_range_index_raises(dataset):
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[-1]
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[len(dataset)]


# ---------- Image size ----------


@pytest.mark.parametrize("size", [(50, 50), (224, 224)])
def test_custom_image_size_is_respected(test_data_path, size):
    ds = BCZDataset(data_path=test_data_path, image_size=size)
    assert ds[0]["image"].shape == (3, *size)


# ---------- Transform ----------


def test_transform_is_actually_applied(test_data_path):
    # Constant-output transform: if it wasn't called, pixel values won't all be 0.42.
    ds = BCZDataset(
        data_path=test_data_path,
        image_size=(100, 100),
        transform=lambda img: img * 0.0 + 0.42,
    )
    img = ds[0]["image"]
    assert torch.allclose(img, torch.full_like(img, 0.42))


# ---------- DataLoader integration ----------


def test_dataloader_batches_samples(dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    for key, (shape, _) in _EXPECTED.items():
        assert batch[key].shape == (2, *shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
