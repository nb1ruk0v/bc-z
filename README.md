# BC-Z Implementation

Educational project for improving Claude Code skills and implementing BC-Z (Behavioral Cloning with Zero-shot task generalization) for robotics.

**Status:** Week 3 Complete ✅ - Data Pipeline + Model + Training Loop (trackio, cosine LR) implemented

## About the project

BC-Z is an imitation learning system for robotics with zero-shot generalization to new tasks. The project is based on Google Research's work on multi-task imitation learning.

**Key features:**
- Language/video-conditioned policies with pretrained embeddings
- ResNet18/50 + FiLM architecture for task-conditioned control
- Multi-waypoint trajectory prediction (10 waypoints)
- Orientation representation via axis-angle
- Residual actions relative to the current state

## Project structure

```
bc-z/
├── src/                          # Main implementation code
│   ├── data/
│   │   └── dataset.py           # ✅ BCZDataset for TFRecord data
│   ├── models/                   # ✅ ResNet + FiLM + BCZPolicy
│   │   ├── film.py              # ✅ FiLMLayer (task conditioning)
│   │   ├── backbone.py          # ✅ ResNetBackbone (resnet18/50)
│   │   └── policy.py            # ✅ BCZPolicy
│   ├── training/                 # 🔜 Training loop and trainer
│   └── utils/
│       └── metrics.py           # ✅ Evaluation metrics (13 tests)
├── tests/
│   ├── fixtures/
│   │   ├── test_data.tfrecord   # Test data (5 samples, 154 KB)
│   │   └── create_test_data.py  # Script for creating test data
│   ├── test_dataset.py          # ✅ 29 tests for BCZDataset
│   ├── test_metrics.py          # ✅ 13 tests for metrics
│   └── test_models.py           # ✅ 13 tests for models
├── examples/
│   └── dataloader.py            # ✅ Dataset usage example
├── docs/
│   └── dataset_usage.md         # Dataset documentation
├── data/                         # Real TFRecord data (321,970 samples)
├── bc-z/                         # Original Google implementation
├── .plan.md                      # Detailed implementation plan
└── CLAUDE.md                     # Instructions for Claude Code

✅ = Done    🔜 = Planned
```

## Installation

The project uses **uv** for dependency management (Python 3.12+):

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Main dependencies:**
- PyTorch - for neural networks
- TensorFlow - for reading TFRecord files
- Pillow - for image processing
- pytest - for testing
- ruff - for linting and formatting

## Dataset usage

### Basic example

```python
from pathlib import Path
from src.data.dataset import BCZDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = BCZDataset(
    data_path=Path("data/bcz-21task_v9.0.1.tfrecord"),
    image_size=(100, 100),
    mode="train",
    num_waypoints=10,
)

# Get a single sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")              # (3, 100, 100)
print(f"Embedding: {sample['sentence_embedding'].shape}")   # (512,)
print(f"Position: {sample['present_xyz'].shape}")           # (3,)
print(f"Actions: {sample['future_xyz_residual'].shape}")    # (10, 3)

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for batch in dataloader:
    images = batch["image"]              # (32, 3, 100, 100)
    embeddings = batch["sentence_embedding"]  # (32, 512)
    # ... training code
```

### Data structure

Each sample contains:

| Key | Shape | Description |
|------|-------|----------|
| `image` | (3, H, W) | RGB image (normalized to [0, 1]) |
| `sentence_embedding` | (512,) | Task embedding from a pretrained model |
| `present_xyz` | (3,) | Current end-effector position (x, y, z) |
| `present_axis_angle` | (3,) | Current orientation (axis-angle) |
| `present_gripper` | (1,) | Current gripper state [0=open, 1=closed] |
| `future_xyz_residual` | (10, 3) | Residual actions for position |
| `future_axis_angle_residual` | (10, 3) | Residual actions for orientation |
| `future_target_close` | (10, 1) | Target gripper states |

**More examples:** see `examples/dataloader.py` and `docs/dataset_usage.md`

## Testing

Run all tests:

```bash
# All tests
uv run pytest tests/ -v

# Only dataset tests
uv run pytest tests/test_dataset.py -v

# Only metrics tests
uv run pytest tests/test_metrics.py -v
```

**Current status:** 55/55 tests passing ✅
- Dataset: 29 tests
- Metrics: 13 tests
- Models: 13 tests

Tests use self-contained fixture data in `tests/fixtures/` and don't depend on external large datasets.

## Linting and formatting

```bash
# Automatic formatting
uv run ruff format .

# Check and auto-fix
uv run ruff check --fix .

# Run all checks
uv run ruff format . && uv run ruff check --fix . && uv run pytest tests/
```

## Development

### Code style

- **Formatter:** ruff (line length: 100)
- **Type hints:** required for all functions
- **Docstrings:** Google-style for all public functions
- **Parameters:** each on its own line (including `self`)

```python
def example_function(
    self,
    param1: str,
    param2: int = 10,
) -> dict[str, Tensor]:
    """
    Short description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description
    """
```

### Workflow

The project follows a two-phase workflow:

1. **Design** - Research and planning
2. **Build** - Implementation with commits and testing

**Recommended patterns:**
- Explore → Plan → Code → Commit
- Write tests, commit; code, iterate, commit
- Write code, screenshot result, iterate

See the sections below for details.

## Current progress

**Week 1: Foundation + Data Pipeline** ✅ COMPLETED
- [x] Project structure
- [x] Dependency setup (uv, tensorflow)
- [x] Linting (ruff) and pre-commit hooks
- [x] BCZDataset for TFRecord format
- [x] Evaluation metrics
- [x] 42 unit tests (all passing)
- [x] Documentation and examples

**Week 2: Model Implementation** ✅ COMPLETED
- [x] ResNet18/50 backbone (`src/models/backbone.py`)
- [x] FiLM layers for task conditioning (`src/models/film.py`)
- [x] BCZPolicy: stem → 4×(stage+FiLM) → GAP → concat(state) → MLP head
- [x] Single MLP head (TODO: split into 3 heads)
- [x] 13 model tests (shape, affine-identity, gradient flow, batch independence)

**Week 3: Training Loop** ✅ COMPLETED
- [x] Loss functions (Huber + BCE, BC-Z Appendix D weights)
- [x] Trainer class with Gaussian embedding noise
- [x] Optimizer (Adam) and LR scheduler (cosine / step, configurable)
- [x] Experiment tracking via trackio
- [x] YAML-configured entrypoint (`main.py`) with smoke config

See `.plan.md` for the detailed plan.

## Documentation

- **`.plan.md`** - Detailed 9-phase implementation plan
- **`CLAUDE.md`** - Instructions for development and AI assistant
- **`docs/dataset_usage.md`** - Detailed dataset guide
- **`bc-z/`** - Original Google Research implementation

## Try common workflows
Claude Code doesn’t impose a specific workflow, giving you the flexibility to use it how you want. Within the space this flexibility affords, several successful patterns for effectively using Claude Code have emerged across our community of users:

a. Explore, plan, code, commit
This versatile workflow suits many problems:

Ask Claude to read relevant files, images, or URLs, providing either general pointers ("read the file that handles logging") or specific filenames ("read logging.py"), but explicitly tell it not to write any code just yet.
This is the part of the workflow where you should consider strong use of subagents, especially for complex problems. Telling Claude to use subagents to verify details or investigate particular questions it might have, especially early on in a conversation or task, tends to preserve context availability without much downside in terms of lost efficiency.
Ask Claude to make a plan for how to approach a specific problem. We recommend using the word "think" to trigger extended thinking mode, which gives Claude additional computation time to evaluate alternatives more thoroughly. These specific phrases are mapped directly to increasing levels of thinking budget in the system: "think" < "think hard" < "think harder" < "ultrathink." Each level allocates progressively more thinking budget for Claude to use.
If the results of this step seem reasonable, you can have Claude create a document or a GitHub issue with its plan so that you can reset to this spot if the implementation (step 3) isn’t what you want.
Ask Claude to implement its solution in code. This is also a good place to ask it to explicitly verify the reasonableness of its solution as it implements pieces of the solution.
Ask Claude to commit the result and create a pull request. If relevant, this is also a good time to have Claude update any READMEs or changelogs with an explanation of what it just did.
Steps #1-#2 are crucial—without them, Claude tends to jump straight to coding a solution. While sometimes that's what you want, asking Claude to research and plan first significantly improves performance for problems requiring deeper thinking upfront.

b. Write tests, commit; code, iterate, commit
This is an Anthropic-favorite workflow for changes that are easily verifiable with unit, integration, or end-to-end tests. Test-driven development (TDD) becomes even more powerful with agentic coding:

Ask Claude to write tests based on expected input/output pairs. Be explicit about the fact that you’re doing test-driven development so that it avoids creating mock implementations, even for functionality that doesn’t exist yet in the codebase.
Tell Claude to run the tests and confirm they fail. Explicitly telling it not to write any implementation code at this stage is often helpful.
Ask Claude to commit the tests when you’re satisfied with them.
Ask Claude to write code that passes the tests, instructing it not to modify the tests. Tell Claude to keep going until all tests pass. It will usually take a few iterations for Claude to write code, run the tests, adjust the code, and run the tests again.
At this stage, it can help to ask it to verify with independent subagents that the implementation isn’t overfitting to the tests
Ask Claude to commit the code once you’re satisfied with the changes.
Claude performs best when it has a clear target to iterate against—a visual mock, a test case, or another kind of output. By providing expected outputs like tests, Claude can make changes, evaluate results, and incrementally improve until it succeeds.

c. Write code, screenshot result, iterate
Similar to the testing workflow, you can provide Claude with visual targets:

Give Claude a way to take browser screenshots (e.g., with the Puppeteer MCP server, an iOS simulator MCP server, or manually copy / paste screenshots into Claude).
Give Claude a visual mock by copying / pasting or drag-dropping an image, or giving Claude the image file path.
Ask Claude to implement the design in code, take screenshots of the result, and iterate until its result matches the mock.
Ask Claude to commit when you're satisfied.
Like humans, Claude's outputs tend to improve significantly with iteration. While the first version might be good, after 2-3 iterations it will typically look much better. Give Claude the tools to see its outputs for best results.
