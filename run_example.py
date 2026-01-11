"""Convenience script to run examples from project root."""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BC-Z examples")
    parser.add_argument(
        "example",
        choices=["dataloader", "dataset"],
        help="Which example to run",
    )

    args = parser.parse_args()

    if args.example == "dataloader":
        from examples.dataloader import main

        main()
    elif args.example == "dataset":
        from test_dataset_load import main

        main()
