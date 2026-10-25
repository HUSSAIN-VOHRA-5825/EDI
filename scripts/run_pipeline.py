"""CLI Entry Point: Execute the Controlled Face Image Variation Pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/default.yaml --variations identity_test brightness_test --levels 1 2 3 4 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline import PipelineRunner


def setup_logging() -> None:
    """Configure informative console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run controlled face image variations with clinical preservation checking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--variations",
        nargs="+",
        default=None,
        help="Optional list of variation names to execute (e.g. --variations identity_test brightness_test)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of severity levels to execute (e.g. --levels 1 2 3 4 5)",
    )

    args = parser.parse_args()

    print("\n=======================================================")
    print("   Controlled Face Image Variation Tool - Pipeline")
    print("=======================================================\n")

    runner = PipelineRunner(config=args.config)
    df = runner.run(variations=args.variations, levels=args.levels)

    print("\nPipeline execution complete.")
    print(f"Total outputs recorded: {len(df)}")
    if not df.empty:
        passed_count = int(df["passed"].sum())
        print(f"Preservation Check Passed: {passed_count}/{len(df)}")
        print(f"Metadata file: {runner.metadata_file.resolve()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
