"""CLI entry point for the end-to-end PV segmentation and impact analysis pipeline."""
import argparse
import sys
from typing import Callable

from pv_segmentation.analysis import (
    mosaic,
    vectorize,
    capacity,
    yield_model,  
    economics,
    emissions,
    priority,
    monte_carlo,
)

STEPS: dict[str, Callable[[], None]] = {
    "mosaic": mosaic.main,
    "vectorize": vectorize.main,
    "capacity": capacity.main,
    "yield": yield_model.main,
    "economics": economics.main,
    "emissions": emissions.main,
    "priority": priority.main,
    "monte_carlo": monte_carlo.main,
}


def main():
    parser = argparse.ArgumentParser(description="Run the PV segmentation and impact analysis pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(STEPS.keys()),
        choices=list(STEPS.keys()),
        help="Subset or full sequence of pipeline steps to execute.",
    )
    args = parser.parse_args()

    print("=== Starting Analysis Pipeline Execution ===")
    print(f"Executing steps: {', '.join(args.steps)}\n")

    for step_name in args.steps:
        step_fn = STEPS.get(step_name)
        if not step_fn:
            print(f"[ERROR] Unknown pipeline step: '{step_name}'", file=sys.stderr)
            sys.exit(1)

        print(f"{'=' * 10} [STEP] {step_name.upper()} {'=' * 10}", flush=True)
        try:           
            step_fn()
            print(f"[SUCCESS] Step '{step_name}' completed successfully.\n")
        except Exception as e:
            print(f"[FAILED] Pipeline failed at step '{step_name}': {e}", file=sys.stderr)
            sys.exit(1)

    print("=== Pipeline Execution Complete Successfully ===")


if __name__ == "__main__":
    main()