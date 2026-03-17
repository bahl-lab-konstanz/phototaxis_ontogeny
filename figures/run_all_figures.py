"""
run_all_figures.py
==================
Runs all figure-generation scripts in the correct order.
Execute from the figures/ directory:

    cd figures
    python run_all_figures.py

Optionally skip specific figures by passing their names as arguments:

    python run_all_figures.py --skip fig3 figS5

"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Figure scripts in the intended order.
# ---------------------------------------------------------------------------
FIGURE_SCRIPTS = [
    # Graphical abstract
    "fig0_graphical_abstract.py",
    # Main figures
    "fig1_and_4.py",
    "fig1_and_4_all_agents.py",
    "fig2.py",
    "fig3_fit_models.py",
    "fig3.py",
    "fig4B.py",
    # Supplemental figures
    "figS1_wall_interactions.py",
    "figS1B_2D_SEM.py",
    "figS2A_wall_interactions.py",
    "figS3.py",
    "figS4A_offset.py",
    "figS4BC_radius.py",
    "figS4D.py",
    "figS5.py",
    "figS6.py",
    "figS7.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all figure scripts in order.")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="SCRIPT",
        help=(
            "Names of scripts to skip (with or without .py extension). "
            "Example: --skip fig3 figS5"
        ),
    )
    return parser.parse_args()


def normalise_skip(skip_list):
    """Ensure all skip entries end with .py for consistent comparison."""
    return {s if s.endswith(".py") else f"{s}.py" for s in skip_list}


def main():
    args = parse_args()
    skip = normalise_skip(args.skip)

    figures_dir = Path(__file__).parent.resolve()

    results = {}
    total = len(FIGURE_SCRIPTS)

    for i, script in enumerate(FIGURE_SCRIPTS, start=1):
        if script in skip:
            print(f"[{i}/{total}] SKIPPED  {script}")
            results[script] = "skipped"
            continue

        script_path = figures_dir / script
        if not script_path.exists():
            print(f"[{i}/{total}] MISSING  {script}  (file not found, skipping)")
            results[script] = "missing"
            continue

        print(f"[{i}/{total}] RUNNING  {script} ...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(figures_dir),
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"[{i}/{total}] OK  ({elapsed:.1f}s)\n")
            results[script] = "ok"
        else:
            print(f"[{i}/{total}] FAILED (exit code {result.returncode}, {elapsed:.1f}s)\n")
            results[script] = "failed"

    # Summary ----------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for script, status in results.items():
        icon = {"ok": "✓", "failed": "✗", "skipped": "–", "missing": "?"}.get(status, " ")
        print(f"  {icon}  {script:<45}  {status}")

    failed = [s for s, st in results.items() if st == "failed"]
    if failed:
        print(f"\n{len(failed)} script(s) failed.")
        sys.exit(1)
    else:
        print("\nAll scripts completed successfully.")


if __name__ == "__main__":
    main()

