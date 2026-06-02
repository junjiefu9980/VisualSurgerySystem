"""
Run final evaluation scripts.
"""
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# main
# ================================================
def main():
    from src.pipeline import evaluate_kalman
    from src.pipeline import build_final_summary

    # Step 1: evaluate kalman
    print("\n", "=" * 20, "[1. Evaluate Kalman]", "=" * 20)
    evaluate_kalman.main()

    # Step 2: final summary
    print("\n", "=" * 20, "[2. Final Summary]", "=" * 20)
    build_final_summary.main()


if __name__ == "__main__":
    main()
