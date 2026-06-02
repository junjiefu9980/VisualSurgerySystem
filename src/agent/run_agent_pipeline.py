"""
Run agent pipeline.
choose cases, draw figures, and write feedback
"""
import sys
import shutil
from pathlib import Path


# ===============================================
# path
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_AGENT_DIR = REPO_ROOT / "output" / "agent"

# run config
AUTO_SELECT = True
CLEAN_OLD = True

CASE_ID = ""
N_PER_BUCKET = 1

FPS = 30
K_MAD = 6.0
K_BREAK = 3.0
K_GATE = 6.0
TRACK_WINDOW = 30
MIN_LEN = 5
KPT_ID = "auto"
VIEW_SIDE = "both"


# ===============================================
# 1. Clean old output
# ================================================
def clean_output(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return

    for name in [
        ".DS_Store",
        "evidence_pack.json",
        "summary.csv",
        "feedback.md",
        "rule_feedback.md",
        "feedback_prompt.txt",
        "evidence_chain.csv",
    ]:
        p = out_dir / name
        if p.exists():
            p.unlink()

    fig_dir = out_dir / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)


# 2. Run agent
# ================================================
def main():
    from src.agent import build_evidence_pack
    from src.agent import build_feedback

    if CLEAN_OLD:
        print("\n", "=" * 20, "[0. Clean old agent output]", "=" * 20)
        clean_output(OUT_AGENT_DIR)
        print("[DONE] [0. Clean old agent output]")

    print("\n", "=" * 20, "[1. Build evidence pack]", "=" * 20)
    build_evidence_pack.run(
        case_id=CASE_ID,
        auto_select=AUTO_SELECT,
        n_per_bucket=N_PER_BUCKET,
        fps=FPS,
        k_mad=K_MAD,
        k_break=K_BREAK,
        k_gate=K_GATE,
        track_window=TRACK_WINDOW,
        continuity=True,
        min_len=MIN_LEN,
        out_dir=OUT_AGENT_DIR,
        kpt_id=KPT_ID,
        view_side=VIEW_SIDE,
    )
    print("[DONE] [1. Build evidence pack]")

    print("\n", "=" * 20, "[2. Build feedback]", "=" * 20)
    build_feedback.run()
    print("[DONE] [2. Build feedback]")


if __name__ == "__main__":
    main()
