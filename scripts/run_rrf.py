"""
Run E: RRF fusion of Run C (BM25F+RM3) and Run D (ColBERTv2 reranked).

Combines two ranked lists using Reciprocal Rank Fusion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RUNS_DIR, RUN_NAMES
from fuse import load_trec_run, reciprocal_rank_fusion
from retrieve import save_trec_run


def main():
    run_name = RUN_NAMES["E"]
    print(f"Run E: {run_name}")
    print("=" * 40)

    run_c_path = RUNS_DIR / f"{RUN_NAMES['C']}.txt"
    run_d_path = RUNS_DIR / f"{RUN_NAMES['D']}.txt"

    if not run_c_path.exists():
        print(f"Error: Run C not found at {run_c_path}")
        print("Run scripts/run_rm3.py first.")
        sys.exit(1)

    if not run_d_path.exists():
        print(f"Error: Run D not found at {run_d_path}")
        print("Run scripts/run_colbert.py first (requires GPU).")
        sys.exit(1)

    print(f"Loading Run C: {run_c_path}")
    run_c = load_trec_run(run_c_path)

    print(f"Loading Run D: {run_d_path}")
    run_d = load_trec_run(run_d_path)

    print("Fusing with RRF...")
    fused = reciprocal_rank_fusion([run_c, run_d])
    print(f"Fused results for {len(fused)} topics")

    output_path = RUNS_DIR / f"{run_name}.txt"
    save_trec_run(fused, output_path, run_name)


if __name__ == "__main__":
    main()
