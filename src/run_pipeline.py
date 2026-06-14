"""
Main entry point — runs all pipeline stages sequentially.

Usage:
    python run_pipeline.py [--stages 1 2 3 4] [--skip-stage3] [--baseline]

    --stages      Which stages to run (default: all)
    --skip-stage3 Skip LLM report generation (avoids API cost)
    --baseline    Use Grounding DINO zero-shot instead of YOLOv11 (for comparison)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))


def main():
    parser = argparse.ArgumentParser(description="OPG Analysis Pipeline — YOLOv11 + LLM")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4])
    parser.add_argument("--skip-stage3", action="store_true",
                        help="Skip Stage 3 (report generation via API)")
    parser.add_argument("--baseline", action="store_true",
                        help="Use Grounding DINO zero-shot (baseline) instead of YOLOv11")
    args = parser.parse_args()

    if args.skip_stage3 and 3 in args.stages:
        args.stages.remove(3)

    s1_label = "DETECTION (Grounding DINO zero-shot)" if args.baseline else "DETECTION (YOLOv11 fine-tuned)"
    print("=" * 60)
    print("  OPG ANALYSIS PIPELINE")
    print(f"  Stage 1: {s1_label}")
    print(f"  Stages:  {args.stages}")
    print("=" * 60)

    s1_results = None

    if 1 in args.stages:
        print(f"\n[1/4] {s1_label}")
        t = time.time()
        if args.baseline:
            from stage1_detection import run as run_s1
        else:
            from stage1_yolo import run as run_s1
        s1_results = run_s1()
        print(f"      Time: {time.time()-t:.0f}s")

    if 2 in args.stages:
        print("\n[2/4] SPONTANEOUS RECALL EVALUATION (Stage 2 — unbiased)")
        t = time.time()
        from stage2_validation import run as run_s2
        run_s2(s1_results)
        print(f"      Time: {time.time()-t:.0f}s")

    if 3 in args.stages:
        print("\n[3/4] PRE-REPORT GENERATION (LLM — no data leakage)")
        t = time.time()
        from stage3_report import run as run_s3
        run_s3(s1_results)
        print(f"      Time: {time.time()-t:.0f}s")

    if 4 in args.stages:
        print("\n[4/4] METRICS & FIGURES")
        t = time.time()
        from stage4_metrics import run as run_s4
        run_s4()
        print(f"      Time: {time.time()-t:.0f}s")

    print("\n✓ Pipeline complete. Results in: results/")


if __name__ == "__main__":
    main()
