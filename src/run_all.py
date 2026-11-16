"""
run_all.py -- reproduce every result in this repository, in order.

    python src/run_all.py            full reproduction (needs a GPU, ~95 min)
    python src/run_all.py --quick    correctness checks only (~5 min, CPU is fine)
    python src/run_all.py --list     show the steps without running anything

Each experiment runs as its own process, so one failing step cannot leave
stale state behind for the next. Results are written to <repo>/results by
default; set UMLCAM_OUT to send them elsewhere.

A note on hardware: the full run is roughly 25,000 forward passes through
ResNet-50. That is about 95 minutes on a Tesla T4, but CPU inference is
20-50x slower, so without CUDA a full run takes many hours. Use --quick on a
machine without a GPU; it still demonstrates that the implementation is
correct and that the pipeline measures something real.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script, what it does, rough minutes on a T4)
STEPS = [
    ("make_environment_record.py", "Record the environment (versions, GPU)", 1),
    ("run_validation.py", "Correctness gate: Grad-CAM vs library, Deletion vs random", 3),
    ("run_benchmark.py", "Main benchmark, bootstrap CIs, Holm tests, sweeps", 45),
    ("run_localization.py", "Pointing Game / IoU / EBPG + qualitative grid", 10),
    ("run_vgg16.py", "Second architecture (VGG-16)", 10),
    ("run_uncertainty.py", "TTA uncertainty, sparsification curves, H2", 25),
    ("run_sanity.py", "Cascading randomization sanity check", 2),
]

QUICK = {"make_environment_record.py", "run_validation.py", "run_sanity.py"}


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="run only the fast correctness checks (no GPU needed)")
    ap.add_argument("--list", action="store_true",
                    help="list the steps and exit")
    args = ap.parse_args()

    steps = [s for s in STEPS if s[0] in QUICK] if args.quick else STEPS

    if args.list:
        total = sum(m for _, _, m in steps)
        for script, what, minutes in steps:
            print(f"  {script:30s} {what}  (~{minutes} min)")
        print(f"\n  estimated total: ~{total} min on a GPU")
        return 0

    if args.quick:
        banner("QUICK MODE - correctness checks only (no GPU required)")
    else:
        banner("FULL REPRODUCTION - a GPU is strongly recommended (~95 min)")

    results = []
    started = time.time()
    for script, what, _ in steps:
        banner(f"{script}  --  {what}")
        t0 = time.time()
        outcome = subprocess.run([sys.executable, str(HERE / script)])
        elapsed = (time.time() - t0) / 60
        ok = outcome.returncode == 0
        results.append((script, ok, elapsed))
        if not ok:
            print(f"\n*** {script} FAILED (exit code {outcome.returncode}). "
                  f"Stopping here so the error is not buried. ***")
            break

    banner("SUMMARY")
    for script, ok, elapsed in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {script:30s} {elapsed:5.1f} min")
    print(f"\n  total: {(time.time() - started) / 60:.1f} min")

    failed = [s for s, ok, _ in results if not ok]
    if failed:
        print(f"\n  {len(failed)} step(s) failed - results are incomplete.")
        return 1
    if args.quick:
        print("\n  Quick checks passed. Run without --quick (on a GPU) for the "
              "full benchmark.")
    else:
        print("\n  All experiments completed. See results/ for the outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
