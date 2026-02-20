"""
scan_and_push.py — Run the daily scan pipeline locally and push results to GitHub.

Usage:
    python scan_and_push.py

Steps:
    1. Download latest price bars (incremental)
    2. Rebuild feature parquets
    3. Run signal scanner
    4. Build trade cards
    5. Commit + push results to GitHub (Streamlit Cloud auto-updates)
"""

import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT    = pathlib.Path(__file__).resolve().parent
PYTHON  = sys.executable

# Files to commit after scan
SCAN_OUTPUTS = [
    "data/processed/latest_scan.parquet",
    "data/processed/trade_cards.parquet",
]


def run(label: str, module: str) -> None:
    """Run a pipeline module, printing progress. Exits on failure."""
    print(f"\n{'-'*50}")
    print(f"  {label}")
    print(f"{'-'*50}")
    result = subprocess.run(
        [PYTHON, "-m", module],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\nFAILED  {label} failed. Aborting.")
        sys.exit(1)
    print(f"OK  {label} done.")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)


def push_results() -> None:
    """Stage scan outputs + signalled ticker parquets, commit, and push."""
    print(f"\n{'-'*50}")
    print("  Pushing results to GitHub")
    print(f"{'-'*50}")

    # Stage scan output files
    for f in SCAN_OUTPUTS:
        git("add", f)

    # Force-add per-ticker parquets for signalled tickers (Stock Detail page)
    try:
        import pandas as pd
        cards_path = ROOT / "data" / "processed" / "trade_cards.parquet"
        if cards_path.exists():
            tickers = pd.read_parquet(cards_path)["ticker"].tolist()
            for t in tickers:
                p = ROOT / "data" / "processed" / "equities" / f"{t}.parquet"
                if p.exists():
                    git("add", "-f", str(p))
            print(f"  Staged parquets for signalled tickers: {tickers}")
    except Exception as e:
        print(f"  Warning: could not stage ticker parquets: {e}")

    # Check if anything actually changed
    status = git("diff", "--staged", "--quiet")
    if status.returncode == 0:
        print("  No changes — scan results unchanged since last push.")
        return

    # Commit
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result = git("commit", "-m", f"scan: {ts}")
    if result.returncode != 0:
        print(f"  Git commit failed:\n{result.stderr}")
        sys.exit(1)

    # Push
    result = git("push")
    if result.returncode != 0:
        print(f"  Git push failed:\n{result.stderr}")
        sys.exit(1)

    print("OK  Pushed — Streamlit Cloud will update shortly.")


def main() -> None:
    start = datetime.now()
    print(f"\nML Scanner — Daily Scan & Push")
    print(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    run("Downloading latest price data",  "ingestion.downloader")
    run("Building feature parquets",      "features.pipeline")
    run("Running signal scanner",         "signals.scanner")
    run("Building trade cards",           "signals.trade_setup")
    push_results()

    elapsed = (datetime.now() - start).seconds
    print(f"\nDone in {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
