"""Add a truthful partial-run record to the cumulative training ledger."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs" / "reports" / "training_ledger.json"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--base-weights", required=True)
    args = parser.parse_args()

    results = args.results if args.results.is_absolute() else ROOT / args.results
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    with results.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No completed epochs found in {results}")
    last = rows[-1]
    ledger_data = json.loads(LEDGER.read_text(encoding="utf-8"))
    entries = ledger_data.get("entries", [])
    if any(item.get("run_name") == args.run_name for item in entries):
        raise SystemExit(f"Run already recorded: {args.run_name}")
    entries.append({
        "run_name": args.run_name,
        "status": "interrupted",
        "mode": "full",
        "epochs": 0,
        "completed_epochs": len(rows),
        "base_weights": args.base_weights,
        "batch_used": 8,
        "started_at": args.started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "last_completed_epoch": int(float(last["epoch"])) if last.get("epoch") else len(rows),
        "mAP50": float(last["metrics/mAP50(B)"]),
        "mAP50_95": float(last["metrics/mAP50-95(B)"]),
        "checkpoint": rel(checkpoint),
        "results": rel(results),
        "note": "Interrupted by user after this many completed epochs; resumed from saved checkpoint.",
    })
    ledger_data["entries"] = entries
    ledger_data["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    LEDGER.write_text(json.dumps(ledger_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(rows)} completed epochs for {args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
