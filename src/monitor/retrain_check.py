"""Retrain trigger: exit 1 when monitoring says the models are stale.

Reads reports/monitoring_summary.json (written by src.monitor.drift).
GitHub Actions uses the exit code to decide whether to launch a retrain.
Decision rule (deliberate, defensible): retrain on *business* signal
(simulated fill rate below floor) OR *statistical* signal (feature drift
share above threshold) — not on a blind calendar.
"""

from __future__ import annotations

import json
import sys

from src.monitor.drift import SUMMARY_FILE

if __name__ == "__main__":
    if not SUMMARY_FILE.exists():
        print("no monitoring summary found — run `make drift` first")
        sys.exit(2)
    s = json.loads(SUMMARY_FILE.read_text())
    print(f"drift_share={s['drift_share']} (threshold {s['drift_threshold']}) | "
          f"simulated_fill_rate={s['simulated_fill_rate']} (floor {s['fill_rate_floor']})")
    if s["retrain_needed"]:
        print("RETRAIN NEEDED")
        sys.exit(1)
    print("models healthy — no retrain")
