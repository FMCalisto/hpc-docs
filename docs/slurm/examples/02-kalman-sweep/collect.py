"""Gather the sweep results into a table, a CSV file and a plot."""

import csv
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = sorted(
    (json.loads(f.read_text()) for f in results_dir.glob("sigma-*.json")),
    key=lambda r: r["sigma"],
)
if not rows:
    sys.exit(f"No results in {results_dir}")

print(f"{'sigma':>6} {'RMSE meas':>10} {'RMSE KF':>9} {'gain':>6}")
for r in rows:
    gain = r["rmse_measurement"] / r["rmse_kalman"]
    print(f"{r['sigma']:6.2f} {r['rmse_measurement']:10.3f} {r['rmse_kalman']:9.3f} {gain:5.1f}x")

with open(results_dir / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [r["sigma"] for r in rows]
    plt.plot(s, [r["rmse_measurement"] for r in rows], "o-", label="raw measurement")
    plt.plot(s, [r["rmse_kalman"] for r in rows], "o-", label="Kalman filter")
    plt.xlabel("sensor noise sigma (m)")
    plt.ylabel("position RMSE (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(results_dir / "summary.png", dpi=120, bbox_inches="tight")
    print(f"Plot: {results_dir / 'summary.png'}")
except ImportError:
    pass
print(f"CSV:  {results_dir / 'summary.csv'}")
