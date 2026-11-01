"""Monte Carlo test of a constant-velocity Kalman filter at one sensor noise level.

A target moves in 2D with random accelerations. A sensor measures its position
with Gaussian noise of standard deviation `sigma`. For each run we compare the
position error of the raw measurements with that of the Kalman filter estimate.

    python kalman.py --sigma 2.0 --runs 500 --out results/sigma-2.0.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def simulate(rng, sigma, steps=400, dt=0.1, accel_std=0.5):
    # State x = [px, py, vx, vy]
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    Q = G @ G.T * accel_std**2
    R = np.eye(2) * sigma**2

    x_true = np.zeros(4)
    x_est = np.zeros(4)
    P = np.eye(4) * 10.0

    err_meas, err_kf = [], []
    for _ in range(steps):
        x_true = F @ x_true + G @ rng.normal(0, accel_std, 2)
        z = H @ x_true + rng.normal(0, sigma, 2)

        # Predict
        x_est = F @ x_est
        P = F @ P @ F.T + Q
        # Update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_est = x_est + K @ (z - H @ x_est)
        P = (np.eye(4) - K @ H) @ P

        err_meas.append(np.sum((z - x_true[:2]) ** 2))
        err_kf.append(np.sum((x_est[:2] - x_true[:2]) ** 2))

    # Skip the first 50 steps while the filter converges.
    return np.sqrt(np.mean(err_meas[50:])), np.sqrt(np.mean(err_kf[50:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--runs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    res = np.array([simulate(rng, a.sigma) for _ in range(a.runs)])
    out = {
        "sigma": a.sigma,
        "runs": a.runs,
        "rmse_measurement": float(res[:, 0].mean()),
        "rmse_kalman": float(res[:, 1].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
