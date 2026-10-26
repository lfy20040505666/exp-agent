"""Generate the synthetic training runs in examples/runs.

The numbers are made up (seeded random curves shaped like 3D Gaussian Splatting
training logs). They exist so the agent, tests and evals have something
realistic to read; they are NOT real experiment results.

    python examples/make_sample_runs.py
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

OUT = Path(__file__).parent / "runs"
LOG_EVERY = 500

RUNS = [
    # name, scene, iterations, overrides, final eval (None = crashed), extra
    ("lego_baseline", "lego", 30000, {}, {"psnr": 35.12, "ssim": 0.9812, "lpips": 0.0271}, {}),
    ("lego_low_thresh", "lego", 30000, {"densify_grad_threshold": 0.0001},
     {"psnr": 35.47, "ssim": 0.9829, "lpips": 0.0238}, {"gauss_scale": 1.9}),
    ("lego_high_lr", "lego", 30000, {"position_lr_init": 0.00064}, None, {"diverge_at": 6500}),
    ("room_baseline", "room", 30000, {}, {"psnr": 31.36, "ssim": 0.9174, "lpips": 0.2191}, {}),
    ("room_short", "room", 7000, {}, {"psnr": 29.81, "ssim": 0.8902, "lpips": 0.2733}, {}),
]

BASE_CONFIG = {
    "position_lr_init": 0.00016,
    "densify_grad_threshold": 0.0002,
    "densify_until_iter": 15000,
    "sh_degree": 3,
}


def make_metrics(rng: random.Random, iterations: int, extra: dict) -> list[dict]:
    rows = []
    gaussians = 100_000
    scale = extra.get("gauss_scale", 1.0)
    diverge_at = extra.get("diverge_at")
    for step in range(LOG_EVERY, iterations + 1, LOG_EVERY):
        loss = 0.02 + 0.15 * math.exp(-step / 3000) + rng.gauss(0, 0.002)
        psnr = 20 + 15 * (1 - math.exp(-step / 4000)) + rng.gauss(0, 0.2)
        if step <= BASE_CONFIG["densify_until_iter"]:
            gaussians = int(100_000 + scale * 180_000 * (1 - math.exp(-step / 5000)))
        if diverge_at is not None and step >= diverge_at:
            # Loss spikes once, then goes NaN and training stops two logs later.
            if step == diverge_at:
                loss, psnr = loss * 6, psnr - 8
            else:
                loss, psnr = float("nan"), float("nan")
        rows.append({"step": step, "loss": round(loss, 5) if math.isfinite(loss) else loss,
                     "train_psnr": round(psnr, 3) if math.isfinite(psnr) else psnr,
                     "num_gaussians": gaussians})
        if diverge_at is not None and step >= diverge_at + 2 * LOG_EVERY:
            break
    return rows


def main() -> None:
    rng = random.Random(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, scene, iterations, overrides, final, extra in RUNS:
        run_dir = OUT / name
        run_dir.mkdir(parents=True)
        config = {"scene": scene, "iterations": iterations, **BASE_CONFIG, **overrides}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        with open(run_dir / "metrics.jsonl", "w") as f:
            for row in make_metrics(rng, iterations, extra):
                f.write(json.dumps(row) + "\n")  # NaN is written as a bare NaN, like many real loggers
        if final is not None:
            (run_dir / "eval.json").write_text(json.dumps({"step": iterations, **final}, indent=2) + "\n")
    print(f"wrote {len(RUNS)} runs to {OUT}")


if __name__ == "__main__":
    main()
