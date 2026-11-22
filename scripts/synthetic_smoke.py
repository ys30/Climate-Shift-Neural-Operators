"""CPU smoke run on fabricated fields. NEVER interpret as ERA5/CMIP6 evidence."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from climate_shift.models import FNO2d
from climate_shift.metrics import regression_metrics

def make_toy(n: int = 48, size: int = 16, seed: int = 2026):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    fields = []
    for _ in range(n):
        p, q = rng.uniform(0, 2 * np.pi, size=2)
        fields.append(np.sin(2*np.pi*xx/size + p) + 0.4*np.cos(2*np.pi*yy/size+q))
    x = np.asarray(fields, np.float32)[:, None]
    # Synthetic stationary local averaging, not a physically simulated weather process.
    y = (x + np.roll(x, 1, -1) + np.roll(x, -1, -1)) / 3
    return x, y

def run(steps: int, seed: int) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x, y = make_toy(seed=seed)
    # Deterministic predeclared split; no validation/test-driven hyperparameter tuning.
    train_x, train_y = torch.from_numpy(x[:32]), torch.from_numpy(y[:32])
    test_x, test_y = torch.from_numpy(x[40:]), torch.from_numpy(y[40:])
    model = FNO2d(width=8, modes_y=4, modes_x=4, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((model(train_x) - train_y) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(test_x).numpy()
    result = {"dataset": "SYNTHETIC_ONLY", "seed": seed, "steps": steps,
              "fno": regression_metrics(pred, test_y.numpy()),
              "persistence": regression_metrics(test_x.numpy(), test_y.numpy())}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be > 0')
    print(json.dumps(run(args.steps, args.seed), indent=2))
