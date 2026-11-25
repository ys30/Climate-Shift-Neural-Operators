"""Audit a transparent, parameter-near capacity grid without training.

This planner matches *real scalar degrees of freedom*, not FLOPs, latency,
optimization difficulty, or scientific model capacity. Never interpret a
parameter-near pair as a fair compute-matched benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from climate_shift.parameter_audit import inventory


def plan_capacity(
    grid_size: int = 17,
    fno_configs: tuple[tuple[int, int], ...] = ((8, 4), (12, 6), (16, 8)),
    unet_widths: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48),
) -> dict:
    if not fno_configs or not unet_widths:
        raise ValueError("Provide nonempty FNO and U-Net candidate grids")
    if len(set(unet_widths)) != len(unet_widths):
        raise ValueError("U-Net widths must be unique")
    if any(w < 1 for w in unet_widths):
        raise ValueError("U-Net widths must be positive")
    if len(set(fno_configs)) != len(fno_configs):
        raise ValueError("FNO configurations must be unique")
    unet_counts = {
        width: inventory(width=width, modes=1, grid_size=grid_size)["unet"][
            "real_scalar_degrees_of_freedom"
        ] for width in unet_widths
    }
    pairs = []
    for width, modes in fno_configs:
        counts = inventory(width=width, modes=modes, grid_size=grid_size)["fno"]
        target = counts["real_scalar_degrees_of_freedom"]
        matched = min(unet_counts, key=lambda w: (abs(unet_counts[w] - target), w))
        gap = (unet_counts[matched] - target) / target
        pairs.append({
            "fno_width": width,
            "fno_modes": modes,
            "fno_depth": 2,
            "fno_complex_torch_elements": counts["torch_parameter_elements"],
            "fno_real_dof": target,
            "unet_width": matched,
            "unet_real_dof": unet_counts[matched],
            "relative_unet_minus_fno_real_dof": gap,
            "reference_grid": [grid_size, grid_size],
        })
    return {
        "status": "design_only_no_training_or_results",
        "pairs": pairs,
        "candidate_unet_widths": list(unet_widths),
        "match_metric": "absolute difference in real scalar degrees of freedom",
        "fixed_context": {
            "variable": "ERA5 2m temperature",
            "lead_hours": 24,
            "train_years": [2008, 2009, 2010],
            "validation_years": [2011],
            "test_years": [2012],
        },
        "limitations": [
            "Parameter matching does not equate computational cost or architecture expressivity",
            "Complex Fourier weights require explicit two-real-scalar accounting",
            "Fourier resolution and depth should be sensitivity axes, not tuned on test year",
            "Independent multi-year evaluation is required for scientific inference",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = plan_capacity(grid_size=a.grid_size)
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
