"""Parameter accounting for compact E1 operator and convolution baselines."""
from __future__ import annotations
import argparse
import json
import torch
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d


def parameter_report(model: torch.nn.Module) -> dict:
    """Distinguish PyTorch element count from real scalar degrees of freedom."""
    groups: dict[str, dict] = {}
    elements = 0
    real_dof = 0
    bytes_total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        dof = 2 * count if param.is_complex() else count
        group = name.split(".")[0]
        entry = groups.setdefault(group, {"torch_elements": 0, "real_scalar_dof": 0, "storage_bytes": 0})
        entry["torch_elements"] += count
        entry["real_scalar_dof"] += dof
        entry["storage_bytes"] += count * param.element_size()
        elements += count
        real_dof += dof
        bytes_total += count * param.element_size()
    return {
        "torch_parameter_elements": elements,
        "real_scalar_degrees_of_freedom": real_dof,
        "parameter_storage_bytes": bytes_total,
        "parameter_groups": groups,
    }


def inventory(width: int = 8, modes: int = 4, grid_size: int = 17) -> dict:
    if width < 1 or modes < 1 or grid_size < max(4, 2 * modes):
        raise ValueError("Invalid width, modes or grid dimension for FNO")
    if grid_size // 2 + 1 < modes:
        raise ValueError("Grid insufficient for rfft modes")
    fno = FNO2d(width=width, modes_y=modes, modes_x=modes, depth=2)
    unet = UNet2d(width=width)
    # Model instantiation alone does not establish comparative compute cost.
    return {
        "configuration": {"width": width, "fno_modes": modes, "fno_depth": 2,
                          "single_input_channel": True, "grid_size": grid_size},
        "fno": parameter_report(fno),
        "unet": parameter_report(unet),
        "persistence": {"torch_parameter_elements": 0, "real_scalar_degrees_of_freedom": 0},
        "caution": ("FNO spectral weights are complex64; numel counts complex elements, "
                    "while each complex element represents two real scalar degrees of freedom. "
                    "Neither count alone matches compute cost between architectures."),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--output")
    args = p.parse_args()
    result = inventory(args.width, args.modes, args.grid_size)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        dest = Path(args.output)
        if dest.exists():
            raise FileExistsError(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
