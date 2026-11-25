import pytest
from scripts.plan_e1_capacity import plan_capacity
from climate_shift.parameter_audit import inventory


def test_parameter_near_capacity_grid_is_reproducible():
    plan = plan_capacity()
    assert plan["status"] == "design_only_no_training_or_results"
    assert len(plan["pairs"]) == 3
    assert [(x["fno_width"], x["fno_modes"]) for x in plan["pairs"]] == [
        (8, 4), (12, 6), (16, 8)
    ]
    for row in plan["pairs"]:
        fno = inventory(width=row["fno_width"], modes=row["fno_modes"], grid_size=17)["fno"]
        unet = inventory(width=row["unet_width"], modes=1, grid_size=17)["unet"]
        assert row["fno_real_dof"] == fno["real_scalar_degrees_of_freedom"]
        assert row["unet_real_dof"] == unet["real_scalar_degrees_of_freedom"]
        assert row["relative_unet_minus_fno_real_dof"] == pytest.approx(
            (row["unet_real_dof"] - row["fno_real_dof"]) / row["fno_real_dof"]
        )


def test_capacity_planner_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        plan_capacity(fno_configs=())
    with pytest.raises(ValueError):
        plan_capacity(unet_widths=(8, 8))
    with pytest.raises(ValueError):
        plan_capacity(fno_configs=((8, 10),))
    with pytest.raises(ValueError):
        plan_capacity(unet_widths=(0, 8))
