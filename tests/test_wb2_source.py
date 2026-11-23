import pytest
from climate_shift.wb2_source import open_public_wb2


def test_reject_unapproved_dataset_without_network():
    with pytest.raises(ValueError):
        open_public_wb2("gs://other-private-bucket/a.zarr")
    with pytest.raises(ValueError):
        open_public_wb2("https://somewhere.example/data.zarr")
