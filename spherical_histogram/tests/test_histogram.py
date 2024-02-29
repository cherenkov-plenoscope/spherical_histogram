import spherical_histogram
import numpy as np


def test_init():
    hh = spherical_histogram.HemisphereHistogram(
        num_vertices=200,
        max_zenith_distance_rad=np.deg2rad(90),
    )

    # test str
    str(hh)
