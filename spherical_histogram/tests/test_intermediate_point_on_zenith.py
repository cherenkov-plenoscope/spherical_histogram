from spherical_histogram import mesh
import spherical_coordinates as sc
import numpy as np


def test_intermediate_point_on_zenith():
    d2r = np.deg2rad
    prng = np.random.Generator(np.random.PCG64(132))

    epsilon_rad = 1e-9
    cut_zenith_rad = d2r(70.0)

    for case in range(1000):
        a_zd = d2r(prng.uniform(low=50, high=90))

        a = sc.az_zd_to_cx_cy_cz(
            azimuth_rad=d2r(prng.uniform(low=-180, high=180)), zenith_rad=a_zd
        )
        if a_zd > d2r(70.0):
            b_zd = d2r(prng.uniform(low=50, high=70))
        else:
            b_zd = d2r(prng.uniform(low=70, high=90))

        b = sc.az_zd_to_cx_cy_cz(
            azimuth_rad=d2r(prng.uniform(low=-180, high=180)),
            zenith_rad=b_zd,
        )

        p1 = mesh.estimate_intermediate_vertex_at_zenith(
            a=a,
            b=b,
            zenith_rad=cut_zenith_rad,
            epsilon_rad=epsilon_rad,
        )

        p1_az, p1_zd = sc.cx_cy_cz_to_az_zd(cx=p1[0], cy=p1[1], cz=p1[2])

        assert (
            cut_zenith_rad - epsilon_rad < p1_zd < cut_zenith_rad + epsilon_rad
        )
