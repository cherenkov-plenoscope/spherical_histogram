import spherical_histogram as sh
import numpy as np
import spherical_coordinates as sc


def draw_az_zd(prng, size=None):
    return sc.random.uniform_az_zd_in_cone(
        prng=prng,
        azimuth_rad=0.0,
        zenith_rad=0.0,
        min_half_angle_rad=0.0,
        max_half_angle_rad=0.5 * np.pi,
        size=size,
    )


def draw_cx_cy_cz(prng, size=None):
    az, zd = draw_az_zd(prng=prng, size=size)
    return sc.az_zd_to_cx_cy_cz(azimuth_rad=az, zenith_rad=zd)


def draw_cx_cy(prng, size=None):
    cx, cy, _ = draw_cx_cy_cz(prng=prng, size=size)
    return cx, cy


def test_init_and_inputs():
    hemihist = sh.HemisphereHistogram(
        num_vertices=200,
        max_zenith_distance_rad=np.deg2rad(90),
    )

    prng = np.random.Generator(np.random.PCG64(232))
    ha = np.deg2rad(15)

    # test cone
    # ---------
    az, zd = draw_az_zd(prng=prng)
    hemihist.assign_cone_azimuth_zenith(
        azimuth_rad=az, zenith_rad=zd, half_angle_rad=ha
    )

    cx, cy = draw_cx_cy(prng=prng)
    hemihist.assign_cone_cx_cy(cx=cx, cy=cy, half_angle_rad=ha)

    cx, cy, cz = draw_cx_cy_cz(prng=prng)
    hemihist.assign_cone_cx_cy_cz(cx=cx, cy=cy, cz=cz, half_angle_rad=ha)

    # test regular pointing
    # ---------------------

    # array-like
    SIZE = 100
    az, zd = draw_az_zd(prng=prng, size=SIZE)
    hemihist.assign_azimuth_zenith(azimuth_rad=az, zenith_rad=zd)

    cx, cy = draw_cx_cy(prng=prng, size=SIZE)
    hemihist.assign_cx_cy(cx=cx, cy=cy)

    cx, cy, cz = draw_cx_cy_cz(prng=prng, size=SIZE)
    hemihist.assign_cx_cy_cz(cx=cx, cy=cy, cz=cz)

    # scalar like
    SIZE = None
    az, zd = draw_az_zd(prng=prng, size=SIZE)
    hemihist.assign_azimuth_zenith(azimuth_rad=az, zenith_rad=zd)

    cx, cy = draw_cx_cy(prng=prng, size=SIZE)
    hemihist.assign_cx_cy(cx=cx, cy=cy)

    cx, cy, cz = draw_cx_cy_cz(prng=prng, size=SIZE)
    hemihist.assign_cx_cy_cz(cx=cx, cy=cy, cz=cz)

    # test str
    str(hemihist)
