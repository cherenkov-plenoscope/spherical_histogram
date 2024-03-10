from spherical_histogram import mesh
import numpy as np


def test_draw():
    prng = np.random.Generator(np.random.PCG64(132))

    for t in range(100):
        a = prng.uniform(low=-1, high=1, size=3)
        b = prng.uniform(low=-1, high=1, size=3)
        c = prng.uniform(low=-1, high=1, size=3)

        abc = np.c_[a, b, c]
        axis_aligned_bounding_box = [
            [min(abc[0, :]), max(abc[0, :])],
            [min(abc[1, :]), max(abc[1, :])],
            [min(abc[2, :]), max(abc[2, :])],
        ]

        for i in range(100):
            point = mesh.draw_point_on_triangle(prng=prng, a=a, b=b, c=c)

            # is inside axis aligned bounding box of triangle
            for dim in range(3):
                assert (
                    axis_aligned_bounding_box[dim][0]
                    <= point[dim]
                    <= axis_aligned_bounding_box[dim][1]
                )

            assert mesh.is_point_in_triangle(a=a, b=b, c=c, p=point)


def test_inside_triangle():
    a = [-0.5, 0, 0]
    b = [0.5, 0, 0]
    c = [0, 1.0, 0]

    assert not mesh.is_point_in_triangle(a=a, b=b, c=c, p=[-1, 0, 0])
    assert not mesh.is_point_in_triangle(a=a, b=b, c=c, p=[1, 0, 0])
    assert not mesh.is_point_in_triangle(a=a, b=b, c=c, p=[0, 2, 0])

    assert mesh.is_point_in_triangle(a=a, b=b, c=c, p=[0, 1, 0])
    assert mesh.is_point_in_triangle(a=a, b=b, c=c, p=[0, 0.5, 0])
    assert not mesh.is_point_in_triangle(a=a, b=b, c=c, p=[0, 0.5, 1e-3])
