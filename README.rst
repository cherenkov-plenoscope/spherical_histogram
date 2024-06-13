###################
Spherical Histogram
###################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

Histograms directions into a hemisphere with bins of roughly the same solid angle.

|FigExampleWithCherenkovLight|

An example histogram of the sky down to zenith distance of 70``deg`` showing the
Cherenkov emission of an atmospheric shower with millions of photon directions in it.

*******
Install
*******

.. code-block:: bash

    pip install spherical_histogram


*****
Usage
*****

First a ``HemisphereHistogram`` histogram is initialized.
When providing general parameters for the desired binning such as
``num_vertices`` and ``max_zenith_distance_rad``, the geometry of
the hemisphere will be made on the fly using a Fibonacci spacing.

.. code-block:: python

    import spherical_histogram
    import numpy as np

    hist = spherical_histogram.HemisphereHistogram(
        num_vertices=200,
        max_zenith_distance_rad=np.deg2rad(90),
    )

|HemisphereGrid|

A Fibonacci spaced mesh of triangles which defines the bins of the
histogram.

|FigSolidAngleDistribution|

Distribution of solid angles in the upper mesh. The triangles have
similar zizes. Outliers are mostly caused by the hard cut on the
zenith distance.

Or by defining the binning explicitly using a triangle mesh with
``vertices`` and ``faces``.

.. code-block:: python

    import spherical_histogram

    geom = spherical_histogram.geometry.HemisphereGeometry(
        vertices=[[0, 0, 1], [0, 0.02, 1], [0.02, 0, 1]],
        faces=[[0, 1, 2]],
    )

    hist = spherical_histogram.HemisphereHistogram(bin_geometry=geom)

Afer initializing, we can histogram directions. This can be done multiple
times with any of the three options

Azimuth and zenith angle

.. code-block:: python

    hist.assign_azimuth_zenith(azimuth_rad=0.2, zenith_rad=0.1)

The direction vector's ``x`` and ``y`` components

.. code-block:: python

    hist.assign_cx_cy(cx=0.3, cy=0.2)

Or with the full direction vector (``x``, ``y``, and ``z``).

.. code-block:: python

    hist.assign_cx_cy_cz(cx=0.2, cy=0.3, cz=np.sqrt(1 - 0.2 ** 2 - 0.3 ** 2))

After all directions where assigned to the histogram, the result is found in

.. code-block:: python

    hist.bin_counts

and in

.. code-block:: python

    hist.overflow

where ``overflow`` counts all the directions which could not be assigned to a bin
and ``bin_counts`` is an array with one bount for each face in the hemispherical
mesh of triangles.

The ``assign`` functions accept both scalar and array like parameters for an easy
``numpy`` integration. When the directions are assignes in array like parameters
the loop for the assignment happens in the underlying ``c`` implementation and is
rather fast and efficient.

.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/spherical_histogram/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/spherical_histogram/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/spherical_histogram
    :target: https://pypi.org/project/spherical_histogram

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |MITLicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. |FigSolidAngleDistribution| image:: https://github.com/cherenkov-plenoscope/spherical_histogram/blob/main/readme/skymap_solid_angles.jpg?raw=True
    :width: 50%

.. |FigExampleWithCherenkovLight| image:: https://github.com/cherenkov-plenoscope/spherical_histogram/blob/main/readme/000000.primary_to_cherenkov.jpg?raw=True
    :width: 50%

.. |HemisphereGrid| image:: https://github.com/cherenkov-plenoscope/spherical_histogram/blob/main/readme/skymap_render_crop.jpg?raw=True
    :width: 50%
