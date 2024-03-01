from .version import __version__
from . import mesh
from . import tree
from . import geometry

import spherical_coordinates
import numpy as np
import copy


class HemisphereHistogram:
    """
    Histogram pointings/directions in a hemisphere.

    Fields
    ------
    bin_counts : numpy.array
        The contetn of the bins.
    overflow : int
        When a pointong is assigned to the histogram which does not hit any bin
        this overflow counter is raised.
    bin_geometry : spherical_histogram.geometry.HemisphereGeometry
        The geometry of the bins. Each bin is a triangular face on the unit
        sphere. Faces are defined by their vertices. The bin_geometry stores
        theses vertices and the faces. It further knows the face's neigboring
        relations and each face's solid angle.
    """

    def __init__(
        self,
        num_vertices=2047,
        max_zenith_distance_rad=np.deg2rad(89.0),
        bin_geometry=None,
    ):
        """
        Provide either a ``bin_geometry``, or ``num_vertices`` and
        ``max_zenith_distance_rad`` to create a bin_geometry on the fly.
        """
        if bin_geometry is None:
            self.bin_geometry = geometry.HemisphereGeometry.from_num_vertices_and_max_zenith_distance_rad(
                num_vertices=num_vertices,
                max_zenith_distance_rad=max_zenith_distance_rad,
            )
        else:
            self.bin_geometry = bin_geometry

        self.reset()

    def reset(self):
        """
        Resets the bin content ``bin_counts`` and  the ``overflow`` to zero.
        """
        self.overflow = 0
        self.bin_counts = np.zeros(len(self.bin_geometry.faces), dtype=int)

    def solid_angle(self, threshold=1):
        """
        Returns the total solid angle of all bins with a content >= threshold.

        Parameters
        ----------
        threshold : int / float
            Minimum content of a bin in order to sum its solid angle.

        Returns
        -------
        solid_angle : float
            The total solid angle covered by all bins with a
            content >= threshold.
        """
        if threshold == 0:
            return np.sum(self.bin_geometry.faces_solid_angles)

        total_sr = 0.0
        for iface in self.bin_counts:
            if self.bin_counts[iface] >= threshold:
                total_sr += self.bin_geometry.faces_solid_angles[iface]
        return total_sr

    def assign_cx_cy_cz(self, cx, cy, cz):
        faces = self.bin_geometry.query_cx_cy_cz(cx=cx, cy=cy, cz=cz)
        self._assign(faces)

    def assign_cx_cy(self, cx, cy):
        faces = self.bin_geometry.query_cx_cy(cx=cx, cy=cy)
        self._assign(faces)

    def assign_azimuth_zenith(self, azimuth_rad, zenith_rad):
        faces = self.bin_geometry.query_azimuth_zenith(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
        )
        self._assign(faces)

    def assign_cone_cx_cy_cz(self, cx, cy, cz, half_angle_rad):
        self._assign(
            self.bin_geometry.query_cone_cx_cy_cz(
                cx=cx, cy=cy, cz=cz, half_angle_rad=half_angle_rad
            )
        )

    def assign_cone_cx_cy(self, cx, cy, half_angle_rad):
        self._assign(
            self.bin_geometry.query_cone_cx_cy(
                cx=cx, cy=cy, half_angle_rad=half_angle_rad
            )
        )

    def assign_cone_azimuth_zenith(
        self, azimuth_rad, zenith_rad, half_angle_rad
    ):
        self._assign(
            self.bin_geometry.query_cone_azimuth_zenith(
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                half_angle_rad=half_angle_rad,
            )
        )

    def _assign(self, faces):
        faces = np.asarray(faces, dtype=int)
        if faces.ndim == 0:
            faces = faces[np.newaxis]

        valid = faces >= 0
        self.overflow += np.sum(np.logical_not(valid))
        valid_faces = faces[valid]
        unique_faces, counts = np.unique(valid_faces, return_counts=True)
        self.bin_counts[unique_faces] += counts

    def to_dict(self):
        return {"overflow": self.overflow, "bin_counts": self.bin_counts}

    def plot(self, path):
        """
        Writes a plot with the grid's faces to path.
        """
        faces_values = copy.deepcopy(self.bin_counts.astype(float))

        if np.max(faces_values) > 0:
            faces_values /= np.max(faces_values)
        mesh.plot(
            path=path,
            faces=self.bin_geometry.faces,
            vertices=self.bin_geometry.vertices,
            faces_values=faces_values,
        )

    def __repr__(self):
        return "{:s}()".format(
            self.__class__.__name__,
        )
