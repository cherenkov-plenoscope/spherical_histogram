from .version import __version__
from . import mesh
from . import tree
from . import geometry

import spherical_coordinates
import numpy as np
import copy


class HemisphereHistogram:
    def __init__(
        self,
        num_vertices=2047,
        max_zenith_distance_rad=np.deg2rad(89.0),
        bin_geometry=None,
    ):
        if bin_geometry is None:
            self.bin_geometry = geometry.HemisphereGeometry(
                num_vertices=num_vertices,
                max_zenith_distance_rad=max_zenith_distance_rad,
            )
        else:
            self.bin_geometry = bin_geometry

        self.reset()

    def reset(self):
        """
        Resets the bin content to zero.
        """
        self.overflow = 0
        self.bin_counts = np.zeros(len(self.bin_geometry.faces), dtype=int)

    def solid_angle(self, threshold=1):
        """
        Returns the solid angle of all faces with a content above a certain
        threshold.

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

    def assign_cxcycz(self, cxcycz):
        faces = self.bin_geometry.query_cxcycz(cxcycz=cxcycz)
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

    def assign_cone_cxcycz(self, cxcycz, half_angle_rad):
        """
        Assign

        Parameters
        ----------
        cxcycz
        """
        assert half_angle_rad >= 0
        assert 0.99 <= np.linalg.norm(cxcycz) <= 1.01
        third_neighbor_angle_rad = np.max(
            self.bin_geometry.vertices_tree.query(x=cxcycz, k=3)[0]
        )
        query_angle_rad = np.max([half_angle_rad, third_neighbor_angle_rad])
        vidx_in_cone = self.bin_geometry.vertices_tree.query_ball_point(
            x=cxcycz,
            r=query_angle_rad,
        )

        faces = set()  # count only once
        for vidx in vidx_in_cone:
            faces_touching_vidx = self.bin_geometry.vertices_to_faces_map[vidx]
            for face in faces_touching_vidx:
                faces.add(face)
        self._assign(list(faces))

    def assign_cone_cx_cy(self, cx, cy, half_angle_rad):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        cxcycz = np.c_[cx, cy, cz]
        return self.assign_cone_cxcycz(
            cxcycz=cxcycz, half_angle_rad=half_angle_rad
        )

    def assign_cone_azimuth_zenith(
        self, azimuth_rad, zenith_rad, half_angle_rad
    ):
        cxcycz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        return self.assign_cone_cxcycz(
            cxcycz=cxcycz, half_angle_rad=half_angle_rad
        )

    def _assign(self, faces):
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
        return "{:s}(num_vertices={:d}, max_zenith_distance_rad={:f})".format(
            self.__class__.__name__,
            self.bin_geometry._init_num_vertices,
            self.bin_geometry.max_zenith_distance_rad,
        )
