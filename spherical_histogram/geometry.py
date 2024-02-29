from . import tree
from . import mesh

import numpy as np
import scipy
from scipy import spatial


class HemisphereGeometry:
    """
    A hemispherical grid with a Fibonacci-spacing.
    """

    def __init__(
        self,
        num_vertices,
        max_zenith_distance_rad,
    ):
        """
        Parameters
        ----------
        num_vertices : int
            A guideline for the number of vertices in the grid's mesh.
            See mesh.make_vertices().
        max_zenith_distance_rad : float
            Vertices will only be put up to this zenith-distance.
            The ring-vertices will be put right at this zenith-distance.
        """
        self._init_num_vertices = int(num_vertices)
        self.max_zenith_distance_rad = float(max_zenith_distance_rad)
        self.vertices = mesh.make_vertices(
            num_vertices=self._init_num_vertices,
            max_zenith_distance_rad=self.max_zenith_distance_rad,
        )
        self.vertices_tree = scipy.spatial.cKDTree(data=self.vertices)
        self.faces = mesh.make_faces(vertices=self.vertices)
        self.vertices_to_faces_map = mesh.estimate_vertices_to_faces_map(
            faces=self.faces, num_vertices=len(self.vertices)
        )
        self.faces_solid_angles = mesh.estimate_solid_angles(
            vertices=self.vertices,
            faces=self.faces,
        )
        self.tree = tree.Tree(vertices=self.vertices, faces=self.faces)

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        """
        Returns the index of the face hit at direction
        (azimuth_rad, zenith_rad).
        """
        return self.tree.query_azimuth_zenith(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )

    def query_cx_cy(self, cx, cy):
        """
        Returns the index of the face hit at direction (cx, cy).
        """
        return self.tree.query_cx_cy(cx=cx, cy=cy)

    def query_cx_cy_cz(self, cx, cy, cz):
        """
        Returns the index of the face hit by the vector cxcycz.
        """
        return self.tree.query_cx_cy_cz(cx, cy, cz)

    def plot(slef, path):
        """
        Writes a plot with the grid's faces to path.
        """
        mesh.plot(vertices=self.vertices, faces=self.faces, path=path)

    def __repr__(self):
        return "{:s}(num_vertices={:d}, max_zenith_distance_rad={:f})".format(
            self.__class__.__name__,
            self._init_num_vertices,
            self.max_zenith_distance_rad,
        )
