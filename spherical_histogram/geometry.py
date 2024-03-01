from . import tree
from . import mesh

import numpy as np
import spherical_coordinates
import scipy
from scipy import spatial


class HemisphereGeometry:
    """
    A hemispherical grid with a Fibonacci-spacing.
    """

    def __init__(
        self,
        vertices,
        faces,
    ):
        """
        Parameters
        ----------
        vertices : [[cx,cy,cz], [cx,cy,cz], ... ]
            List of 3D vertices on the unit sphere (cx, cy, cz)
        faces : [[a1,b1,c1], [a2, b2, c2], ... ]
            List of indices to reference the three (exactly three) vertices
            which form a face on the unit sphere.
        """
        self.vertices = vertices
        self.faces = faces

        self.vertices_tree = scipy.spatial.cKDTree(data=self.vertices)
        self.vertices_to_faces_map = mesh.estimate_vertices_to_faces_map(
            faces=self.faces, num_vertices=len(self.vertices)
        )
        self.faces_solid_angles = mesh.estimate_solid_angles(
            vertices=self.vertices,
            faces=self.faces,
        )
        self.tree = tree.Tree(vertices=self.vertices, faces=self.faces)

    @classmethod
    def from_num_vertices_and_max_zenith_distance_rad(
        cls, num_vertices, max_zenith_distance_rad
    ):
        vertices = mesh.make_vertices(
            num_vertices=num_vertices,
            max_zenith_distance_rad=max_zenith_distance_rad,
        )
        faces = mesh.make_faces(vertices=vertices)
        return cls(vertices=vertices, faces=faces)

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        return self.tree.query_azimuth_zenith(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )

    def query_cx_cy(self, cx, cy):
        return self.tree.query_cx_cy(cx=cx, cy=cy)

    def query_cx_cy_cz(self, cx, cy, cz):
        return self.tree.query_cx_cy_cz(cx, cy, cz)

    def query_cone_cx_cy(self, cx, cy, half_angle_rad):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        return self.query_cone_cx_cy_cz(
            cx=cx, cy=cy, cz=cz, half_angle_rad=half_angle_rad
        )

    def query_cone_azimuth_zenith(
        self, azimuth_rad, zenith_rad, half_angle_rad
    ):
        cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        return self.query_cone_cx_cy_cz(
            cx=cx, cy=cy, cz=cz, half_angle_rad=half_angle_rad
        )

    def query_cone_cx_cy_cz(self, cx, cy, cz, half_angle_rad):
        cxcycz = np.asarray([cx, cy, cz])
        assert cxcycz.ndim == 1
        assert half_angle_rad >= 0
        assert 0.99 <= np.linalg.norm(cxcycz) <= 1.01

        # find the angle to the 3rd nearest neighbor vertex
        # -------------------------------------------------
        third_neighbor_angle_rad = np.max(
            self.vertices_tree.query(x=cxcycz, k=3)[0]
        )

        # make sure the query angle is at least as big as the angle
        # to the 3rd nearest neighbor vertex
        # ---------------------------------------------------------
        query_angle_rad = np.max([half_angle_rad, third_neighbor_angle_rad])

        # query vertices
        # --------------
        vidx_in_cone = self.vertices_tree.query_ball_point(
            x=cxcycz,
            r=query_angle_rad,
        )

        # identify the faces related to the vertices
        # ------------------------------------------
        faces = set()  # count each face only once
        for vidx in vidx_in_cone:
            faces_touching_vidx = self.vertices_to_faces_map[vidx]
            for face in faces_touching_vidx:
                faces.add(face)
        return np.array(list(faces))

    def plot(slef, path):
        """
        Writes a plot with the grid's faces to path.
        """
        mesh.plot(vertices=self.vertices, faces=self.faces, path=path)

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)
