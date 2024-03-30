from . import tree
from . import mesh

import numpy as np
import spherical_coordinates
import solid_angle_utils
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
        self.faces_neighbors = mesh.find_faces_neighbors(
            faces=self.faces,
            vertices_to_faces_map=self.vertices_to_faces_map,
        )

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

    def query_cone_weiths_azimuth_zenith(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        num_probing_rays_per_sr=4e5,
        path=None,
    ):
        cone_solid_angle_sr = solid_angle_utils.cone.solid_angle(
            half_angle_rad=half_angle_rad
        )
        num_probing_rays = cone_solid_angle_sr * num_probing_rays_per_sr
        num_probing_rays = int(np.ceil(num_probing_rays))
        seed = np.abs(hash((azimuth_rad, zenith_rad, half_angle_rad)))
        prng = np.random.Generator(np.random.PCG64(seed))

        _cx, _cy, _cz = draw_in_cone(
            prng=prng,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
            size=num_probing_rays,
        )

        _faces = self.query_cx_cy_cz(_cx, _cy, _cz)

        if len(_faces) == 0:
            out = (np.array([], dtype=int), np.array([], dtype=float))
        else:
            unique_faces, counts = np.unique(_faces, return_counts=True)
            weights = counts / np.sum(counts)
            out = unique_faces, weights

        if path is not None:
            w = np.zeros(len(self.faces))
            w[out[0]] = out[1]
            w = w / np.max(w)
            mesh.plot(
                vertices=self.vertices,
                faces=self.faces,
                faces_values=w,
                path=path,
            )

        return out

    def plot(self, **kwargs):
        """
        Writes a plot with the grid's faces to path.
        """
        mesh.plot(vertices=self.vertices, faces=self.faces, **kwargs)

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)


def draw_in_cone(prng, azimuth_rad, zenith_rad, half_angle_rad, size):
    min_half_angle_rad = 0.0

    # Adopted from CORSIKA
    rd2 = prng.uniform(size=size)
    ct1 = np.cos(min_half_angle_rad)
    ct2 = np.cos(half_angle_rad)
    ctt = rd2 * (ct2 - ct1) + ct1
    theta = np.arccos(ctt)
    phi = prng.uniform(low=0.0, high=np.pi * 2.0, size=size)

    # temporary cartesian coordinates
    cx1, cy1, cz1 = spherical_coordinates.az_zd_to_cx_cy_cz(
        azimuth_rad=phi, zenith_rad=theta
    )
    ____0 = 0.0
    ____1 = 1.0

    # Rotate around y axis
    cosZd = np.cos(zenith_rad)
    sinZd = np.sin(zenith_rad)

    cx2 = +cx1 * cosZd + cy1 * ____0 + cz1 * sinZd
    cy2 = +cx1 * ____0 + cy1 * ____1 + cz1 * ____0
    cz2 = -cx1 * sinZd + cy1 * ____0 + cz1 * cosZd

    cosAz = np.cos(azimuth_rad)
    sinAz = np.sin(azimuth_rad)

    # rotate around z
    cx3 = +cx2 * cosAz - cy2 * sinAz + cz2 * ____0
    cy3 = +cx2 * sinAz + cy2 * cosAz + cz2 * ____0
    cz3 = +cx2 * ____0 + cy2 * ____0 + cz2 * ____1

    return cx3, cy3, cz3
