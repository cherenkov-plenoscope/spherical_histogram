from . import mesh

import merlict
import spherical_coordinates
import numpy as np


def make_merlict_scenery_py(vertices, faces):
    scenery_py = merlict.scenery.init(default_medium="vacuum")
    scenery_py["geometry"]["objects"]["hemisphere"] = mesh.make_obj(
        vertices=vertices, faces=faces, mtlkey="sky"
    )
    scenery_py["materials"]["surfaces"][
        "perfect_absorber"
    ] = merlict.materials.surfaces.init(key="perfect_absorber")

    scenery_py["geometry"]["relations"]["children"].append(
        {
            "id": 0,
            "pos": [0, 0, 0],
            "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
            "obj": "hemisphere",
            "mtl": {"sky": "abc"},
        }
    )
    scenery_py["materials"]["boundary_layers"]["abc"] = {
        "inner": {"medium": "vacuum", "surface": "perfect_absorber"},
        "outer": {"medium": "vacuum", "surface": "perfect_absorber"},
    }
    return scenery_py


class Tree:
    """
    An acceleration structure to allow fast queries for rays hitting a
    mesh defined by vertices and faces.
    """

    def __init__(self, vertices, faces):
        """
        Parameters
        ----------
        vertices : numpy.array, shape(M, 3), float
            The xyz-coordinates of the M vertices. The vertices are expected
            to be on the unit-sphere.
        faces : numpy.array, shape(N, 3), int
            A list of N faces referencing their vertices.
        """
        scenery_py = make_merlict_scenery_py(vertices=vertices, faces=faces)
        self._tree = merlict.compile(sceneryPy=scenery_py)

    def _make_probing_rays(self, cxcycz):
        assert len(cxcycz.shape) == 2
        assert cxcycz.shape[1] == 3

        size = cxcycz.shape[0]
        rays = merlict.ray.init(size)
        rays["support.x"] = np.zeros(size)
        rays["support.y"] = np.zeros(size)
        rays["support.z"] = np.zeros(size)
        rays["direction.x"] = cxcycz[:, 0]
        rays["direction.y"] = cxcycz[:, 1]
        rays["direction.z"] = cxcycz[:, 2]
        return rays

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        cxcycz = np.c_[cx, cy, cz]
        return self.query_cxcycz(cxcycz=cxcycz)

    def query_cx_cy(self, cx, cy):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        cxcycz = np.c_[cx, cy, cz]
        return self.query_cxcycz(cxcycz=cxcycz)

    def query_cxcycz(self, cxcycz):
        assert len(cxcycz.shape) == 2
        assert cxcycz.shape[1] == 3
        size = cxcycz.shape[0]

        rays = self._make_probing_rays(cxcycz=cxcycz)
        _hits, _intersecs = self._tree.query_intersection(rays)

        face_ids = np.zeros(size, dtype=int)
        face_ids[np.logical_not(_hits)] = -1
        face_ids[_hits] = _intersecs["geometry_id.face"][_hits]
        return face_ids
