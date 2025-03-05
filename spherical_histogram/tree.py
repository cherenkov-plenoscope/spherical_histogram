from . import mesh

import merlict
import spherical_coordinates
import numpy as np


def make_merlict_scenery_py(vertices, faces):
    scenery_py = merlict.scenery.init()
    scenery_py["geometry"]["objects"][
        "hemisphere"
    ] = mesh.vertices_and_faces_to_obj(
        vertices=vertices, faces=faces, mtlkey="sky"
    )

    # spectra
    scenery_py["materials"]["spectra"]["vacuum_absorption"] = (
        merlict.materials.spectra.init_from_resources("vacuum_absorption")
    )
    scenery_py["materials"]["spectra"]["vacuum_refraction"] = (
        merlict.materials.spectra.init_from_resources("vacuum_refraction")
    )
    scenery_py["materials"]["spectra"]["perfect_absorber_reflection"] = (
        merlict.materials.spectra.init_from_resources(
            "perfect_absorber_reflection"
        )
    )

    # media
    scenery_py["materials"]["media"]["vacuum"] = {
        "refraction_spectrum": "vacuum_refraction",
        "absorption_spectrum": "vacuum_absorption",
    }
    scenery_py["materials"]["default_medium"] = "vacuum"

    # surfaces
    scenery_py["materials"]["surfaces"]["absorber"] = {
        "type": "cook-torrance",
        "reflection_spectrum": "perfect_absorber_reflection",
        "diffuse_weight": 0.0,
        "specular_weight": 0.0,
        "roughness": 0.0,
    }

    # boundary layers
    scenery_py["materials"]["boundary_layers"]["abc"] = {
        "inner": {"medium": "vacuum", "surface": "absorber"},
        "outer": {"medium": "vacuum", "surface": "absorber"},
    }

    # geometry
    scenery_py["geometry"]["relations"]["children"].append(
        {
            "id": 0,
            "pos": [0, 0, 0],
            "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
            "obj": "hemisphere",
            "mtl": {"sky": "abc"},
        }
    )

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

    def _make_probing_rays(self, cx, cy, cz):
        size = len(cx)
        rays = merlict.ray.init(size)
        rays["support.x"] = np.zeros(size)
        rays["support.y"] = np.zeros(size)
        rays["support.z"] = np.zeros(size)
        rays["direction.x"] = cx
        rays["direction.y"] = cy
        rays["direction.z"] = cz
        return rays

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        return self.query_cx_cy_cz(cx=cx, cy=cy, cz=cz)

    def query_cx_cy(self, cx, cy):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        return self.query_cx_cy_cz(cx=cx, cy=cy, cz=cz)

    def query_cx_cy_cz(self, cx, cy, cz):
        cx_is_scalar, cx = spherical_coordinates.dimensionality._in(x=cx)
        cy_is_scalar, cy = spherical_coordinates.dimensionality._in(x=cy)
        cz_is_scalar, cz = spherical_coordinates.dimensionality._in(x=cz)
        assert cx_is_scalar == cy_is_scalar
        assert cx_is_scalar == cz_is_scalar
        is_scalar = cx_is_scalar

        rays = self._make_probing_rays(cx=cx, cy=cy, cz=cz)
        _hits, _intersecs = self._tree.query_intersection(rays)
        size = len(rays)

        face_ids = np.zeros(size, dtype=int)
        face_ids[np.logical_not(_hits)] = -1
        face_ids[_hits] = _intersecs["geometry_id.face"][_hits]
        return spherical_coordinates.dimensionality._out(
            is_scalar=is_scalar,
            x=face_ids,
        )
