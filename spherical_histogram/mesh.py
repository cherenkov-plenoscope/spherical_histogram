import binning_utils
import scipy
from scipy import spatial
import numpy as np
import solid_angle_utils
import spherical_coordinates
import triangle_mesh_io
import svg_cartesian_plot


def make_vertices(
    num_vertices,
    max_zenith_distance_rad,
):
    """
    Makes vertices on a unit-sphere using a Fibonacci-space.
    This is done to create mesh-faces of approximatly equal solid angles.

    Additional vertices are added at the horizon all around the azimuth to make
    shure the resulting mesh reaches the horizon at any azimuth.

    The Fibinacci-vertices and the horizon-ring-vertices are combined, while
    Fibonacci-vertices will be dropped when they are too close to existing
    vertices on the horizon-ring.

    Parameters
    ----------
    num_vertices : int
        A guidence for the number of verties in the mesh.
    max_zenith_distance_rad : float
        Vertices will only be put up to this zenith-distance.
        The ring-vertices will be put right at this zenith-distance.

    Returns
    -------
    vertices : numpy.array, shape(N, 3)
        The xyz-coordinates of the vertices.
    """
    PI = np.pi
    TAU = 2 * PI

    assert 0 < max_zenith_distance_rad <= np.pi / 2
    assert num_vertices > 0
    num_vertices = int(num_vertices)

    inner_vertices = binning_utils.sphere.fibonacci_space(
        size=num_vertices,
        max_zenith_distance_rad=max_zenith_distance_rad,
    )

    _hemisphere_solid_angle = 2.0 * np.pi
    _expected_num_faces = 2.0 * num_vertices
    _face_expected_solid_angle = _hemisphere_solid_angle / _expected_num_faces
    _face_expected_edge_angle_rad = np.sqrt(_face_expected_solid_angle)
    num_horizon_vertices = int(np.ceil(TAU / _face_expected_edge_angle_rad))

    horizon_vertices = []
    for az_rad in np.linspace(0, TAU, num_horizon_vertices, endpoint=False):
        uvec = np.array(
            spherical_coordinates.az_zd_to_cx_cy_cz(
                azimuth_rad=az_rad,
                zenith_rad=max_zenith_distance_rad,
            )
        )
        horizon_vertices.append(uvec)
    horizon_vertices = np.array(horizon_vertices)

    vertices = []

    _horizon_vertices_tree = scipy.spatial.cKDTree(data=horizon_vertices)
    for inner_vertex in inner_vertices:
        delta_rad, vidx = _horizon_vertices_tree.query(inner_vertex)

        if delta_rad > _face_expected_edge_angle_rad:
            vertices.append(inner_vertex)

    for horizon_vertex in horizon_vertices:
        vertices.append(horizon_vertex)

    return np.array(vertices)


def make_faces(vertices):
    """
    Makes Delaunay-Triangle-faces for the given vertices. Only the x- and
    y coordinate are taken into account.

    Parameters
    ----------
    vertices : numpy.array
        The xyz-coordinates of the vertices.

    Returns
    -------
    delaunay_faces : numpy.array, shape(N, 3), int
        A list of N faces, where each face references the vertices it is made
        from.
    """
    delaunay = scipy.spatial.Delaunay(points=vertices[:, 0:2])
    delaunay_faces = delaunay.simplices
    return delaunay_faces


def estimate_vertices_to_faces_map(faces, num_vertices):
    """
    Parameters
    ----------
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    num_vertices : int
        The total number of vertices in the mesh

    Returns
    -------
    nn : dict of lists
        A dict with an entry for each vertex referencing the faces it is
        connected to.
    """
    nn = {}
    for iv in range(num_vertices):
        nn[iv] = set()

    for iface, face in enumerate(faces):
        for iv in face:
            nn[iv].add(iface)

    out = {}
    for key in nn:
        out[key] = list(nn[key])
    return out


def estimate_solid_angles(vertices, faces, geometry="spherical"):
    """
    For a given hemispherical mesh defined by vertices and faces, calculate the
    solid angle of each face.

    Parameters
    ----------
    vertices : numpy.array, shape(M, 3), float
        The xyz-coordinates of the M vertices.
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    geometry : str, default="spherical"
        Whether to apply "spherical" or "flat" geometry. Where "flat" geometry
        is only applicable for small faces.

    Returns
    -------
    solid : numpy.array, shape=(N, ), float
        The individual solid angles of the N faces in the mesh
    """
    solid = np.nan * np.ones(len(faces))
    for i in range(len(faces)):
        face = faces[i]
        if geometry == "spherical":
            face_solid_angle = solid_angle_utils.triangle.solid_angle(
                v0=vertices[face[0]],
                v1=vertices[face[1]],
                v2=vertices[face[2]],
            )
        elif geometry == "flat":
            face_solid_angle = (
                solid_angle_utils.triangle._area_of_flat_triangle(
                    v0=vertices[face[0]],
                    v1=vertices[face[1]],
                    v2=vertices[face[2]],
                )
            )
        else:
            raise ValueError(
                "Expected geometry to be either 'flat' or 'spherical'."
            )

        solid[i] = face_solid_angle
    return solid


def vertices_and_faces_to_obj(vertices, faces, mtlkey="sky"):
    """
    Makes an object-wavefron dict() from the mesh defined by
    vertices and faces.

    Parameters
    ----------
    vertices : numpy.array, shape(M, 3), float
        The xyz-coordinates of the M vertices. The vertices are expected to be
        on the unit-sphere.
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    mtlkey : str, default="sky"
        Key indicating the first and only material in the object-wavefront.

    Returns
    -------
    obj : dict representing an object-wavefront
        Includes vertices, vertex-normals, and materials ('mtl's) with faces.
    """
    obj = triangle_mesh_io.obj.init()
    for vertex in vertices:
        obj["v"].append(vertex)
        # all vertices are on a sphere
        # so the vertex is parallel to its surface-normal.
        obj["vn"].append(vertex)
    obj["mtl"] = {}
    obj["mtl"][mtlkey] = []
    for face in faces:
        obj["mtl"][mtlkey].append({"v": face, "vn": face})
    return obj


def obj_to_vertices_and_faces(obj, mtlkey="sky"):
    vertices = []
    faces = []
    for v in obj["v"]:
        vertices.append(v)
    for f in obj["mtl"][mtlkey]:
        faces.append(f["v"])
    return np.asarray(vertices), np.array(faces)


def plot(vertices, faces, path, faces_values=None, fill_color="RoyalBlue"):
    """
    Writes an svg figure to path.
    """
    scp = svg_cartesian_plot

    fig = scp.Fig(cols=1080, rows=1080)
    ax = scp.hemisphere.Ax(fig=fig)
    mesh_look = scp.hemisphere.init_mesh_look(
        num_faces=len(faces),
        stroke=None,
        fill=scp.color.css(fill_color),
        fill_opacity=1.0,
    )

    if faces_values is not None:
        for i in range(len(faces)):
            mesh_look["faces_fill_opacity"][i] = faces_values[i]
    scp.hemisphere.ax_add_mesh(
        ax=ax,
        vertices=vertices,
        faces=faces,
        max_radius=1.0,
        **mesh_look,
    )
    scp.hemisphere.ax_add_grid(ax=ax)
    scp.fig_write(fig=fig, path=path)
