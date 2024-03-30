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

        if delta_rad > 0.5 * _face_expected_edge_angle_rad:
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


def plot(
    vertices,
    faces,
    path,
    faces_values=None,
    fill_color="RoyalBlue",
    show_grid=True,
):
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
    if show_grid:
        scp.hemisphere.ax_add_grid(ax=ax)
    scp.fig_write(fig=fig, path=path)


def find_faces_potential_neighbors(faces, vertices_to_faces_map):
    mm = {}
    for iface in range(len(faces)):
        pnn = set()
        for v in faces[iface]:
            for jface in vertices_to_faces_map[v]:
                pnn.add(jface)
        pnn.remove(iface)
        mm[iface] = list(pnn)
    return mm


def find_faces_neighbors(faces, vertices_to_faces_map):
    pn = find_faces_potential_neighbors(
        faces=faces,
        vertices_to_faces_map=vertices_to_faces_map,
    )
    nn = {}
    for iface in range(len(faces)):
        iset = set(faces[iface])
        for pface in pn[iface]:
            pset = set(faces[pface])
            if len(iset.intersection(pset)) == 2:
                if iface not in nn:
                    nn[iface] = [pface]
                else:
                    nn[iface].append(pface)
    return nn


def fill_faces_mask_if_two_neighbors_true(faces_mask, faces_neighbors):
    assert len(faces_mask) == len(faces_neighbors)
    out = np.asarray(faces_mask).copy()
    num_neighbors_high = np.zeros(len(faces_mask), int)

    for iface in range(len(faces_mask)):
        for nface in faces_neighbors[iface]:
            if faces_mask[nface]:
                num_neighbors_high[iface] += 1

    for iface in range(len(faces_mask)):
        if num_neighbors_high[iface] >= 2:
            out[iface] = True
    return out


def list_faces_inside_onedge_outside_zenith_distance(
    faces, vertices, zenith_rad
):
    inside = []
    onedge = []
    outside = []

    for iface in range(len(faces)):
        v0 = vertices[faces[iface][0]]
        v1 = vertices[faces[iface][1]]
        v2 = vertices[faces[iface][2]]

        _, z0 = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=v0[0], cy=v0[1], cz=v0[2]
        )
        _, z1 = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=v1[0], cy=v1[1], cz=v1[2]
        )
        _, z2 = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=v2[0], cy=v2[1], cz=v2[2]
        )

        o0 = z0 >= zenith_rad
        o1 = z1 >= zenith_rad
        o2 = z2 >= zenith_rad

        if o0 and o1 and o2:
            outside.append(iface)
        elif not o0 and not o1 and not o2:
            inside.append(iface)
        else:
            onedge.append(iface)

    return inside, onedge, outside


def estimate_intermediate_vertex_at_zenith(
    a,
    b,
    zenith_rad,
    epsilon_rad=1e-9,
    max_num_iterations=1000,
):
    assert max_num_iterations > 0
    assert epsilon_rad > 0

    a = np.asarray(a)
    b = np.asarray(b)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    _, a_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
        cx=a[0], cy=a[1], cz=a[2]
    )
    _, b_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
        cx=b[0], cy=b[1], cz=b[2]
    )

    assert np.abs(a_zd - b_zd) > epsilon_rad

    min_zd = min([a_zd, b_zd])
    max_zd = max([a_zd, b_zd])

    axis = np.cross(a, b)
    alpha = 0.5 * spherical_coordinates.angle_between_xyz(a, b)
    inum = 0
    delta_rad = 2 * np.pi
    proto = [0, 0, 0]

    rot_matrix = rot_matrix_from_axis_angle(axis=axis, angle_rad=alpha)
    proto = np.dot(rot_matrix, a)
    _, p_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
        proto[0], proto[1], proto[2]
    )
    if p_zd < min_zd or p_zd > max_zd:
        alpha *= -1.0

    while True:
        inum += 1
        assert inum <= max_num_iterations

        rot_matrix = rot_matrix_from_axis_angle(axis=axis, angle_rad=alpha)
        proto = np.dot(rot_matrix, a)

        _, p_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
            proto[0], proto[1], proto[2]
        )

        delta_rad = zenith_rad - p_zd

        if np.abs(delta_rad) < epsilon_rad:
            break

        alpha = alpha + 0.5 * delta_rad

    return proto


def rot_matrix_from_axis_angle(axis, angle_rad):
    a = np.asarray(axis)
    a = a / np.linalg.norm(a)

    x = a[0]
    y = a[1]
    z = a[2]

    cT = np.cos(angle_rad)
    sT = np.sin(angle_rad)

    r00 = cT + x * x * (1 - cT)
    r01 = x * y * (1 - cT) - z * sT
    r02 = x * z * (1 - cT) + y * sT

    r10 = y * x * (1 - cT) + z * sT
    r11 = cT + y * y * (1 - cT)
    r12 = y * z * (1 - cT) - x * sT

    r20 = z * x * (1 - cT) - y * sT
    r21 = z * y * (1 - cT) + x * sT
    r22 = cT + z * z * (1 - cT)

    R = np.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ]
    )
    return R


def draw_point_on_triangle(prng, a, b, c):
    """
             c---------------t
            / -             /
       ac /     -         /
        /         -     /
      /             - /
    a----------------b
           ab
    """
    _a = np.asarray(a)
    _b = np.asarray(b)
    _c = np.asarray(c)

    ab = _b - _a
    ac = _c - _a

    u1 = prng.uniform()
    u2 = prng.uniform()

    if u1 + u2 > 1.0:
        u1 = 1 - u1
        u2 = 1 - u2

    return _a + ab * u1 + ac * u2


def is_point_in_triangle(a, b, c, p):
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    p = np.asarray(p)

    norm = np.linalg.norm
    cross = np.cross

    ab = b - a
    ac = c - a

    pa = a - p
    pb = b - p
    pc = c - p

    area = norm(cross(ab, ac)) / 2.0

    alpha = norm(cross(pb, pc)) / (2 * area)
    beta = norm(cross(pc, pa)) / (2 * area)
    gamma = norm(cross(pa, pb)) / (2 * area)

    _range = (alpha + beta + gamma) / 3.0
    epsilon = 1e-6 * _range

    if 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0:
        if np.abs(alpha + beta + gamma - 1.0) < epsilon:
            return True
        else:
            return False
    else:
        return False


def average(faces, vertices, faces_weights):
    total = np.zeros(3)

    for i in range(len(faces)):
        v0 = vertices[faces[i]][0]
        v1 = vertices[faces[i]][1]
        v2 = vertices[faces[i]][2]
        vm = (v0 + v1 + v2) / 3.0
        vm = vm * (faces_weights[i] / np.linalg.norm(vm))
        total += vm

    return total / np.linalg.norm(total)
