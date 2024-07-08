"""
A lightly altered version of `trimesh.creation.triangulate_polygon`
for benchmarking in mostly isolated circumstances
"""
import collections
import json
import os
import timeit
from logging import StreamHandler, getLogger
from typing import Optional
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

# triangle crashes hard without this
from trimesh import grouping


# has license people complain about
# produces highest "quality" meshes if you care about that
import triangle
# release for Numpy2 blocked by PR
import mapbox_earcut
# new option being benchmarked
import manifold3d



log = getLogger(__name__)
log.addHandler(StreamHandler())
_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))

engines = ["triangle", "manifold", "earcut"]


def triangulate_polygon(
    polygon, triangle_args: Optional[str] = None, engine: Optional[str] = None, **kwargs
):
    """
    Given a shapely polygon create a triangulation using a
    python interface to the permissively licensed `mapbox-earcut`
    or the more robust `triangle.c`.
    > pip install manifold3d
    > pip install triangle
    > pip install mapbox_earcut

    Parameters
    ---------
    polygon : Shapely.geometry.Polygon
        Polygon object to be triangulated.
    triangle_args : str or None
        Passed to triangle.triangulate i.e: 'p', 'pq30', 'pY'="don't insert vert"
    engine : None or str
      None or 'earcut' will use earcut, 'triangle' will use triangle

    Returns
    --------------
    vertices : (n, 2) float
       Points in space
    faces : (n, 3) int
       Index of vertices that make up triangles
    """

    if polygon is None or polygon.is_empty:
        return [], []

    if "earcut" in engine:
        # get vertices as sequence where exterior
        # is the first value
        vertices = [np.array(polygon.exterior.coords)]
        vertices.extend(np.array(i.coords) for i in polygon.interiors)
        # record the index from the length of each vertex array
        rings = np.cumsum([len(v) for v in vertices])
        # stack vertices into (n, 2) float array
        vertices = np.vstack(vertices)
        # run triangulation
        faces = (
            mapbox_earcut.triangulate_float64(vertices, rings)
            .reshape((-1, 3))
            .astype(np.int64)
            .reshape((-1, 3))
        )

        return vertices, faces

    elif engine == "manifold":
        # the outer ring is wound counter-clockwise
        rings = [
            np.array(polygon.exterior.coords)[:: (1 if polygon.exterior.is_ccw else -1)]
        ]
        # wind interiors
        rings.extend(
            np.array(b.coords)[:: (-1 if b.is_ccw else 1)] for b in polygon.interiors
        )
        faces = manifold3d.triangulate(rings)
        vertices = np.vstack(rings)
        return vertices, faces

    elif engine == "triangle":
        # set default triangulation arguments if not specified
        if triangle_args is None:
            triangle_args = "p"
            # turn the polygon in to vertices, segments, and holes
        arg = _polygon_to_kwargs(polygon)
        # run the triangulation
        result = triangle.triangulate(arg, triangle_args)
        return result["vertices"], result["triangles"]

    log.warning(
        "try running `pip install manifold3d`"
        + "or `triangle`, `mapbox_earcut`, then explicitly pass:\n"
        + '`triangulate_polygon(*args, engine="triangle")`\n'
        + "to use the non-FSF-approved-license triangle engine"
    )
    raise ValueError("No available triangulation engine!")


def _polygon_to_kwargs(polygon):
    """
    Given a shapely polygon generate the data to pass to
    the triangle mesh generator

    Parameters
    ---------
    polygon : Shapely.geometry.Polygon
      Input geometry

    Returns
    --------
    result : dict
      Has keys: vertices, segments, holes
    """

    if not polygon.is_valid:
        raise ValueError("invalid shapely polygon passed!")

    def round_trip(start, length):
        """
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal.

        Examples
        ---------
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        """
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        # coords is an (n, 2) ordered list of points on the polygon boundary
        # the first and last points are the same, and there are no
        # guarantees on points not being duplicated (which will
        # later cause meshpy/triangle to shit a brick)
        coords = np.array(boundary.coords)
        # find indices points which occur only once, and sort them
        # to maintain order
        unique = np.sort(grouping.unique_rows(coords)[0])
        cleaned = coords[unique]

        vertices.append(cleaned)
        facets.append(round_trip(start, len(cleaned)))

        # holes require points inside the region of the hole, which we find
        # by creating a polygon from the cleaned boundary region, and then
        # using a representative point. You could do things like take the mean of
        # the points, but this is more robust (to things like concavity), if
        # slower.
        test = Polygon(cleaned)
        holes.append(np.array(test.representative_point().coords)[0])

        return len(cleaned)

    # sequence of (n,2) points in space
    vertices = []
    # sequence of (n,2) indices of vertices
    facets = []
    # list of (2) vertices in interior of hole regions
    holes = []

    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try:
            start += add_boundary(interior, start)
        except BaseException:
            log.warning("invalid interior, continuing")
            continue

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices = np.vstack(vertices)
    facets = np.vstack(facets).tolist()
    # shapely polygons can include a Z component
    # strip it out for the triangulation
    if vertices.shape[1] == 3:
        vertices = vertices[:, :2]
    result = {"vertices": vertices, "segments": facets}
    # holes in meshpy lingo are a (h, 2) list of (x,y) points
    # which are inside the region of the hole
    # we added a hole for the exterior, which we slice away here
    holes = np.array(holes)[1:]
    if len(holes) > 0:
        result["holes"] = holes
    return result


def cross_2d(a: NDArray, b: NDArray) -> NDArray:
    """
    Numpy 2.0 depreciated cross products of 2D arrays.
    """
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def triangle_area(triangles):
    vectors = np.diff(triangles, axis=1)
    cross = cross_2d(vectors)
    areas = np.sqrt((crosses**2).sum(axis=1)) / 2.0
    return areas.sum()


with open(os.path.join(_cwd, "polygons.json")) as f:
    polygons = [shapely.from_wkt(L) for L in json.load(f)]


def benchmark(iterations: int = 3):
    times = {e: collections.defaultdict(list) for e in engines}

    # check triangulation of both meshpy and triangle engine
    # including an example that has interiors
    for engine in engines:
        print(engine)
        # make sure all our polygons triangulate reasonably
        for i, poly in enumerate(polygons):
            print(poly.area, len(poly.interiors))
            v, f = triangulate_polygon(poly, engine=engine)
            # run asserts
            # check_triangulation(v, f, poly.area)
            try:
                # do a quick benchmark per engine
                r = [
                    i / iterations
                    for i in timeit.repeat(
                        "t(p, engine=e)",
                        repeat=3,
                        number=iterations,
                        globals={
                            "t": triangulate_polygon,
                            "p": poly,
                            "e": engine,
                        },
                    )
                ]
                times[engine][i] = r
            except BaseException:
                log.error("failed to benchmark triangle", exc_info=True)
    log.info(f"benchmarked triangulation on {len(polygons)} polygons: {times!s}")

    return times


if __name__ == "__main__":
    """
    overall stats
       
    """

    b = benchmark()

    stats = {k: np.array([min(t) for t in v.values()]) for k, v in b.items()}

    metrics = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "median": lambda x: np.median(x),
        "total": lambda x: x.sum(),
    }

    results = {}
    for metric, func in metrics.items():
        item = {k: func(s) for k, s in stats.items()}
        results[metric] = {k: v / min(item.values()) for k, v in item.items()}

    mark = ["| METRIC |" + " | ".join(engines) + " |"]
    mark.append("| -- | -- | -- | -- |")

    for m, vs in results.items():
        mark.append(f"| {m} |" + " | ".join(f" {vs[e]:0.4f}" for e in engines) + " |")

    report = "Here are the metrics, normalized against the fastest. As you can tell `mapbox_earcut` wins on performance, i.e. for the mean time `earcut` is 1.0000 and manifold is 32.56 that means the mean time of manifold is 32.56x that of earcut."

    report += "\n\n" + "\n".join(mark)

    with open("results.md", "w") as f:
        f.write(report)
