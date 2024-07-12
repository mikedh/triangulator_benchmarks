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
from shapely.geometry import Polygon
import shapely.ops

import trimesh

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

engines = ["shapely", "triangle", "manifold", "earcut"]


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
    elif engine == "shapely":
        tri = shapely.ops.triangulate(polygon)
        # vertices are not merged
        raw = np.array([t.exterior.coords for t in tri], dtype=np.float64)[:, :3, :]

        # note this is only *almost* apples-apples as triangles are 100% disconnected
        # soup that would need a `trimesh.grouping.unique_rows` or something to merge
        vertices = raw.reshape((-1, 2))
        faces = np.arange(len(vertices)).reshape((-1, 3))

        # this slows down the results substantially but is the real apples-apples
        # it had no effect on the error we were seeingg
        # unique, inverse = trimesh.grouping.unique_rows(vertices)
        # faces = inverse[faces]
        # vertices = vertices[unique]
        # show it as a 3D thing and press `w` to see the triangultion
        # trimesh.Trimesh(np.column_stack((vertices, np.zeros(len(vertices)))), faces).show()

        return vertices, faces

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


# load the corpus
with open(os.path.join(_cwd, "polygons.json")) as f:
    polygons = [shapely.from_wkt(L) for L in json.load(f)]


def benchmark(iterations: int = 3):
    times = {e: collections.defaultdict(list) for e in engines}
    error = {e: [] for e in engines}

    # check triangulation of both meshpy and triangle engine
    # including an example that has interiors
    for engine in engines:
        print(engine)
        # make sure all our polygons triangulate reasonably
        for i, poly in enumerate(polygons):
            print(poly.area, len(poly.interiors))
            # now check the area of the source polygon vs the result area
            v, f = triangulate_polygon(poly, engine=engine)

            area_check = trimesh.triangles.area(
                np.column_stack((v, np.zeros(len(v))))[f]
            )

            error[engine].append(np.abs(area_check.sum() - poly.area) / poly.area)

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

    error = {k: np.array(v) for k, v in error.items()}

    return times, error


if __name__ == "__main__":
    """
    overall stats
       
    """

    b, err = benchmark()
    stats = {k: np.array([min(t) for t in v.values()]) for k, v in b.items()}

    metrics = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "median": lambda x: np.median(x),
        "total": lambda x: x.sum(),
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
    }

    results = {}
    for metric, func in metrics.items():
        item = {k: func(s) for k, s in stats.items()}
        results[metric] = {k: v / min(item.values()) for k, v in item.items()}

    mark = ["| TIME METRIC |" + " | ".join(engines) + " |"]
    mark.append("| -- | -- | -- | -- | -- |")

    for m, vs in results.items():
        mark.append(f"| {m} |" + " | ".join(f" {vs[e]:0.4f}" for e in engines) + " |")

    report = "# triangulator_benchmarks\n\n  A corpus of polygons in the WKT format and a somewhat apples-to-apples benchmark of triangulation options for Python. `pip install manifold3d triangle mapbox_earcut trimesh` should be enough (trimesh is only for `grouping.unique_rows`, which `triangle` needs to not segfault on duplicate points).\n\n"

    report += "Here are the time metrics normalized against the fastest. As you can tell `mapbox_earcut` wins on performance, i.e. for the mean time `earcut` is 1.0000 and if manifold is 30.0 that means the mean time of manifold is 30x slower than that of earcut."

    report += "\n\n" + "\n".join(mark)

    report += '\n\nTriangulation quality and error is a "whole thing" that I don\'t have a great amount of insight into or opinions on. This is measuring the simplest possible thing I could think of: the difference between triangle sum area and polygon area normalized to polygon area. These values are per-polygon and probably the most relevant value is `max`.'

    mark = ["| ERROR METRIC |" + " | ".join(engines) + " |"]
    mark.append("| -- | -- | -- | -- | -- |")

    for mn, mf in metrics.items():
        mark.append(
            f"| {mn} |" + " | ".join(f" {mf(err[e]):0.4f}" for e in engines) + " |"
        )

    report += "\n\n" + "\n".join(mark)

    with open("results.md", "w") as f:
        f.write(report)
