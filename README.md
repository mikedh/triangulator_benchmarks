# triangulator_benchmarks

  A corpus of polygons in the WKT format and a somewhat apples-to-apples benchmark of triangulation options for Python. `pip install manifold3d triangle mapbox_earcut trimesh` should be enough (trimesh is only for `grouping.unique_rows`, which `triangle` needs to not segfault on duplicate points).

Here are the time metrics normalized against the fastest. As you can tell `mapbox_earcut` wins on performance, i.e. for the mean time `earcut` is 1.0000 and if manifold is 30.0 that means the mean time of manifold is 30x slower than that of earcut.

| TIME METRIC |shapely | triangle | manifold | earcut |
| -- | -- | -- | -- | -- |
| mean | 19.6280 |  1.6204 |  32.2106 |  1.0000 |
| std | 14.3287 |  1.2129 |  34.3223 |  1.0000 |
| median | 39.5365 |  6.6337 |  2.1846 |  1.0000 |
| total | 19.6280 |  1.6204 |  32.2106 |  1.0000 |
| max | 13.5885 |  1.2316 |  34.1605 |  1.0000 |
| min | 1.2904 |  4.7891 |  1.2192 |  1.0000 |

Triangulation quality and error is a "whole thing" that I don't have a great amount of insight into or opinions on. This is measuring the simplest possible thing I could think of: the difference between triangle sum area and polygon area normalized to polygon area. These values are per-polygon and probably the most relevant value is `max`.

| ERROR METRIC |shapely | triangle | manifold | earcut |
| -- | -- | -- | -- | -- |
| mean | 0.0892 |  0.0000 |  0.0000 |  0.0000 |
| std | 0.1497 |  0.0000 |  0.0001 |  0.0000 |
| median | 0.0348 |  0.0000 |  0.0000 |  0.0000 |
| total | 3.9245 |  0.0000 |  0.0006 |  0.0000 |
| max | 0.7848 |  0.0000 |  0.0003 |  0.0000 |
| min | 0.0000 |  0.0000 |  0.0000 |  0.0000 |