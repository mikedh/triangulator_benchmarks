Here are the metrics, normalized against the fastest. As you can tell `mapbox_earcut` wins on performance, i.e. for the mean time `earcut` is 1.0000 and manifold is 31.9110 that means the mean time of manifold is more than 30x slower than earcut for the triangulation. The manifold one is almost certainly more robust than earcut, and triangle almost certainly produces meshes more usable for analysis tasks. So in conclusion, hmm.

| METRIC | triangle | manifold | earcut |
| -- | -- | -- | -- |
| mean | 1.7481 |  31.9110 |  1.0000 |
| std | 1.3322 |  34.1918 |  1.0000 |
| median | 6.6889 |  2.1765 |  1.0000 |
| total | 1.7481 |  31.9110 |  1.0000 |
