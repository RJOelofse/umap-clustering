# A Projection Pursuit Clustering Algorithm Incorporating Uniform Manifold Approximation and Projection

## Getting Started

Download or clone this repository: https://github.com/RJOelofse/umap-clustering.

Install as package
```
pip install path/to/umap-clustering
```

To use umap plotting and clustering plotting functionality, install the plotting dependencies
```
pip install umap-clustering[plot]
```

## Example

An illustrative example of using the algorithm on the MNIST dataset is shown below.

```

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_openml
import umappc
import umappc.plot

mnist = fetch_openml('mnist_784')

mapper = umappc.UMAP(
    n_neighbors=15,
    n_components=2,
    n_epochs=400,
    learning_rate=0.1,
    learning_rate_decay=False,
    min_dist=0,
    negative_sample_rate=5,
    clustering=True,
    learning_rate_clustering=0.1,
    learning_rate_decay_clustering=False,
    n_cluster_cycles=3,
    find_umap_embedding=True,
    start_from_init_embedding=False,
    cluster_init='kmeans++',
    n_clusters=10,
    lagrange=1,
).fit(mnist.data)

dpi = plt.rcParams["figure.dpi"]
fig, axes = plt.subplots(figsize=(800 / dpi, 800 / dpi))
umappc.plot.points(
    mapper,
    points=mapper.embedding_,
    labels=mnist.target, ax=axes, use_datashader=False)
umappc.plot.plot_kmeans_voronoi_cells(axes,
                                      mapper.embedding_,
                                      10,
                                      mapper.final_cluster_centers)
plt.show()

ari = metrics.adjusted_rand_score(mnist.target, mapper.final_cluster_labels)
ami = metrics.adjusted_mutual_info_score(mnist.target, mapper.final_cluster_labels)

```

## UMAP Documentation

For further documentation on UMAP, please refer to the original UMAP repository: https://github.com/lmcinnes/umap
