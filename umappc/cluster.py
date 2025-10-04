import numpy as np

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import KMeans


def init_clusters(
        embedding,
        n_clusters,
        seed,
        cluster_init="kmeans++",
        kmeans_n_init=None
):
    """Initialize the cluster labels (cluster assignments) and cluster centers for the given embedding.

    Parameters
    ----------
    embedding: array of shape (n_samples, n_components)
        The embedding for the current cluster epoch of UMAP-PPC.

    n_clusters: int
        The number of clusters

    seed: int
        Seed for random number generation. Used by sklearn KMeans to initialize cluster centroids,
        or by numpy to initialize the random generator for cluster assignments.

    cluster_init: string (optional, default 'kmeans++')
        The method for cluster initialization. The options are:
            * 'kmeans': Choose ``n_clusters`` observations at random from the data for the initial
                centroids. Then, the k-means algorithm is run with these initial centroids.
                Uses ``kmeans_n_init`` = 10 by default.
            * 'kmeans++': Selects initial cluster centroids using sampling based on an empirical
                probability distribution of the points' contribution to the overall inertia.
                Then, the k-means algorithm is run with these initial centroids.
                Uses ``kmeans_n_init`` as provided.
            * 'random': Assign each observation to a random cluster using a uniform distribution.
                Cluster centers are then computed as the mean of the points assigned.
                Uses ``kmeans_n_init`` = 1 by default.

    kmeans_n_init : int (optional, default 1 or 10)
        Number of times the k-means algorithm will be run with different centroid seeds. The final
        results will be the best output of ``kmeans_n_init`` consecutive runs in terms of inertia.
        If ``cluster_init`` is 'kmeans', then ``kmeans_n_init`` is set to 10 by default for more
        stable results. If ``cluster_init`` is 'random', ``kmeans_n_init`` is set to 1 by default.

    Returns
    -------
    The clusters labels for the embedding, the ``n_clusters`` cluster centers and calculated inertia.
    """
    if kmeans_n_init is None:
        if cluster_init == "kmeans":
            kmeans_n_init = 10
        elif cluster_init == "kmeans++":
            kmeans_n_init = 1
        elif cluster_init == "random":
            kmeans_n_init = 1

    match cluster_init:
        case "kmeans":
            return fit_kmeans(embedding, n_clusters, "random", kmeans_n_init, seed)
        case "kmeans++":
            return fit_kmeans(embedding, n_clusters, "k-means++", kmeans_n_init, seed)
        case "random":
            n_samples = embedding.shape[0]
            rng_generator = np.random.default_rng(seed=seed)

            # Ensure each cluster gets at least one sample
            initial_labels = np.arange(n_clusters)
            remaining_labels = rng_generator.integers(low=0, high=n_clusters, size=n_samples - n_clusters)
            cluster_labels = np.concatenate([initial_labels, remaining_labels])
            rng_generator.shuffle(cluster_labels)

            cluster_centers = compute_cluster_centers(embedding, cluster_labels, n_clusters)

            inertia = compute_inertia(embedding, cluster_labels, cluster_centers, n_clusters)
            
            return format_cluster_results(cluster_labels, cluster_centers, inertia)

def optimize_clusters(
        embedding,
        n_clusters,
        seed,
        kmeans_n_init=None
):
    """Optimize the cluster labels (cluster assignments) for the given embedding.

    Parameters
    ----------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding for the current cluster epoch of UMAP-PPC.

    n_clusters: int
        The number of clusters

    seed: int
        Seed for random number generation. Used by sklearn KMeans to initialize cluster centroids.

    kmeans_n_init : int (optional, default 1)
        Number of times the k-means algorithm will be run with different centroid seeds. The final
        results will be the best output of ``kmeans_n_init`` consecutive runs in terms of inertia.

    Returns
    -------
    The clusters labels for the embedding, the ``n_clusters`` cluster centers and calculated inertia.
    """
    if kmeans_n_init is None:
        kmeans_n_init = 1
    return fit_kmeans(embedding, n_clusters, "k-means++", kmeans_n_init, seed)

def fit_kmeans(
        embedding,
        n_clusters,
        init_method,
        n_init,
        seed
):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        random_state=seed
    ).fit(embedding)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    return format_cluster_results(cluster_labels, cluster_centers, inertia)

def compute_cluster_centers(
        embedding,
        cluster_labels,
        n_clusters
):
    return np.array([
        embedding[cluster_labels == i].mean(axis=0)
        for i in range(n_clusters)
    ])

def compute_inertia(
        embedding,
        cluster_labels,
        cluster_centers,
        n_clusters
):
    # Calculate inertia = sum of squared distances to cluster center
    return np.sum([
        np.sum((embedding[cluster_labels == i] - cluster_centers[i]) ** 2)
        for i in range(n_clusters)
    ])

def format_cluster_results(
        cluster_labels,
        cluster_centers,
        inertia
):
    cluster_labels = np.array(cluster_labels, dtype=int)
    cluster_centers = np.array(cluster_centers, dtype=np.float32)
    return cluster_labels, cluster_centers, inertia
