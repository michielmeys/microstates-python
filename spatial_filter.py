import scipy.spatial
import numpy as np


def spatial_filter(data, pos, n_neighbors=6):
    """
    Non-linear filter to suppress outliers and spatially smoothen sensor data.
    At any time point: each electrode value along with its 'n_neighbors' closest neighbors are considered.
    Electrodes with minimal and maximal value are discarded.
    New value is weighted mean of remaining electrodes, with weights as inverse distance to central electrode.

    :param data: sensor data to be filtered (n_samples x n_sensors)
    :param pos: three-dimensional coordinates of sensors (n_sensors X 3)
    :param n_neighbors: number of closest sensors to consider for filtering
    :return: spatially filtered data
    """

    # number of channels
    n_channels = data.shape[1]

    # compute distances between all sensors
    dist = scipy.spatial.distance.cdist(pos, pos)
    neighbors = np.argsort(dist, axis=1)[:, :n_neighbors + 1]  # 'n_neighbors' closest sensors
    dist = np.sort(dist, axis=1)[:, :n_neighbors + 1]
    dist /= np.mean(dist[:, 1])  # normalize distances
    dist[:, 0] = 1.  # sensors themselves get weight of one

    # indices for iterating over rows / sensors
    row_idx = np.repeat(list(range(n_channels)), n_neighbors - 1).reshape(n_channels, n_neighbors - 1)

    # iterate over EEG samples and apply spatial smoothing
    for sample_idx, sample in enumerate(data):
        s = sample[neighbors]
        col_idx = np.argsort(s)[:, 1:-1]
        v = s[row_idx, col_idx]  # values of closest sensors
        d = dist[row_idx, col_idx]  # distances to closest sensors
        data[sample_idx] = np.sum(v / d, axis=1) / np.sum(1 / d, axis=1)

    return data
