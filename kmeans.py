import numpy as np
import scipy.sparse


def kmeans(peak_maps, n_states_min=1, n_states_max=12, use_eig=False, threshold=0.5,
           random_state=None, n_runs=100, max_err=1e-6, max_iter=100):
    """
    Computes microstates using modified k-means clustering algorithm, template
    maps are updated either using eigenvector or weighted average method
    (both results are nearly identical, second option is notably faster).

    :param peak_maps: data from which to compute microstates, should be
        average-referenced sensor data, and only data points at GFP peaks
    :param n_states_min: minimum number of microstates to be computed
    :param n_states_max: maximum number of microstates to be computed
    :param use_eig: use eigenvector-based method if True, weighted average otherwise
    :param threshold: samples with correlation to a microstate lower than this value,
        will be left unassigned
    :param random_state: for initialization of k-means run
    :param n_runs: number of times to repeat k-means with new initialization
    :param max_err: relative error that has to be reached before convergence
    :param max_iter: max number of iterations of k-means run
    :return: maps_all: dictionary with number of microstates as keys, and
        corresponding template maps as values
    """

    # dimensions of input data
    n_peaks, n_channels = peak_maps.shape

    # compute global field potential GFP
    gfp_values = peak_maps.std(axis=1)
    gfp2 = np.sum(gfp_values ** 2)

    # store best maps for each number of microstates
    maps_all = {}

    # compute template maps with eigenvectors
    if use_eig:

        # compute template maps for different numbers of states
        for n_states in range(n_states_min, n_states_max + 1):

            gev_best = 0.0
            print(f'Calculating for {n_states} microstates ...')

            # run k-means algorithm a certain amount of times
            for run in range(n_runs):

                # randomly select initial template maps
                if isinstance(random_state, int):
                    init_times = np.random.RandomState(random_state + run).choice(n_peaks, size=n_states, replace=False)
                else:
                    init_times = np.random.RandomState(random_state).choice(n_peaks, size=n_states, replace=False)

                maps = peak_maps[init_times]
                maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))  # normalize template maps

                # compute activation between initial template maps and GFP peaks, and determine labels
                activation = maps @ peak_maps.T
                labels = np.argmax(np.abs(activation), axis=0)

                # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
                spat_corr = np.abs(activation[labels, np.arange(n_peaks)]) / (np.sqrt(n_channels) * gfp_values)
                spat_corr[spat_corr < threshold] = 0.0

                # compute global explained variance
                gev = np.sum((gfp_values * spat_corr) ** 2) / gfp2

                # initiate parameters for iteration of modified k-means algorithm
                n_iter = 0
                gev_prev = 0.0001

                # apply k-means algorithm until results stop improving or maximum number of iterations is reached
                while (np.abs((gev - gev_prev) / gev_prev) > max_err) & (n_iter < max_iter):

                    # for each label
                    for label in range(n_states):

                        # get maps assigned to current label
                        label_maps = peak_maps[labels == label]
                        matrix = label_maps.T @ label_maps

                        # new map as eigenvector corresponding to largest eigenvalue
                        _, eigenvector = scipy.sparse.linalg.eigsh(matrix, k=1)
                        maps[label] = eigenvector[:, 0]

                    # compute activation between template maps and GFP peaks, and determine labels
                    activation = maps @ peak_maps.T
                    labels = np.argmax(np.abs(activation), axis=0)

                    # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
                    spat_corr = np.abs(activation[labels, np.arange(n_peaks)]) / (np.sqrt(n_channels) * gfp_values)
                    spat_corr[spat_corr < threshold] = 0.0

                    # compute global explained variance
                    gev_prev = gev
                    gev = np.sum((gfp_values * spat_corr) ** 2) / gfp2

                    n_iter += 1

                # update best template maps
                if gev > gev_best:
                    gev_best = gev
                    maps_best = maps

            # store best maps for
            maps_all[n_states] = maps_best

    # compute template maps with weighted average
    else:

        # compute template maps for different numbers of states
        for n_states in range(n_states_min, n_states_max + 1):

            gev_best = 0.0
            print(f'Calculating for {n_states} microstates ...')

            # run k-means algorithm a certain amount of times
            for run in range(n_runs):

                # randomly select initial template maps
                if isinstance(random_state, int):
                    init_times = np.random.RandomState(random_state + run).choice(n_peaks, size=n_states, replace=False)
                else:
                    init_times = np.random.RandomState(random_state).choice(n_peaks, size=n_states, replace=False)

                maps = peak_maps[init_times]
                maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))  # normalize template maps

                # compute activation between initial template maps and GFP peaks, and determine labels
                activation = maps @ peak_maps.T
                labels = np.argmax(np.abs(activation), axis=0)

                # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
                spat_corr = np.abs(activation[labels, np.arange(n_peaks)]) / (np.sqrt(n_channels) * gfp_values)
                spat_corr[spat_corr < threshold] = 0.0

                # compute global explained variance
                gev = np.sum((gfp_values * spat_corr) ** 2) / gfp2

                # initiate parameters for iteration of modified k-means algorithm
                n_iter = 0
                gev_prev = 0.0001

                # apply k-means algorithm until results stop improving or maximum number of iterations is reached
                while (np.abs((gev - gev_prev) / gev_prev) > max_err) & (n_iter < max_iter):
                    # create mask and apply to activation
                    mask = np.zeros(activation.shape, dtype=bool)
                    mask[labels, np.arange(n_peaks)] = True
                    mask *= np.tile(spat_corr > threshold, (n_states, 1))
                    activation[~mask] = 0.0

                    # create new average maps
                    maps = activation @ peak_maps
                    maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))

                    # compute activation between template maps and GFP peaks, and determine labels
                    activation = maps @ peak_maps.T
                    labels = np.argmax(np.abs(activation), axis=0)

                    # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
                    spat_corr = np.abs(activation[labels, np.arange(n_peaks)]) / (np.sqrt(n_channels) * gfp_values)
                    spat_corr[spat_corr < threshold] = 0.0

                    # compute global explained variance
                    gev_prev = gev
                    gev = np.sum((gfp_values * spat_corr) ** 2) / gfp2

                    n_iter += 1

                # update best template maps
                if gev > gev_best:
                    gev_best = gev
                    maps_best = maps

            # store best maps for
            maps_all[n_states] = maps_best

    return maps_all
