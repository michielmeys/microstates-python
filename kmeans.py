import numpy as np


def kmeans(data, n_states_min=1, n_states_max=12, threshold=0.5, n_runs=100,
           max_err=1e-6, max_iter=100, random_state=None):
    """
    Function for computation of microstates using a modified k-means algorithm.

    :param data: sensor data to compute microstates for (n_samples x n_channels)
    :param n_states_min: minimum number of microstates to be computed
    :param n_states_max: maximum number of microstates to be computed
    :param threshold: value between 0 and 1, data points with correlation to
           best microstate lower than this value will not be assigned
    :param n_runs: number of times to repeat k-means algorithm with new initialization
    :param max_err: convergence criterion, threshold for relative error
    :param max_iter: convergence criterion, maximal number of iterations
    :param random_state: for initialization of k-means runs
    :return: dictionary with number of microstates as keys; and maps as values
    """

    # data dimensions
    n_samples, n_channels = data.shape

    # enforce average reference
    data -= np.mean(data, axis=1, keepdims=True)

    # compute global field potential GFP
    gfp = data.std(axis=1)
    gfp2 = np.sum(gfp ** 2)

    # store best maps for each number of microstates
    maps_all = {}
    maps_best = None

    # compute template maps for different numbers of states
    for n_states in range(n_states_min, n_states_max + 1):

        gev_best = 0.0
        print(f'Calculating for {n_states} microstates ...')

        # run k-means algorithm a certain amount of times
        for run in range(n_runs):

            # randomly select initial template maps
            if isinstance(random_state, int):
                init_times = np.random.RandomState(random_state + run).choice(n_samples, size=n_states, replace=False)
            else:
                init_times = np.random.RandomState(random_state).choice(n_samples, size=n_states, replace=False)

            maps = data[init_times]
            maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))  # normalize template maps

            # compute activation between initial template maps and GFP peaks, and determine labels
            activation = maps @ data.T
            labels = np.argmax(np.abs(activation), axis=0)

            # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
            spat_corr = np.abs(activation[labels, np.arange(n_samples)]) / (np.sqrt(n_channels) * gfp)
            spat_corr[spat_corr < threshold] = 0.0

            # compute global explained variance
            gev = np.sum((gfp * spat_corr) ** 2) / gfp2

            # initiate parameters for iteration of modified k-means algorithm
            n_iter = 0
            gev_prev = 0.0001

            # apply modified k-means algorithm until results stop improving or maximum number of iterations is reached
            while (np.abs((gev - gev_prev) / gev_prev) > max_err) & (n_iter < max_iter):

                # create mask and apply to activation
                mask = np.zeros(activation.shape, dtype=bool)
                mask[labels, np.arange(n_samples)] = True
                mask *= np.tile(spat_corr > threshold, (n_states, 1))
                activation[~mask] = 0.0

                # create new average maps
                maps = activation @ data
                maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))

                # compute activation between template maps and GFP peaks, and determine labels
                activation = maps @ data.T
                labels = np.argmax(np.abs(activation), axis=0)

                # compute spatial correlation (ignoring polarity) and set to zero where value is below threshold
                spat_corr = np.abs(activation[labels, np.arange(n_samples)]) / (np.sqrt(n_channels) * gfp)
                spat_corr[spat_corr < threshold] = 0.0

                # compute global explained variance
                gev_prev = gev
                gev = np.sum((gfp * spat_corr) ** 2) / gfp2

                n_iter += 1

            # update best template maps
            if gev > gev_best:
                gev_best = gev
                maps_best = maps

        # store best maps for
        maps_all[n_states] = maps_best

    return maps_all
