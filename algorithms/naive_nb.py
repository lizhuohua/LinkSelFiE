import numpy as np


def naive_network_benchmarking(network, path_list, bounces, sample_times):
    '''Perform vanilla network benchmarking for each path in the `path_list`.
    Return a tuple: (a list of fidelities, the total number of bounces).
    '''
    fidelity = {}
    cost = 0
    for path in path_list:
        p, bounces_num = network.benchmark_path(path, bounces, sample_times)
        # fidelity.append(result)
        fidelity[path] = p + (1 - p) / 2  # Convert the estimated depolarizing parameter `p` into fidelity
        cost += bounces_num
    # print("Estimated fidelity:", fidelity)
    # best_path = np.argmax(fidelity) + 1
    best_path = max(fidelity, key=fidelity.get)
    correctness = best_path == network.best_path
    best_path_fidelity = fidelity[best_path]
    return correctness, cost, best_path_fidelity
