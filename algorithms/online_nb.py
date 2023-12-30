import math


def online_network_benchmarking(network, path_list, bounces):
    # Initialization
    candidate_set = path_list
    s = 0  # Phase
    C = 0.01  # Constant
    delta = 0.1  # Error
    cost = 0
    estimated_fidelities = {}
    # error_probability = {}  # Map bounces number to the probability that the arm with largest mean is not the best arm
    # epoch_len = 20
    # epoch = 0
    # correct_times = 0
    while len(candidate_set) > 1:
        s += 1
        Ns = math.ceil(C * 2**(2 * s) * math.log2(2**s * len(candidate_set) / delta))
        if Ns < 4:
            Ns = 4
        # print(f"Ns: {Ns}")
        sample_times = {}
        for i in bounces:
            sample_times[i] = Ns

        p_s = {}
        for path in candidate_set:
            p, bounces_num = network.benchmark_path(path, bounces, sample_times)
            # print(f"Estimated Fidelity of path {path}: {p}")
            # Convert the estimated depolarizing parameter `p` into fidelity
            estimated_fidelities[path] = p + (1 - p) / 2
            p_s[path] = p
            cost += bounces_num
        p_max = max(p_s.values())
        # current_best_path = max(p_s, key=p_s.get)
        # if current_best_path == network.best_path:
        #     correct_times += 1
        new_candidate_set = []
        for path in candidate_set:
            # print(f"p_s[path] + 2**(-s): {p_s[path] + 2**(-s)}")
            # print(f"p_max - 2**(-s): {p_max - 2**(-s)}")
            if p_s[path] + 2**(-s) > p_max - 2**(-s):
                new_candidate_set.append(path)
        candidate_set = new_candidate_set

    assert len(candidate_set) == 1
    best_path = candidate_set[0]
    correctness = best_path == network.best_path
    best_path_fidelity = estimated_fidelities[best_path]
    # print(f"Best path: {best_path}, estimated parameter p: {p_s[best_path]}, cost: {cost}")
    return correctness, cost, best_path_fidelity
