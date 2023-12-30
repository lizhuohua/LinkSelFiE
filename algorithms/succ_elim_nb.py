import math


def successive_elimination_network_benchmarking(network, path_list, bounces):
    # Initialization
    L = len(path_list)
    active_set = path_list
    C = 0.15
    N = 4
    cost = 0
    delta = 0.1
    sample_times = {}
    for i in bounces:
        sample_times[i] = N

    mean = {path: 0 for path in path_list}
    n = {path: 0 for path in path_list}
    t = 0
    while len(active_set) > 1:
        t += 1
        ucb = {}
        lcb = {}
        for path in active_set:
            p, bounces_num = network.benchmark_path(path, bounces, sample_times)
            if p > 1.5:
                print(f"Get an abnormal p={p}")
            cost += bounces_num
            mean[path] = (mean[path] * n[path] + p) / (n[path] + 1)
            n[path] += 1
            r = C * math.sqrt(math.log(4 * L * t * t / delta) / (2 * t))
            # print(f"r={r}, {math.log(4 * L * t * t / delta)}")
            ucb[path] = mean[path] + r
            lcb[path] = mean[path] - r
            # print(f"mean[{path}] = {mean[path]}")
            # print(f"ucb[{path}] = {ucb[path]}")
            # print(f"lcb[{path}] = {lcb[path]}")
        new_active_set = []
        for path1 in active_set:
            ok = True
            for path2 in active_set:
                if path1 != path2 and ucb[path1] < lcb[path2]:
                    ok = False
                    break
            if ok:
                new_active_set.append(path1)

        active_set = new_active_set
        # print(f"Length of active set: {len(active_set)}")

    assert len(active_set) == 1
    best_path = active_set[0]
    correctness = best_path == network.best_path
    # print(f"Succ Elim NB: Best path: {best_path}, estimated parameter p: {mean[best_path]}, cost: {cost}")
    estimated_fidelity = {}
    for path in path_list:
        p = mean[path]
        # Convert the estimated depolarizing parameter `p` into fidelity
        estimated_fidelity[path] = p + (1 - p) / 2
    best_path_fidelity = estimated_fidelity[best_path]
    return correctness, cost, best_path_fidelity
