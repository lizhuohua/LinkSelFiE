from .naive_nb import naive_network_benchmarking  # noqa: F401
from .online_nb import online_network_benchmarking  # noqa: F401
from .succ_elim_nb import \
    successive_elimination_network_benchmarking  # noqa: F401


def benchmark_using_algorithm(network, path_list, algorithm_name, bounces, sample_times):
    if algorithm_name == "Vanilla NB":
        return naive_network_benchmarking(network, path_list, bounces, sample_times)
    elif algorithm_name == "LinkSelFiE":
        return online_network_benchmarking(network, path_list, bounces)
    elif algorithm_name == "Succ. Elim. NB":
        return successive_elimination_network_benchmarking(network, path_list, bounces)
    else:
        print("Error: Unknown algorithm name")
        exit(1)
