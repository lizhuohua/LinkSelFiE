from multiprocessing.pool import Pool

from evaluation import (plot_cost_vs_gap, plot_cost_vs_path_num,
                        plot_error_vs_path_num)
from utils import set_random_seed

if __name__ == '__main__':
    set_random_seed(12)

    # Run in parallel
    p = Pool(4)
    noise_model_names = ["Depolar", "Dephase", "AmplitudeDamping", "BitFlip"]
    algorithm_names = ["Vanilla NB", "Succ. Elim. NB", "LinkSelFiE"]

    path_num_list = [2, 3, 4, 5, 6, 7]  # For cost vs path num
    path_num_list2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # For error vs path num
    gap_list = [0.025 + 0.02 * i for i in range(1, 7)]
    results = []
    for noise_model in noise_model_names:
        results.append(p.apply_async(plot_cost_vs_path_num, args=(path_num_list, algorithm_names, noise_model, 20)))
        results.append(p.apply_async(plot_cost_vs_gap, args=(4, gap_list, algorithm_names, noise_model, 20)))
        results.append(p.apply_async(plot_error_vs_path_num, args=(path_num_list2, algorithm_names, noise_model, 10)))
    p.close()
    p.join()
    for r in results:
        r.get()
