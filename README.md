# LinkSelFiE: Link Selection and Fidelity Estimation in Quantum Networks

This repository contains the source code for reproducing the results of our INFOCOM'24 paper titled *LinkSelFiE: Link Selection and Fidelity Estimation in Quantum Networks*.

## Prerequisites

To get started, ensure you have the following packages installed:

[NetSquid](https://netsquid.org/), scipy, matplotlib

## Repository Structure

* [algorithms](./algorithms): Implementation of various link selection & fidelity estimation algorithms.
    * [naive_nb.py](./algorithms/naive_nb.py): The naive algorithm based on network benchmarking.
    * [online_nb.py](./algorithms/online_nb.py): Our proposed LinkSelFiE algorithm.
    * [succ_elim_nb.py](./algorithms/succ_elim_nb.py): A successive elimination-based network benchmarking algorithm.
* [evaluation.py](./evaluation.py): Script to visualize evaluation results and generate figures in the paper.
* [nb_protocol.py](./nb_protocol.py): Implementation of the network benchmarking protocol.
* [network.py](./network.py): Builds the quantum network structure for the experiments.
* [utils](./utils.py): A collection of helper functions.

## How to Run

Execute the following command to reproduce all the figures in the paper:

```sh
python main.py
```

## License

See [LICENSE](LICENSE)
