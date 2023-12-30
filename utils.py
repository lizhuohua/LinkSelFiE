# The code mainly comes from the tutorial of NetSquid:
# https://docs.netsquid.org/latest-release/tutorial.intro.html

import itertools
import math
import random

import netsquid as ns
import netsquid.qubits.ketstates as ks
import numpy
import numpy as np
from netsquid.components import (DephaseNoiseModel, DepolarNoiseModel,
                                 PhysicalInstruction, QuantumChannel,
                                 QuantumProcessor)
from netsquid.components.instructions import (INSTR_MEASURE_BELL, INSTR_X,
                                              INSTR_Z)
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.nodes.connections import Connection
from netsquid.qubits import StateSampler


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    ns.set_random_state(seed=seed)


def pairwise(iterable):
    """E.g., pairwise([1, 2, 3, 4]) outputs [(1, 2), (2, 3), (3, 4)]
       If input size is less or equal to 1, output []
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def pairs(lst):
    """Iterate over pairs in a list (circular fashion).
    E.g., if `lst=[0, 1, 2, ..., 9]`, the function returns (0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6) (6, 7) (7, 8) (8, 9) (9, 0).
    Reference: https://stackoverflow.com/questions/1257413/iterate-over-pairs-in-a-list-circular-fashion-in-python
    """
    n = len(lst)
    for i in range(n):
        yield lst[i], lst[(i + 1) % n]


def fidelity_to_error_param(f, noise_model):
    if noise_model == "Depolar":
        # Deploarizing channel: E[ρ] = pρ+(1-p)I/2
        # f = p + (1 - p)/2  ==>  p = -1 + 2f
        # The domain of f is [0.5, 1]
        assert f >= 0.5 and f <= 1
        p = -1 + 2 * f
        return 1 - p
    elif noise_model == "Dephase":
        # Dephasing channel: E[ρ] = pρ+(1-p)ZρZ'
        # f = (2 p + 1)/3  ==>  p = 1/2 (-1 + 3 f)
        # The domain of f is [1/3, 1]
        assert f >= 1 / 3 and f <= 1
        p = 0.5 * (-1 + 3 * f)
        return 1 - p
    elif noise_model == "AmplitudeDamping":
        # Amplitude damping channel
        # f = 2/3 - p/6 + Sqrt[1 - p]/3  ==> p = 2 (1 - 3 f + Sqrt[2] Sqrt[-1 + 3 f])
        # The domain of f is [1/2, 1]
        # print("AAAAA", f, 2 * (1 - 3 * f + math.sqrt(2) * math.sqrt(-1 + 3 * f)))
        assert f >= 1 / 2 and f <= 1
        return 2 * (1 - 3 * f + math.sqrt(2) * math.sqrt(-1 + 3 * f))
    elif noise_model == "BitFlip":
        # Bit flip channel: E[ρ] = pρ+(1-p)XρX'
        # f = (2 p + 1)/3  ==>  p = 1/2 (-1 + 3 f)
        # The domain of f is [1/3, 1]
        assert f >= 1 / 3 and f <= 1
        p = 0.5 * (-1 + 3 * f)
        return 1 - p
    else:
        print("Error: Unknown error model")
        exit(1)


class EntanglingConnectionOnDemand(Connection):
    """A connection that generates an entanglement upon receiving a request in port "trigger".

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Parameters
    ----------
    fidelity : float

    """

    # Static variable used in the name of QSource. This guarantees that all the generated qubits' name are distinct.
    qsource_index = 1

    def __init__(self, noise_model, fidelity):
        name = "EntanglingConnection"
        name = name + str(EntanglingConnectionOnDemand.qsource_index)
        EntanglingConnectionOnDemand.qsource_index += 1
        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2, status=SourceStatus.EXTERNAL)
        self.add_subcomponent(qsource, name="qsource")
        self.fidelity = fidelity
        error_parameter = fidelity_to_error_param(fidelity, noise_model)

        if noise_model == "Depolar":
            noise_model = DepolarNoiseModel(error_parameter, time_independent=True)
        elif noise_model == "Dephase":
            noise_model = DephaseNoiseModel(error_parameter, time_independent=True)
        elif noise_model == "AmplitudeDamping":
            noise_model = AmplitudeDampingNoiseModel(error_parameter)
        elif noise_model == "BitFlip":
            noise_model = BitFlipNoiseModel(error_parameter)
        else:
            print("Error: Unknown error model")
            exit(1)

        qchannel_c2a = QuantumChannel("qchannel_C2A", models={"quantum_noise_model": noise_model})
        qchannel_c2b = QuantumChannel("qchannel_C2B")
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])


def create_qprocessor(num_positions, gate_noise_rate=0, mem_noise_rate=0):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    num_positions : int
        The number of qubits that the quantum memory can maintain.

    gate_noise_rate : float
        The probability that quantum operation results will depolarize.

    mem_noise_rate : float
        The probability that qubits stored in quantum memory will depolarize.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    gate_noise_model = DepolarNoiseModel(gate_noise_rate, time_independent=True)
    mem_noise_model = DepolarNoiseModel(mem_noise_rate, time_independent=True)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=1, quantum_noise_model=None),
        PhysicalInstruction(INSTR_Z, duration=1, quantum_noise_model=None),
        # We have to set `apply_q_noise_after=False` to make sure the noise is added before measurement
        # Otherwise the measurement results will be precise
        PhysicalInstruction(INSTR_MEASURE_BELL,
                            duration=7,
                            quantum_noise_model=gate_noise_model,
                            apply_q_noise_after=False),
    ]
    qproc = QuantumProcessor("QuantumProcessor",
                             num_positions=num_positions,
                             fallback_to_nonphysical=False,
                             mem_noise_models=[mem_noise_model] * num_positions,
                             phys_instructions=physical_instructions)
    return qproc


class BitFlipNoiseModel(QuantumErrorModel):
    """Bit Flip Noise Model.

       (1-gamma) * |PHI><PHI| + gamma * X|PHI><PHI|X.
    Parameters
    ----------
    gamma : float
        Bit flip parameter
    Raises
    ------
    ValueError
        If gamma is <0 or >1
    """

    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self._properties.update({'gamma': gamma})
        if gamma < 0:
            raise ValueError("gamma {} is negative".format(self.gamma))
        if gamma > 1:
            raise ValueError("gamma {} is larger than one".format(self.gamma))

        self._properties.update({'gamma': gamma})

    @property
    def gamma(self):
        return self._properties['gamma']

    @gamma.setter
    def gamma(self, value):
        self._properties['gamma'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on component [ns].

        """
        for qubit in qubits:
            self.apply_noise(qubit)

    def apply_noise(self, qubit):
        """Applies noise to the qubit, depending on gamma."""
        # Check whether the memory is empty, if so we do nothing
        if qubit is None:
            return
        # Apply noise
        ns.qubits.qubitapi.apply_pauli_noise(qubit, (1 - self.gamma, self.gamma, 0, 0))


class AmplitudeDampingNoiseModel(QuantumErrorModel):
    """Amplitude Damping Noise model

    Parameters
    ----------
    gamma : float
        Damping parameter

    Raises
    ------
    ValueError
        If gamma is <0 or >1
    """

    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self._properties.update({'gamma': gamma})
        if gamma < 0:
            raise ValueError("gamma {} is negative".format(self.gamma))
        if gamma > 1:
            raise ValueError("gamma {} is larger than one".format(self.gamma))

        self._properties.update({'gamma': gamma})

    @property
    def gamma(self):
        return self._properties['gamma']

    @gamma.setter
    def gamma(self, value):
        self._properties['gamma'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on component [ns].

        """
        for qubit in qubits:
            self.apply_noise(qubit)

    def apply_noise(self, qubit):
        """Applies noise to the qubit, depending on gamma."""
        # Check whether the memory is empty, if so we do nothing
        if qubit is None:
            return
        # Apply noise
        ns.qubits.qubitapi.amplitude_dampen(qubit, gamma=self.gamma, prob=1)
