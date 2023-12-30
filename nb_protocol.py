import random as rd

import netsquid as ns
import numpy as np
from netsquid.protocols import NodeProtocol
from netsquid.qubits import QFormalism, ketstates
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi
from netsquid.qubits.cliffords import local_cliffords
from netsquid.qubits.operators import Operator
from netsquid.qubits.qubitapi import create_qubits, measure, operate
from scipy.optimize import curve_fit

CLIFFORD_OPERATORS = [Operator(op.name, op.arr) for op in local_cliffords]
ns.set_qstate_formalism(QFormalism.DM)


def EXP(x, p, A):
    return A * p**(2 * x)


def REGRESSION(bounces, mean_bm):
    popt_AB, pcov_AB = curve_fit(EXP, bounces, mean_bm, p0=[0.9, 0.5], maxfev=100000)
    # print("poptAB", popt_AB[0])
    return [popt_AB[0], popt_AB[1]]


def GET_FIDELITY(info_qubit, gates):
    [ref_qubit1, ref_qubit2] = create_qubits(2)
    # qubitapi.assign_qstate(ref_qubit1, ketstates.s1)
    # qubitapi.assign_qstate(ref_qubit2, ketstates.s1)
    operate(ref_qubit2, ops.X)
    for gate_instr in gates:
        operate(ref_qubit1, gate_instr)
        operate(ref_qubit2, gate_instr)
    fidelity = qubitapi.exp_value(
        info_qubit, ops.Operator("ref", (ns.qubits.reduced_dm(ref_qubit1) - ns.qubits.reduced_dm(ref_qubit2)) / 2))
    # fidelity = abs(fidelity)
    # if fidelity < 0:
    #     print("NEGATIVE", fidelity)
    #     # print("A:", ns.qubits.reduced_dm(ref_qubit1))
    #     # print("B:", ns.qubits.reduced_dm(ref_qubit2))
    #     # print("Operator:", (ns.qubits.reduced_dm(ref_qubit1) - ns.qubits.reduced_dm(ref_qubit2)) / 2)
    # # Add Gaussian noise to simulate measurement noise, but we need to make sure fidelity is within a valid range
    # if fidelity < 0.005:
    #     # If the fidelity is too small, adding noise will likely to make it negative, so we skip adding noise
    #     return fidelity
    # while True:
    #     # measure_error = np.random.normal(0, np.sqrt(((1 + fidelity) * (1 - fidelity))) / np.sqrt(1000))
    #     noisy_fidelity = fidelity + np.random.normal(0, 0.015)
    #     if noisy_fidelity >= 0 and noisy_fidelity <= 1:
    #         break

    noisy_fidelity = fidelity + np.random.normal(0, 0.015)
    return noisy_fidelity


def teleport(epr_qubit, info_qubit):
    """Perform teleportation and return two classical bits."""
    operate([epr_qubit, info_qubit], ns.CNOT)
    operate(epr_qubit, ns.H)
    m1, _ = measure(epr_qubit)
    m2, _ = measure(info_qubit)
    return [m1, m2]


def correction(epr_qubit, measurement_results):
    """Perform correction to recover the information qubit."""
    if measurement_results[0]:
        operate(epr_qubit, ns.Z)
    if measurement_results[1]:
        operate(epr_qubit, ns.X)
    return epr_qubit


class NBProtocolAlice(NodeProtocol):
    # bounce: bounce number, type: list
    # num_samples: repetition times for each bounce, type: dict bounce: times

    def __init__(self, node, bounce=[], num_samples={}, qconn=None):
        super().__init__(node)
        self._qconn = qconn
        if isinstance(bounce, list):
            self._bounce_list = bounce
        elif isinstance(bounce, int):
            self._bounce_list = [bounce]
        self._num_samples = num_samples
        self._gates = []  # record the clifford operations we used.
        self._data_record = {}
        self._target_protocol = None
        self._cost = 0  # The totoal number of bounces
        self.add_signal("ALICE_MEASUREMENT_READY")
        self.add_signal("BOB_MEASUREMENT_READY")
        self.add_signal("ENTANGLEMENT_READY")

    def set_target_protocol(self, bob_protocol):
        self._target_protocol = bob_protocol

    def request_ERP(self):
        """Generate an EPR pair by triggering the quantum source of the quantum channel."""
        self._qconn.subcomponents["qsource"].ports["trigger"].tx_input("trigger")
        yield self.await_timer(100)  # Wait for Alice and Bob to receive and store their qubits

    def run(self):
        for current_bounce in self._bounce_list:
            current_max_sample = self._num_samples[current_bounce]
            if current_bounce not in self._data_record:
                self._data_record[current_bounce] = []
            # print("current bounce:", current_bounce, "bounce_list:", self._bounce_list)
            for current_sample in range(current_max_sample):
                # print("current sample:", current_sample)
                self._gates.clear()
                info_qubit = create_qubits(1)[0]
                # qubitapi.assign_qstate(info_qubit, ketstates.s1)
                for _ in range(current_bounce):
                    # Start one bounce

                    # clifford operation to info qubit
                    instr = rd.choice(CLIFFORD_OPERATORS)
                    self._gates.append(instr)
                    operate(info_qubit, instr)

                    # Request an ERP pair
                    self._qconn.subcomponents["qsource"].ports["trigger"].tx_input("trigger")
                    yield self.await_timer(100)  # Wait for Alice and Bob to receive and store their qubits

                    # Extract EPR pair
                    # print(f"At {ns.sim_time()} {self.node} get one EPR pair and starts teleportation")
                    epr_qubit = self.node.qmemory.pop(0)[0]
                    # print('At', ns.sim_time(), "alice's epr pair", epr_qubit.qstate.qrepr)

                    # Teleport info qubit to Bob using the EPR pair
                    measurement_results = teleport(epr_qubit, info_qubit)
                    # msg = MeasurementResult(measurement_results)
                    self.send_signal("ALICE_MEASUREMENT_READY", result=measurement_results)

                    self._qconn.subcomponents["qsource"].ports["trigger"].tx_input("trigger")
                    yield self.await_timer(100)  # Wait for Alice and Bob to receive and store their qubits

                    # epr_qubit = self.node.qmemory.pop(entanglement.source_position)[0]
                    epr_qubit = self.node.qmemory.pop(0)[0]
                    self.send_signal("ENTANGLEMENT_READY")
                    # print('At', ns.sim_time(), "alice's epr pair", epr_qubit.qstate.qrepr)
                    yield self.await_signal(self._target_protocol, "BOB_MEASUREMENT_READY")

                    measurement_results, instrfrombob = self._target_protocol.get_signal_result("BOB_MEASUREMENT_READY")
                    self._gates.append(instrfrombob)
                    info_qubit = correction(epr_qubit, measurement_results)

                    self._cost += 1  # Finish one bounce

                fidelity = GET_FIDELITY(info_qubit, self._gates)
                self._data_record[current_bounce].append(fidelity)
            # print(f"Finished bounce {current_bounce}")
        # result = self.data_processing()
        # print(f"Estimated fidelity: {result}, cost: {self._cost}")

    def data_processing(self):
        raw_data = self._data_record
        bounces = list(raw_data.keys())
        mean_values = [np.mean(raw_data[key]) for key in bounces]
        # if sorted(mean_values)[0] < 0:
        #     print("Some mean value is negative,", mean_values)
        # print('bounces:', bounces)
        # print("mean_values:", mean_values)
        p, _ = REGRESSION(bounces, mean_values)
        return p, self._cost

    def return_data(self):
        raw_data = self._data_record
        print(raw_data)
        # print()
        bounces = list(raw_data.keys())
        mean_values = [np.mean(raw_data[key]) for key in bounces]

        # assert len(mean_values) == len(self._bounce_list)
        # print('bounces:', bounces)
        # print("mean_values:",mean_values)
        return mean_values


class NBProtocolBob(NodeProtocol):

    def __init__(self, node):
        super().__init__(node)
        self.add_signal("ALICE_MEASUREMENT_READY")
        self.add_signal("BOB_MEASUREMENT_READY")
        self.add_signal("ENTANGLEMENT_READY")

    def set_target_protocol(self, alice_protocol):
        self._target_protocol = alice_protocol

    def run(self):
        while True:
            yield self.await_signal(self._target_protocol, signal_label="ALICE_MEASUREMENT_READY")
            measurement_results_from_target = self._target_protocol.get_signal_result("ALICE_MEASUREMENT_READY")
            # entanglement = measurement_result.entanglement
            epr_qubit = self.node.qmemory.pop(0)[0]
            # measurement_results_from_target = measurement_result.measurement_results
            info_qubit = correction(epr_qubit, measurement_results_from_target)

            yield self.await_signal(self._target_protocol, "ENTANGLEMENT_READY")
            # entanglement = self._target_protocol.get_signal_result("ENTANGLEMENT_READY")
            epr_qubit = self.node.qmemory.pop(0)[0]
            # teleport the qubit to the next node Bob
            instr = rd.choice(CLIFFORD_OPERATORS)
            operate(info_qubit, instr)
            measurement_results = teleport(epr_qubit, info_qubit)
            self.send_signal("BOB_MEASUREMENT_READY", result=[measurement_results, instr])
