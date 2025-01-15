import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli
import hashlib
from typing import List, Tuple, Union
import random
import time
from colorama import init, Fore, Style
import base64

init()

# Tips and descriptions for quantum hash operations
TIPS = {
    "encryption": [
        "Quantum superposition enhances hash security",
        "VQE parameters are crucial for uniqueness",
        "Quantum entanglement adds complexity",
        "Keep quantum parameters secure",
        "Higher qubit count increases security"
    ],
    "decryption": [
        "Exact parameters needed for verification",
        "Quantum state verification is sensitive",
        "Check quantum circuit execution",
        "Verify quantum hash integrity",
        "Parameter precision matters"
    ]
}

CONCEPTS = {
    "VQE": "Variational Quantum Eigensolver - quantum algorithm for optimization",
    "Quantum Hash": "Hash function utilizing quantum properties",
    "Quantum Parameters": "Variables controlling quantum circuit behavior",
    "Quantum Verification": "Process of validating quantum hash authenticity",
}

def print_panel(title: str, subtitle: str, color: str) -> None:
    """Print a formatted panel with title and subtitle"""
    width = 60
    print(f"\n{color}{'=' * width}")
    print(f"{title.center(width)}")
    print(f"{subtitle.center(width)}")
    print(f"{'=' * width}{Style.RESET_ALL}")

def print_description(concept: str) -> None:
    """Print concept description"""
    if concept in CONCEPTS:
        print(f"\n{Fore.CYAN}[{concept}]: {Fore.WHITE}{CONCEPTS[concept]}{Style.RESET_ALL}")

def print_random_tip(tips: List[str]) -> None:
    """Print a random tip"""
    tip = random.choice(tips)
    print(f"\n{Fore.YELLOW} Tip: {tip}{Style.RESET_ALL}")

class QuantumHashVQE:
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        try:
            self.backend = Aer.get_backend('statevector_simulator')
        except Exception as e:
            print(f"{Fore.RED}Error initializing quantum backend: {e}{Style.RESET_ALL}")
            raise
        self.optimizer = SPSA(maxiter=100)
        self.parameters = [Parameter(f'θ_{i}') for i in range(num_qubits * 3)]

    def create_ansatz(self) -> QuantumCircuit:
        """Create the variational quantum circuit (ansatz)"""
        qc = QuantumCircuit(self.num_qubits)

        # Layer of Hadamard gates
        for q in range(self.num_qubits):
            qc.h(q)

        # Parameterized rotations
        for i in range(self.num_qubits):
            qc.rx(self.parameters[i*3], i)
            qc.ry(self.parameters[i*3 + 1], i)
            qc.rz(self.parameters[i*3 + 2], i)

        # Entanglement layer
        for i in range(self.num_qubits-1):
            qc.cx(i, i+1)

        return qc

    def create_hamiltonian(self, target_hash: str) -> PauliSumOp:
        """Create a problem Hamiltonian based on the target hash"""
        # Convert hash to binary and pad/truncate to match qubit count
        hash_int = int(target_hash[:self.num_qubits], 16)
        binary = format(hash_int, f'0{self.num_qubits}b')

        # Create Pauli string based on binary
        pauli_str = ''
        for bit in binary:
            pauli_str += 'Z' if bit == '1' else 'I'

        return PauliSumOp(Pauli(pauli_str))

    def encrypt(self, message: str) -> Tuple[str, List[float]]:
        """Encrypt a message using VQE-based quantum hashing"""
        # Create classical hash first
        classical_hash = hashlib.sha256(message.encode()).hexdigest()

        # Create quantum circuit
        qc = self.create_ansatz()
        hamiltonian = self.create_hamiltonian(classical_hash)

        # Run VQE to find optimal parameters
        def objective(params):
            bound_qc = qc.bind_parameters({p: v for p, v in zip(self.parameters, params)})
            result = execute(bound_qc, self.backend).result()
            state = result.get_statevector()
            return np.real(np.dot(state.conj(), hamiltonian.to_matrix() @ state))

        initial_params = np.random.random(len(self.parameters)) * 2 * np.pi
        optimal_params = self.optimizer.optimize(
            num_vars=len(self.parameters),
            objective_function=objective,
            initial_point=initial_params
        )[0]

        # Generate quantum hash using optimal parameters
        quantum_hash = hashlib.sha256(str(optimal_params).encode()).hexdigest()

        return quantum_hash, optimal_params.tolist()

    def decrypt(self, quantum_hash: str, parameters: List[float]) -> bool:
        """Verify a quantum hash using stored parameters"""
        qc = self.create_ansatz()
        bound_qc = qc.bind_parameters({p: v for p, v in zip(self.parameters, parameters)})

        # Execute circuit with stored parameters
        result = execute(bound_qc, self.backend).result()
        state = result.get_statevector()

        # Verify hash
        verification_hash = hashlib.sha256(str(parameters).encode()).hexdigest()
        return verification_hash == quantum_hash

def interactive_menu():
    while True:
        print_panel("QUANTUM HASH SYSTEM", "Welcome to Quantum VQE Hash!", Fore.CYAN)
        print(f"{Fore.YELLOW}1. {Fore.WHITE}Create Quantum Hash\n{Fore.YELLOW}2. {Fore.WHITE}Verify Quantum Hash\n{Fore.YELLOW}3. {Fore.WHITE}Exit")
        choice = input(f"\n{Fore.GREEN}Choice (1-3): {Style.RESET_ALL}")

        qh = QuantumHashVQE(num_qubits=4)

        if choice == '1':
            print_panel("QUANTUM HASH CREATION", "Enter message to hash:", Fore.BLUE)
            message = input(f"{Fore.WHITE}Message: {Style.RESET_ALL}")
            print_description("VQE")
            print_random_tip(TIPS["encryption"])

            try:
                quantum_hash, parameters = qh.encrypt(message)
                print_panel("QUANTUM HASH RESULT", "Generated quantum hash:", Fore.GREEN)
                print(f"{Fore.CYAN}Quantum Hash:{Style.RESET_ALL} {quantum_hash}")
                print(f"{Fore.RED}Quantum Parameters:{Style.RESET_ALL} {base64.b64encode(str(parameters).encode()).decode()}")
                print(f"\n{Fore.YELLOW}Keep these quantum parameters safe for verification!{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}Error during quantum hashing: {e}{Style.RESET_ALL}")

        elif choice == '2':
            print_panel("HASH VERIFICATION", "Enter quantum hash and parameters:", Fore.MAGENTA)
            quantum_hash = input(f"{Fore.WHITE}Quantum Hash: {Style.RESET_ALL}")
            print_description("Quantum Verification")
            print_random_tip(TIPS["decryption"])

            try:
                params_b64 = input(f"{Fore.WHITE}Quantum Parameters (base64): {Style.RESET_ALL}")
                params_str = base64.b64decode(params_b64).decode()
                parameters = eval(params_str)  # Convert string representation back to list

                is_valid = qh.decrypt(quantum_hash, parameters)
                print_panel("VERIFICATION RESULT", "Hash verification complete:", Fore.GREEN)
                if is_valid:
                    print(f"{Fore.GREEN}✓ Quantum hash verified successfully!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}✗ Invalid quantum hash or parameters{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}Error during verification: {e}{Style.RESET_ALL}")

        elif choice == '3':
            print(f"\n{Fore.CYAN}Thank you for using the Quantum Hash System!{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

        input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Program terminated by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
