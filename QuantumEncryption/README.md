# Quantum-Enhanced Encryption System

This project implements a hybrid classical-quantum encryption system that leverages quantum mechanical properties for enhanced security.

## Features

1. **Quantum Key Generation**
   - Uses quantum superposition to generate truly random encryption keys
   - Implements Hadamard gates and random rotations for increased entropy
   - Keys are generated using quantum measurement

2. **Quantum Encryption**
   - Combines classical encryption with quantum operations
   - Uses multiple quantum gates (H, Z, T) for complex transformations
   - Implements quantum state manipulation for secure encryption

3. **Quantum Decryption**
   - Applies inverse quantum operations for decryption
   - Maintains quantum state coherence during operations
   - Recovers original message with perfect fidelity

4. **Quantum Entanglement**
   - Demonstrates quantum entanglement for correlated key generation
   - Uses Bell states for maximally entangled qubits
   - Shows quantum correlation properties

## Security Features

- Quantum superposition for true randomness
- Multiple quantum gates for complex transformations
- Entanglement-based key generation
- No-cloning theorem protection
- Quantum measurement collapse protection

## Requirements

- .NET 6.0 SDK
- Microsoft Quantum Development Kit
- Q# language support

## Usage

1. Build the project:
   ```
   dotnet build
   ```

2. Run the demo:
   ```
   dotnet run
   ```

## How It Works

1. **Key Generation**
   - Creates qubits in superposition
   - Applies random rotations
   - Measures quantum states for random bits

2. **Encryption Process**
   - Converts classical bits to quantum states
   - Applies quantum operations based on key
   - Measures final state for encrypted output

3. **Decryption Process**
   - Recreates quantum states
   - Applies inverse quantum operations
   - Recovers original classical bits

4. **Entanglement Demo**
   - Creates Bell pairs
   - Demonstrates quantum correlation
   - Shows non-classical behavior

## Security Considerations

- Keys are quantum-generated for true randomness
- Multiple quantum operations increase complexity
- Quantum measurement prevents copying
- Entanglement provides additional security layer

## Future Enhancements

1. Implement longer key lengths
2. Add multiple encryption layers
3. Incorporate quantum error correction
4. Add quantum authentication
5. Implement quantum key distribution protocols
