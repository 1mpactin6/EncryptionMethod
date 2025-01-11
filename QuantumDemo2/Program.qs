namespace QuantumDemo2 {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;

    @EntryPoint()
    operation QuantumSuperposition() : Unit {
        Message("Starting quantum superposition demo...");
        
        // Create multiple trials to demonstrate quantum behavior
        for trial in 1..10 {
            use qubit = Qubit();  // Allocate a qubit
            
            Message($"\nTrial {trial}:");
            
            // Initially qubit is in |0⟩ state
            Message("Initial state: |0⟩");
            
            // Apply Hadamard gate to create superposition
            H(qubit);
            Message("Applied H gate - qubit is now in superposition");
            
            // Measure the qubit - it will collapse to either 0 or 1
            let result = M(qubit);
            Message($"Measured: {result}");
            
            // Reset the qubit before releasing
            Reset(qubit);
        }
        
        Message("\nDemo completed!");
    }
}
