namespace QuantumDemo {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    

    @EntryPoint()
    operation CreateSuperposition() : Result {
        use qubit = Qubit();  // Allocate a qubit
        
        Message("Creating quantum superposition...");
        
        // Put the qubit in superposition
        H(qubit);
        
        // Measure the qubit
        let result = M(qubit);
        
        // Make sure we reset the qubit to |0⟩ before releasing
        Reset(qubit);
        
        Message($"Measurement result: {result}");
        return result;
    }
}
