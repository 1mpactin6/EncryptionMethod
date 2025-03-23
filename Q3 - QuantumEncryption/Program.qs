namespace QuantumEncryption {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Random;
    open Microsoft.Quantum.Logical;

    /// Converts an integer to a boolean array of specified size
    function IntToBoolArray(number : Int, size : Int) : Bool[] {
        mutable result = [false, size = size];
        mutable tempNum = number;
        
        for i in 0..size-1 {
            set result w/= i <- tempNum % 2 == 1;
            set tempNum = tempNum / 2;
        }
        return result;
    }

    /// Converts a boolean array to an integer
    function BoolArrayToInt(bits : Bool[]) : Int {
        mutable result = 0;
        for i in 0..Length(bits)-1 {
            if bits[i] {
                set result = result + PowI(2, i);
            }
        }
        return result;
    }

    /// Generates a quantum random key using quantum superposition and measurement
    operation GenerateQuantumKey(keyLength : Int) : Result[] {
        Message($"\nDebug: Starting quantum key generation for length {keyLength}");
        mutable key = [];
        set key = [Zero, size = keyLength];
        
        for i in 0..keyLength-1 {
            use qubit = Qubit();
            Message($"Debug: Generating bit {i + 1} of {keyLength}");
            
            // Put qubit in superposition
            H(qubit);
            Message("Debug: Applied Hadamard gate");
            
            // Measure the qubit to get a random bit
            set key w/= i <- M(qubit);
            Message($"Debug: Measured qubit {i}: {key[i]}");
            Reset(qubit);
        }
        
        Message($"Debug: Completed key generation: {key}");
        return key;
    }

    /// Applies quantum encryption to a classical bit using a quantum key bit
    operation QuantumEncryptBit(message : Bool, keyBit : Result) : Result {
        Message($"\nDebug: Encrypting bit - Message: {message}, Key bit: {keyBit}");
        use qubit = Qubit();
        
        // Initialize qubit based on message
        if message {
            X(qubit);
            Message("Debug: Applied X gate for message bit 1");
        }
        
        // XOR with key bit using CNOT-like operation
        if keyBit == One {
            // For key bit 1, flip the message bit
            X(qubit);
            Message("Debug: Applied X gate for key bit One (XOR operation)");
        }
        
        let result = M(qubit);
        Message($"Debug: Encryption result: {result}");
        Reset(qubit);
        return result;
    }

    /// Applies quantum decryption to an encrypted bit using the same quantum key bit
    operation QuantumDecryptBit(encrypted : Result, keyBit : Result) : Bool {
        Message($"\nDebug: Decrypting bit - Encrypted: {encrypted}, Key bit: {keyBit}");
        use qubit = Qubit();
        
        // Initialize qubit based on encrypted value
        if encrypted == One {
            X(qubit);
            Message("Debug: Applied X gate for encrypted bit One");
        }
        
        // XOR with key bit using CNOT-like operation (same as encryption)
        if keyBit == One {
            // For key bit 1, flip the bit
            X(qubit);
            Message("Debug: Applied X gate for key bit One (XOR operation)");
        }
        
        let result = M(qubit) == One;
        Message($"Debug: Decryption result: {result}");
        Reset(qubit);
        return result;
    }

    /// Encrypts an array of integers using quantum operations
    operation EncryptData(data : Int[]) : (Result[], Result[]) {
        mutable allBits = [];
        
        // Convert each integer to 8 bits
        for number in data {
            set allBits = allBits + IntToBoolArray(number, 8);
        }
        
        let keyLength = Length(allBits);
        
        // Generate quantum key
        let quantumKey = GenerateQuantumKey(keyLength);
        
        // Encrypt bits
        mutable encryptedBits = [];
        set encryptedBits = [Zero, size = keyLength];
        
        for i in 0..keyLength-1 {
            set encryptedBits w/= i <- QuantumEncryptBit(allBits[i], quantumKey[i]);
        }
        
        return (encryptedBits, quantumKey);
    }

    /// Decrypts encrypted data using the quantum key
    operation DecryptData(encryptedBits : Result[], quantumKey : Result[]) : Int[] {
        let keyLength = Length(encryptedBits);
        mutable decryptedBits = [];
        set decryptedBits = [false, size = keyLength];
        
        // Decrypt all bits
        for i in 0..keyLength-1 {
            set decryptedBits w/= i <- QuantumDecryptBit(encryptedBits[i], quantumKey[i]);
        }
        
        // Convert bits back to integers (8 bits per integer)
        let numInts = keyLength / 8;
        mutable result = [];
        
        for i in 0..numInts-1 {
            let intBits = decryptedBits[i * 8..(i + 1) * 8 - 1];
            let number = BoolArrayToInt(intBits);
            set result = result + [number];
        }
        
        return result;
    }

    /// Entangles two qubits and measures them to generate correlated random bits
    operation GenerateEntangledBits() : (Result, Result) {
        Message("\nDebug: Generating entangled qubits");
        use (q1, q2) = (Qubit(), Qubit());
        
        // Create Bell state (maximally entangled state)
        H(q1);
        Message("Debug: Applied H gate to first qubit");
        CNOT(q1, q2);
        Message("Debug: Applied CNOT gate to entangle qubits");
        
        // Measure both qubits
        let r1 = M(q1);
        let r2 = M(q2);
        Message($"Debug: Measured entangled qubits: ({r1}, {r2})");
        
        Reset(q1);
        Reset(q2);
        return (r1, r2);
    }

    @EntryPoint()
    operation Main() : Unit {
        Message("\n=== Quantum Encryption System Demo ===\n");
        
        // Demo with sample data (ASCII values for "Hi!")
        let message = [72, 105, 33];  // ASCII for "Hi!"
        Message($"\nOriginal Message (ASCII): {message}");
        
        // Encrypt the data
        let (encryptedBits, quantumKey) = EncryptData(message);
        Message($"\nEncrypted Bits: {encryptedBits}");
        Message($"Quantum Key: {quantumKey}");
        
        // Decrypt the data
        let decryptedData = DecryptData(encryptedBits, quantumKey);
        Message($"\nDecrypted Data (ASCII): {decryptedData}");
        
        // Verify if decryption was successful
        mutable success = true;
        for i in 0..Length(message)-1 {
            if message[i] != decryptedData[i] {
                set success = false;
            }
        }
        
        if success {
            Message("\nDecryption successful! Data matches perfectly.");
        } else {
            Message("\nWarning: Decryption mismatch!");
        }
        
        // Demonstrate entanglement-based key generation
        Message("\n=== Quantum Entanglement Demo ===");
        let (bit1, bit2) = GenerateEntangledBits();
        Message($"\nEntangled bits: ({bit1}, {bit2})");
        if (bit1 == bit2) {
            Message("Correlation check: Bits are correlated!");
        } else {
            Message("Correlation check: Unexpected - Bits are not correlated");
        }
    }
}
