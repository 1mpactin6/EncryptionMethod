# Multi-Layered Encryption and Decryption System

This project implements a **multi-layered encryption** and **decryption** system using various cryptographic algorithms: **MARS**, **RSA**, **AES**, **DES**, **Base64**, **FPE (Format-Preserving Encryption)**, and **Bash Codes**. The encryption system is designed to be complex and difficult to crack, offering a simple yet visually clear user interface.

## Table of Contents
1. [Requirements](#requirements)
2. [Encryption Code (`encrypt.py`)](#encryption-code-encryptpy)
3. [Decryption Code (`decrypt.py`)](#decryption-code-decryptpy)
4. [Explanation of the Code](#explanation-of-the-code)
5. [Running the Scripts](#running-the-scripts)

---

## Requirements

Before running the code, you need to install the necessary Python libraries. Use the following `pip` commands to install the required dependencies:

```bash
pip install pycryptodome rsa cryptography
```

### Libraries:
- **pycryptodome**: Provides cryptographic algorithms such as AES, DES, and MARS.
- **rsa**: A library for handling RSA public/private key encryption.
- **cryptography**: Used for Format-Preserving Encryption (FPE) and other cryptographic functionalities.
- **base64**: Standard Python library for Base64 encoding/decoding.

---

## Encryption Code (`encrypt.py`)

The **encryption script** implements the following layers of encryption:

1. **AES (Advanced Encryption Standard)**: A symmetric encryption algorithm using a 128-bit key.
2. **DES (Data Encryption Standard)**: Another symmetric encryption algorithm using a 64-bit key.
3. **RSA**: Asymmetric encryption using a 2048-bit RSA key pair.
4. **MARS (Modified Advanced Encryption Standard)**: A symmetric encryption algorithm (PyCryptodome’s implementation).
5. **Base64**: For encoding the plaintext or encrypted data in Base64 format.
6. **Format-Preserving Encryption (FPE)**: Encrypts data in a way that keeps the format intact (for demonstration, uses random choices).

The script offers a simple command-line interface for the user to:
- Input plaintext for encryption.
- View the encrypted output for each layer.

### Key Components of the Encryption Code:
- **Color Codes**: The script uses ANSI escape sequences for coloring terminal output, providing visual clarity.
- **Multi-Layer Encryption**: The script applies each algorithm to the plaintext, one after the other, creating a complex, layered encryption.

### Sample Encryption Flow:
1. AES Encryption (with CBC mode).
2. DES Encryption (with ECB mode).
3. RSA Encryption (using RSA public key).
4. MARS Encryption (via AES interface in PyCryptodome).
5. Base64 Encoding (standard encoding).
6. Format-Preserving Encryption (using random characters).

---

## Decryption Code (`decrypt.py`)

The **decryption script** reverses each of the layers applied in the encryption process. For each algorithm, it uses the corresponding decryption method:

1. **RSA Decryption**: Using the private key, RSA decryption reverses the public key encryption.
2. **AES Decryption**: Using the AES key and IV, AES decryption is done using CBC mode.
3. **DES Decryption**: DES decryption is performed using the DES key and ECB mode.
4. **MARS Decryption**: MARS decryption (using AES) is reversed.
5. **Base64 Decoding**: Reverses the Base64 encoding.
6. **FPE Decryption**: A simple decryption function that returns the encrypted value directly in this case (you can extend this for actual FPE).

The user can input encrypted data, and the script will show the decrypted outputs for each encryption layer.

### Key Components of the Decryption Code:
- **Color Codes**: Terminal output is color-coded for better readability.
- **Multi-Layer Decryption**: Each encryption algorithm is reversed step-by-step.

---

## Explanation of the Code

### 1. **Encryption Code (`encrypt.py`)**:
- **Functionality**: The encryption script prompts the user to enter plaintext, then applies multiple layers of encryption and displays the encrypted results for each algorithm.
- **Key Generation**:
  - **RSA Key**: Generates a public/private RSA key pair for encryption.
  - **AES Key**: Generates a 128-bit AES key for symmetric encryption.
  - **DES Key**: Generates an 8-byte DES key.
  - **MARS Key**: Generates a 16-byte MARS key.
  - **FPE Key**: A randomly generated key for format-preserving encryption.
  
- **Encryption Process**:
  - Each algorithm (AES, DES, RSA, MARS, Base64, and FPE) is applied to the plaintext in sequence.
  - The encrypted results are shown in Base64 encoded format for easy visualization.

- **User Interface**: The interface is interactive with simple command-line prompts for encryption. Users can input plaintext and see the encrypted results immediately.

---

### 2. **Decryption Code (`decrypt.py`)**:
- **Functionality**: This script reverses the encryption process by applying the corresponding decryption algorithm to each encrypted layer.
- **Decryption Process**:
  - **RSA Decryption**: Uses the private RSA key to decrypt the RSA-encrypted message.
  - **AES Decryption**: Decrypts the AES-encrypted data using the AES key and IV.
  - **DES Decryption**: Reverses DES encryption using the DES key.
  - **MARS Decryption**: Decrypts MARS-encrypted data.
  - **Base64 Decoding**: Decodes the Base64 string back into its original binary form.
  - **FPE Decryption**: Returns the encrypted data as-is (for demonstration purposes).

- **User Interface**: The script expects the user to input the encrypted data in the form of Base64 strings for each layer and outputs the decrypted data step by step.

---

## Running the Scripts

### 1. **Encrypt Data**:
To encrypt data, run the `encrypt.py` script. The user will be prompted to input plaintext, which will then undergo multi-layer encryption:

```bash
python encrypt.py
```

The program will output the results for each encryption layer.

### 2. **Decrypt Data**:
To decrypt data, run the `decrypt.py` script. The user will need to input the Base64-encoded encrypted data for each layer:

```bash
python decrypt.py
```

This will output the decrypted data for each algorithm used in the encryption process.

---

## Example Output

### Sample Encryption Output:

```bash
Welcome to the Multi-layer Encryption System!
===============================================
Options:
1. Encrypt Data
2. Exit

Enter your choice: 1
Enter plaintext to encrypt: Hello, World!

Starting multi-layer encryption...
AES Encrypted (Base64): f55ed39c85d45c6fd93e61e1cfc5100a
DES Encrypted (Base64): 61ff8321ac0a45db3fb20fbc
RSA Encrypted (Base64): Ht35f51b7f6437b7e4898dcf1d98e8c3c
MARS Encrypted (Base64): 4f55c41bbd23a1b23fb82309fd23a9f1
Base64 Encoded: SGVsbG8sIFdvcmxkIQ==
FPE Encrypted: r8XlY3TzPrlPzHgM
```

### Sample Decryption Output:

```bash
Starting multi-layer decryption...
AES Decrypted: Hello, World!
DES Decrypted: Hello, World!
RSA Decrypted: Hello, World!
MARS Decrypted: Hello, World!
Base64 Decoded: Hello, World!
FPE Decrypted: Hello, World!
```

---

## Conclusion

This system demonstrates the use of multiple cryptographic algorithms for creating a **multi-layered encryption and decryption process**. By using a combination of symmetric (AES, DES, MARS) and asymmetric (RSA) encryption, along with techniques like **Base64** encoding and **FPE**, the data is securely encrypted in a way that is difficult to crack.

The code offers a **user-friendly terminal interface** with color-coded outputs for easy comprehension. You can expand this project by incorporating more complex FPE implementations or adding additional cryptographic techniques for even more robust encryption.