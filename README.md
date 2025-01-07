# EncryptionMethod
 A Python code that can encrypt and decrypt any types of binary code (The code will later be updated in different versions)

## Overview

The DecryptionMethod provides an advanced interface for users to encrypt and ecrypt binary codes into custom RSA like code. By implementing a Python-environment users and freely use this tool.

## Features

- Encrypt inputed messages and provide public key.
- Decrypt messages using a provided public key.
- Brute-force decryption by trying different private keys.
- Interactive command-line interface with color-coded output for better readability.
- Detailed explanations of the decryption processes.

- Provides different encryption and decryption methods, currently including:
  - RSA
  - AES

## Requirements

- Python 3.x
- Python Environment

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/1mpactin6/EncryptionMethod.git
   ```

2. Navigate to the project directory:

   ```bash
   cd EncryptionMethod
   ```

3. Ensure you have Python 3 installed on your system. You can check this by running:

   ```bash
   python --version
   ```

4. Ensure you have specific dependencies. You can download it by running:

   ```bash
   pip install pycryptodome
   ```

   and

   ```bash
   pip install rsa
   ```

5. If you already have the dependencies installed. You can upgrade it by:

   ```bash
   pip install --upgrade [dependency]
   ```

## Usage

1. Run the decryption tool:

   ```bash
   python [filename].py
   ```

2. Follow the prompts in the command line:

   - Enter the message you wish to encrypt.
   - Enter the encrypted message you wish to decrypt.
   - Optionally, provide a public key for direct decryption or press Enter to use the brute-force method.

4. The tool will attempt to decrypt the message using the provided public key or by brute-forcing through possible private keys.

5. If brute-force results are found, you will be prompted to either display them or skip to the next step.

6. The output will be color-coded for clarity:
   - **Green**: Explanations and successful decryption messages.
   - **Blue**: Successful decryption outputs.
   - **Red**: Error messages.
   - **Yellow**: Tips and informational messages.

# Encryption and Security Projects Collection

This repository contains a collection of encryption and security-related projects, each focusing on different aspects of cryptography and data security.

## Projects Overview

### 1. RSA Implementation (1-RSA)
Basic implementation of the RSA encryption algorithm.

### 2. AES Implementation (2-AES-Fail)
Implementation of Advanced Encryption Standard (AES) with learning purposes.

### 3. MARS Encryption (3-MARS)
Implementation of the MARS encryption algorithm.

### 4. Multi-Layered EDS (4-Multi-Layered EDS)
A multi-layered encryption and decryption system combining multiple encryption algorithms.

### 5. ML Customizable ES (5-ML Customizable ES)
Machine Learning-based customizable encryption system with adaptable security levels.

### 6. MLC EDS (6-MLC EDS)
Multi-Layer Cryptography Encryption and Decryption System with multiple encryption algorithms.

### 7. Integrated RSA & MARS (7-Integrated 1&3)
Integration of RSA and MARS encryption algorithms for enhanced security.

### 8. Custom EDS (8-CustomEDS)
Custom Encryption and Decryption System with unique implementation.

## Features

- Multiple encryption algorithms (RSA, AES, MARS)
- Multi-layer encryption support
- Custom encryption implementations
- Progress bars for long operations
- Colored terminal output
- File and text encryption support
- Customizable security levels

## Security Notice

These implementations are primarily for educational and learning purposes. For production use, please use well-tested cryptographic libraries and follow current security best practices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

Developed by Bruno Liu
