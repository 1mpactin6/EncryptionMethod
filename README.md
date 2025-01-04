# EncryptionMethod
 A Python code that can encrypt and decrypt any types of binary code

## Overview

The DecryptionMethod provides an advanced interface for users to encrypt and ecrypt binary codes into custom RSA like code. By implementing a Python-environment users and freely use this tool.

## Features

- Encrypt inputed messages and provide public key.
- Decrypt messages using a provided public key.
- Brute-force decryption by trying different private keys.
- Interactive command-line interface with color-coded output for better readability.
- Detailed explanations of the decryption processes.

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

## Usage

1. Run the decryption tool:

   ```bash
   python encryption.py
   python decryption.py
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
