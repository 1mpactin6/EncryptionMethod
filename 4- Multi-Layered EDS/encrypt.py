import base64
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
import os

# Colors for visual clarity
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

# AES Encryption (using CBC mode)
def encrypt_aes(plaintext, aes_key):
    cipher = AES.new(aes_key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return iv, ct

# DES Encryption (using ECB mode)
def encrypt_des(plaintext, des_key):
    cipher = DES.new(des_key, DES.MODE_ECB)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), DES.block_size))
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return ct

# RSA Encryption (using OAEP padding)
def encrypt_rsa(plaintext, rsa_public_key):
    cipher = PKCS1_OAEP.new(rsa_public_key)
    encrypted = cipher.encrypt(plaintext.encode())
    return base64.b64encode(encrypted).decode('utf-8')

# MARS Encryption (simulating with AES)
def encrypt_mars(plaintext, mars_key):
    cipher = AES.new(mars_key, AES.MODE_ECB)  # Using AES as a placeholder for MARS
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return base64.b64encode(ct_bytes).decode('utf-8')

# Base64 Encoding
def base64_encode(data):
    return base64.b64encode(data.encode()).decode('utf-8')

# Format-Preserving Encryption (FPE) - Simulated
def encrypt_fpe(plaintext, fpe_key):
    return fpe_key + plaintext[::-1]  # Simply reversing the string for simulation

# Padding for AES and DES
def pad(data, block_size):
    padding_length = block_size - len(data) % block_size
    padding = bytes([padding_length]) * padding_length  # Create padding in bytes
    return data + padding

# Multi-Layer Encryption Function
def multi_layer_encryption(plaintext, rsa_public_key, aes_key, des_key, mars_key, fpe_key):
    print(GREEN + "\nStarting multi-layer encryption..." + RESET)

    # AES Encryption
    aes_iv, aes_encrypted = encrypt_aes(plaintext, aes_key)
    print(YELLOW + f"AES Encrypted (Base64): {aes_encrypted}" + RESET)

    # DES Encryption
    des_encrypted = encrypt_des(plaintext, des_key)
    print(YELLOW + f"DES Encrypted (Base64): {des_encrypted}" + RESET)

    # RSA Encryption
    rsa_encrypted = encrypt_rsa(plaintext, rsa_public_key)
    print(YELLOW + f"RSA Encrypted (Base64): {rsa_encrypted}" + RESET)

    # MARS Encryption
    mars_encrypted = encrypt_mars(plaintext, mars_key)
    print(YELLOW + f"MARS Encrypted (Base64): {mars_encrypted}" + RESET)

    # Base64 Encoding
    base64_encoded = base64_encode(plaintext)
    print(YELLOW + f"Base64 Encoded: {base64_encoded}" + RESET)

    # FPE Encryption
    fpe_encrypted = encrypt_fpe(plaintext, fpe_key)
    print(YELLOW + f"FPE Encrypted: {fpe_encrypted}" + RESET)

    return aes_iv, aes_encrypted, des_encrypted, rsa_encrypted, mars_encrypted, base64_encoded, fpe_encrypted

# Main function
def main():
    print(GREEN + "\nWelcome to the Multi-layer Encryption System!" + RESET)
    print("===============================================")
    print("Options:")
    print("1. Encrypt Data")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        plaintext = input("Enter plaintext to encrypt: ")

        # Generate RSA keys (for demonstration)
        key = RSA.generate(2048)
        rsa_public_key = key.publickey()

        # Generate random keys for AES, DES, and MARS
        aes_key = get_random_bytes(16)  # AES 128-bit key
        des_key = get_random_bytes(8)   # DES 64-bit key
        mars_key = get_random_bytes(16)  # MARS 128-bit key (simulated with AES)
        fpe_key = "FPE1234"  # A simple FPE key (in real scenarios, use a secure method)

        # Perform multi-layer encryption
        multi_layer_encryption(plaintext, rsa_public_key, aes_key, des_key, mars_key, fpe_key)

if __name__ == "__main__":
    main()
