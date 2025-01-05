import time
from datetime import datetime

#from Crypto.Cipher import MARS
from Crypto.Random import get_random_bytes
import base64

import rsa
import random

from Crypto.Cipher import AES

def encrypt_string_to_aes_binary(input_string):
    key = get_random_bytes(16)  # AES uses keys of 16, 24, or 32 bytes
    cipher = AES.new(key, AES.MODE_ECB)
    while len(input_string) % 16 != 0:
        input_string += ' '
    encrypted_bytes = cipher.encrypt(input_string.encode('utf-8'))
    encrypted_base64 = base64.b64encode(encrypted_bytes).decode('utf-8')
    return encrypted_base64

def decrypt_aes_binary_to_string(encrypted_base64, key):
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_bytes = base64.b64decode(encrypted_base64)
    decrypted_bytes = cipher.decrypt(encrypted_bytes)
    return decrypted_bytes.decode('utf-8').rstrip()

def get_formatted_log(message):
    """
    Returns a formatted log string with timestamp
    Format: [yyyy-MM-dd-HH-mm-SS/OUTPUT:] message
    """
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return f"[{timestamp}/OUTPUT:] {message}"

def string_to_binary(input_string):
    """
    Converts a string to a binary string.
    Each character is converted to its ASCII binary representation.
    """
    return ' '.join(format(ord(char), '08b') for char in input_string)

def binary_to_string(binary_string):
    """
    Converts a binary string back to the original string.
    Assumes that each 8-bit binary number (byte) is separated by space.
    """
    return ''.join(chr(int(binary, 2)) for binary in binary_string.split())

# def encrypt_string_to_mars_binary(input_string):
#     """
#     Encrypts a string using MARS encryption algorithm and returns the encrypted data as a binary string.
#     """
#     key = get_random_bytes(16)  # MARS uses keys from 16 to 32 bytes
#     cipher = MARS.new(key, MARS.MODE_ECB)
#     # Ensure the input string length is a multiple of 16
#     while len(input_string) % 16 != 0:
#         input_string += ' '
#     encrypted_bytes = cipher.encrypt(input_string.encode('utf-8'))
#     # Convert encrypted bytes to a base64 string to ensure safe printability and storage
#     encrypted_base64 = base64.b64encode(encrypted_bytes).decode('utf-8')
#     return encrypted_base64

# def decrypt_mars_binary_to_string(encrypted_base64, key):
#     """
#     Decrypts a MARS encrypted binary string back to the original string using the same key.
#     """
#     cipher = MARS.new(key, MARS.MODE_ECB)
#     encrypted_bytes = base64.b64decode(encrypted_base64)
#     decrypted_bytes = cipher.decrypt(encrypted_bytes)
#     # Strip padding spaces if any were added during encryption
#     return decrypted_bytes.decode('utf-8').rstrip()

def rsa_encrypt_string(input_string):
    """
    Encrypts a string using RSA encryption with a fixed private key and a random public key.
    Returns the encrypted data and the random public key used for encryption.
    """
    # Fixed private key and its corresponding public key (normally you would generate these)
    private_key = rsa.PrivateKey(874799280, 65537, 0, 0, 0)  # Dummy values for p, q, d (not used)
    public_key = rsa.PublicKey(874799280, 65537)
    
    # Encrypt with the private key first
    encrypted_with_private = rsa.encrypt(input_string.encode('utf-8'), private_key)
    
    # Generate a random public key within the specified range and encrypt the data again
    random_exponent = random.randint(100000000, 1000000000)
    random_public_key = rsa.PublicKey(random_exponent, 65537)
    encrypted_with_random_public = rsa.encrypt(encrypted_with_private, random_public_key)
    
    return random_public_key, encrypted_with_random_public

def rsa_decrypt_string(random_public_key, encrypted_data):
    """
    Decrypts the data encrypted by rsa_encrypt_string using the random public key and the fixed private key.
    """
    # Fixed private key (normally you would generate this)
    private_key = rsa.PrivateKey(874799280, 65537, 0, 0, 0)  # Dummy values for p, q, d (not used)
    
    # Decrypt with the random public key first (in practice, you would need the corresponding private key)
    decrypted_with_public = rsa.decrypt(encrypted_data, random_public_key)
    
    # Decrypt with the fixed private key
    decrypted_with_private = rsa.decrypt(decrypted_with_public, private_key)
    
    return decrypted_with_private.decode('utf-8')

def main():
    while True:
        try:
            user_input = input("Enter a string to encrypt (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting the program.")
                break
            
            log_message = get_formatted_log(f"Encrypting: {user_input}")
            print(log_message)
            
            print("In progress...")
            
            binary_string = string_to_binary(user_input)
            encrypted_mars = encrypt_string_to_mars_binary(binary_string)
            random_public_key, encrypted_rsa = rsa_encrypt_string(encrypted_mars)
            
            # Output the results
            print(f"\033[92mEncrypted RSA: {encrypted_rsa}\033[0m")  # Green text for success
            print(f"RSA Public Key: {random_public_key}")
        
        except Exception as e:
            print(f"\033[91mError: {str(e)}\033[0m")  # Red text for error
            continue  # Continue the loop to allow for new input