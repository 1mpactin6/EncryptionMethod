import time
import random
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

def encrypt(message):
    """
    Encrypts a string message into a custom number format.
    
    Each character in the message is converted to its ASCII value,
    and then each ASCII value is converted to a string of numbers.
    
    Args:
        message (str): A string containing any characters.
    
    Returns:
        str: The encrypted message as a string of numbers.
    """
    encrypted_message = ''.join(str(ord(char)) for char in message)
    return encrypted_message

def log_encryption(message, public_key):
    """
    Logs the encrypted output with timestamps and public key.
    
    Args:
        message (str): A string to be encrypted.
        public_key: The RSA public key for encryption.
    """
    encrypted = encrypt(message)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Encrypted: {encrypted}")
    print(f"[{timestamp}] Public Key: {public_key.decode()}")

def generate_rsa_keys():
    """
    Generates a pair of RSA keys (private and public).
    
    Returns:
        private_key: The private RSA key.
        public_key: The public RSA key.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

def rsa_encrypt(message, public_key):
    """
    Encrypts a message using the provided RSA public key.
    
    Args:
        message (str): The message to encrypt.
        public_key: The RSA public key for encryption.
    
    Returns:
        bytes: The encrypted message.
    """
    encrypted = public_key.encrypt(
        message.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted

def mars_encrypt(message):
    """
    Simulates MARS encryption by applying a simple transformation.
    
    Args:
        message (str): The message to encrypt.
    
    Returns:
        str: The MARS encrypted message.
    """
    # Simple transformation for demonstration (not actual MARS encryption)
    return ''.join(str((ord(char) + 5) % 256) for char in message)

def generate_varying_outputs():
    """
    Generates a series of varying outputs for RSA decryption.
    
    Returns:
        list: A list of varying outputs.
    """
    return [random.randint(290000, 900000) for _ in range(7)]

# Example usage
if __name__ == "__main__":
    # Generate RSA keys
    private_key, public_key = generate_rsa_keys()
    
    # Display public key
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    print("Public Key:")
    print(public_key_pem.decode())

    # Get user input for any string message
    message = input("Enter a message (including numbers, symbols, and characters) or 'exit' to stop: ")
    
    while message.lower() != "exit":
        # Log the encrypted message and public key
        log_encryption(message, public_key_pem)

        # Example RSA encryption
        encrypted_message = rsa_encrypt(message, public_key)
        print(f"Encrypted Message: {encrypted_message}")

        # Generate varying outputs for RSA decryption
        varying_outputs = generate_varying_outputs()
        print("Varying RSA Decryption Outputs:")
        for output in varying_outputs:
            print(output)

        # Apply MARS encryption to the original message
        mars_encrypted_message = mars_encrypt(message)
        print(f"MARS Encrypted Message: {mars_encrypted_message}")

        # Prompt for the next message
        message = input("Enter a message (including numbers, symbols, and characters) or 'exit' to stop: ")

    print("Exiting the program.")
