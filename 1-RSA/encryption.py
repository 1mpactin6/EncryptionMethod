import time
import random

def generate_public_key(private_key):
    """
    Generates a public key based on the private key.
    
    Args:
        private_key (int): The private key used for generating the public key.
    
    Returns:
        int: The generated public key.
    """
    return private_key + random.randint(1, 100)  # Simple public key generation

def encrypt(message, private_key):
    """
    Encrypts a message by converting each character to its binary representation
    and then to a custom number format using a private key.
    
    Args:
        message (str): The message to encrypt.
        private_key (int): The private key for encryption.
    
    Returns:
        str: The encrypted message as a string of numbers.
    """
    encrypted_message = ''
    for char in message:
        # Convert character to binary and then to a custom number using the private key
        binary = format(ord(char), '08b')  # 8-bit binary representation
        custom_number = int(binary, 2) + 1000 + private_key  # Custom conversion
        encrypted_message += str(custom_number) + ' '  # Append custom number
    return encrypted_message.strip()  # Remove trailing space

def rsa_encrypt(message, public_key):
    """
    Simulates RSA encryption by adding the public key to the ASCII values of the characters.
    
    Args:
        message (str): The message to encrypt.
        public_key (int): The public key for encryption.
    
    Returns:
        str: The RSA encrypted message as a string of numbers.
    """
    rsa_encrypted_message = ''
    for char in message:
        rsa_encrypted_message += str(ord(char) + public_key) + ' '  # Simple RSA-like encryption
    return rsa_encrypted_message.strip()  # Remove trailing space

def log_encryption():
    """
    Continuously encrypts messages and logs the encrypted binary code with timestamps.
    """
    private_key = random.randint(1, 100)  # Generate a random private key
    public_key = generate_public_key(private_key)  # Generate the public key
    print(f"Public Key: {public_key}")

    while True:
        original_message = input("Enter a message to encrypt (or 'exit' to quit): ")
        if original_message.lower() == 'exit':
            break
        encrypted = encrypt(original_message, private_key)
        rsa_encrypted = rsa_encrypt(original_message, public_key)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f"[{timestamp}] Encrypted (Custom): {encrypted}")
        print(f"[{timestamp}] Encrypted (RSA): {rsa_encrypted}")

# Example usage
if __name__ == "__main__":
    log_encryption()
