import random
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# ANSI escape codes for colored output
GREEN = "\033[92m"  # Green text
RED = "\033[91m"    # Red text
YELLOW = "\033[93m"  # Yellow text
RESET = "\033[0m"   # Reset to default color

def mars_decrypt(encrypted_message):
    """
    Simulates MARS decryption by reversing the transformation applied during encryption.
    
    Args:
        encrypted_message (str): The MARS encrypted message.
    
    Returns:
        str: The decrypted message.
    """
    decrypted_message = ''.join(chr((int(num) - 5) % 256) for num in encrypted_message.split())
    return decrypted_message

def rsa_decrypt(encrypted_message, private_key):
    """
    Decrypts a message using the provided RSA private key.
    
    Args:
        encrypted_message (bytes): The encrypted message to decrypt.
        private_key: The RSA private key for decryption.
    
    Returns:
        str: The decrypted message.
    """
    decrypted = private_key.decrypt(
        encrypted_message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return decrypted.decode()

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

def main():
    # Generate RSA keys for demonstration
    private_key, public_key = generate_rsa_keys()

    # Example RSA encrypted message (for demonstration purposes)
    message = "Hello, RSA!"
    encrypted_message = public_key.encrypt(
        message.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Display the encrypted message
    print(f"{YELLOW}Encrypted RSA Message:{RESET} {encrypted_message}")

    # Decrypt the RSA message
    decrypted_message = rsa_decrypt(encrypted_message, private_key)
    print(f"{GREEN}Decrypted RSA Message:{RESET} {decrypted_message}")

    # Example MARS encrypted message (simulated)
    mars_encrypted_message = "72 101 108 108 111"  # Simulated MARS encrypted message for "Hello"
    print(f"{YELLOW}MARS Encrypted Message:{RESET} {mars_encrypted_message}")

    # Decrypt the MARS message
    decrypted_mars_message = mars_decrypt(mars_encrypted_message)
    print(f"{GREEN}Decrypted MARS Message:{RESET} {decrypted_mars_message}")

    # Allow multiple inputs for decryption
    while True:
        user_input = input(f"{YELLOW}Enter an RSA encrypted message (or MARS encrypted message) to decrypt (or 'exit' to stop): {RESET}")
        
        if user_input.lower() == "exit":
            print(f"{RED}Exiting the program.{RESET}")
            break
        
        # Attempt to decrypt as RSA
        try:
            # Assuming the input is in bytes format for RSA
            decrypted_rsa_message = rsa_decrypt(eval(user_input), private_key)
            print(f"{GREEN}Decrypted RSA Message:{RESET} {decrypted_rsa_message}")
        except Exception as e:
            print(f"{RED}Failed to decrypt RSA message: {e}{RESET}")

        # Attempt to decrypt as MARS
        try:
            decrypted_mars_message = mars_decrypt(user_input)
            print(f"{GREEN}Decrypted MARS Message:{RESET} {decrypted_mars_message}")
        except Exception as e:
            print(f"{RED}Failed to decrypt MARS message: {e}{RESET}")

if __name__ == "__main__":
    main()
