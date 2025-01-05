import time
from datetime import datetime
from Crypto.Random import get_random_bytes
import base64
import rsa
from Crypto.Cipher import AES
from colorama import Fore, Style, init

# Initialize colorama
init()

# Colors for visual clarity
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

# List of random tips to display during encryption
tips = [
    "Tip: Always use a strong and unique key for better security.",
    "Tip: Consider the length of your message; longer messages may require more processing time.",
    "Tip: Use a combination of algorithms for enhanced security.",
    "Tip: Regularly update your encryption methods to stay secure.",
    "Tip: Test your encryption with different types of data to ensure compatibility.",
    "Tip: Keep your encryption keys secure and private.",
    "Tip: Use a secure channel to share your encrypted messages.",
    "Tip: Remember that encryption is only as strong as the key used.",
    "Tip: Consider using a password manager to store your keys securely.",
    "Tip: Always back up your encrypted data in a secure location."
]

# List of descriptions for each encryption algorithm
descriptions = {
    'AES': [
        "AES (Advanced Encryption Standard) is a symmetric encryption algorithm widely used across the globe. It operates on fixed block sizes of 128 bits and supports key sizes of 128, 192, or 256 bits.",
        "AES is known for its speed and security, making it suitable for encrypting sensitive data in various applications, including file encryption and secure communications."
    ],
    'DES': [
        "DES (Data Encryption Standard) is a symmetric-key algorithm that encrypts data in 64-bit blocks using a 56-bit key. It was once a widely used encryption standard but is now considered insecure due to its short key length.",
        "Despite its vulnerabilities, DES laid the groundwork for modern encryption algorithms and is still studied for educational purposes."
    ],
    'Blowfish': [
        "Blowfish is a symmetric-key block cipher that encrypts data in 64-bit blocks and supports variable key lengths from 32 bits to 448 bits. It is known for its speed and effectiveness in software implementations.",
        "Blowfish is often used in applications where speed is critical, such as in network protocols and file encryption."
    ],
    'TripleDES': [
        "Triple DES (3DES) enhances the security of the original DES algorithm by applying the DES cipher three times to each data block. It effectively increases the key length to 168 bits, making it more secure than DES.",
        "While Triple DES is more secure than its predecessor, it is slower and has been largely replaced by AES in modern applications."
    ],
    'FPE': [
        "Format-Preserving Encryption (FPE) allows data to be encrypted while maintaining its original format. This is particularly useful for encrypting sensitive data such as credit card numbers or social security numbers.",
        "FPE ensures that the encrypted output has the same length and format as the input, making it easier to integrate into existing systems without requiring significant changes."
    ]
}

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

def encrypt_string_to_aes_binary(input_string):
    key = get_random_bytes(16)  # AES uses keys of 16, 24, or 32 bytes
    cipher = AES.new(key, AES.MODE_ECB)
    while len(input_string) % 16 != 0:
        input_string += ' '  # Padding to make it a multiple of 16
    encrypted_bytes = cipher.encrypt(input_string.encode('utf-8'))
    encrypted_base64 = base64.b64encode(encrypted_bytes).decode('utf-8')
    return encrypted_base64, key  # Return the encrypted data and key

def decrypt_aes_binary_to_string(encrypted_base64, key):
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_bytes = base64.b64decode(encrypted_base64)
    decrypted_bytes = cipher.decrypt(encrypted_bytes)
    return decrypted_bytes.decode('utf-8').rstrip()

def get_formatted_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return f"[{timestamp}/OUTPUT:] {message}"

def string_to_binary(input_string):
    return ' '.join(format(ord(char), '08b') for char in input_string)

def rsa_encrypt_string(input_string, public_key):
    encrypted_data = rsa.encrypt(input_string.encode('utf-8'), public_key)
    return encrypted_data

def encrypt_message(message, sequence):
    # Implement the encryption sequence logic here
    encrypted = message
    for algorithm in sequence:
        if algorithm.upper() == 'AES':
            encrypted, _ = encrypt_string_to_aes_binary(encrypted)
    return encrypted

def multi_layer_encryption(plaintext, rsa_public_key, aes_key, des_key, mars_key, fpe_key):
    # Implement multi-layer encryption logic here
    encrypted_aes, _ = encrypt_string_to_aes_binary(plaintext)
    return encrypted_aes

def main():
    print(GREEN + "\nWelcome to the Advanced Multi-layer Encryption System!" + RESET)
    print("================================================")
    
    # Generate RSA keys (Public and Private)
    (public_key, private_key) = rsa.newkeys(2048)
    
    while True:
        print("\nOptions:")
        print("1. Basic RSA-AES Encryption")
        print("2. Multi-layer Encryption")
        print("3. Custom Sequence Encryption")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "4":
            print("Exiting the program.")
            break
            
        elif choice == "1":
            try:
                user_input = input("Enter a string to encrypt: ")
                log_message = get_formatted_log(f"Encrypting: {user_input}")
                print(log_message)
                print("In progress...")

                encrypted_aes, aes_key = encrypt_string_to_aes_binary(user_input)
                encrypted_rsa_key = rsa_encrypt_string(base64.b64encode(aes_key).decode('utf-8'), public_key)
                
                print(f"\033[92mEncrypted AES: {encrypted_aes}\033[0m")
                print(f"Encrypted RSA Key: {base64.b64encode(encrypted_rsa_key).decode('utf-8')}")
                
            except Exception as e:
                print(f"\033[91mError: {str(e)}\033[0m")
                
        elif choice == "2":
            plaintext = input("Enter plaintext to encrypt: ")
            aes_key = get_random_bytes(16)
            des_key = get_random_bytes(8)
            mars_key = get_random_bytes(16)
            fpe_key = "FPE1234"
            
            result = multi_layer_encryption(plaintext, public_key, aes_key, des_key, mars_key, fpe_key)
            print(f"\033[92mMulti-layer encrypted result: {result}\033[0m")
            
        elif choice == "3":
            message = input("\nEnter the message to encrypt: ")
            sequence = input("Enter encryption sequence (e.g., AES,DES,Blowfish,TripleDES,FPE): ").split(',')
            encrypted_message = encrypt_message(message, sequence)
            print(f"{Fore.YELLOW}Encrypted message: {encrypted_message}{Style.RESET_ALL}")
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()