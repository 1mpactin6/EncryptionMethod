import base64
import time
import random
import string
from Crypto.Cipher import AES, DES, Blowfish, DES3
from tqdm import tqdm  # For progress bar
from colorama import Fore, Style  # For colored output
from Crypto.Util.Padding import pad, unpad

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

# Function to generate a random key for AES
def generate_aes_key(size):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

# Function to generate a random key for DES
def generate_des_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

# Function to encrypt using AES
def aes_encrypt(data, key):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(ciphertext).decode('utf-8')

# Function to encrypt using DES
def des_encrypt(data, key):
    cipher = DES.new(key.encode('utf-8'), DES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(ciphertext).decode('utf-8')

# Function to encrypt using Blowfish
def blowfish_encrypt(data, key):
    cipher = Blowfish.new(key.encode('utf-8'), Blowfish.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(ciphertext).decode('utf-8')

# Function to encrypt using Triple DES
def triple_des_encrypt(data, key):
    cipher = DES3.new(key.encode('utf-8'), DES3.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(ciphertext).decode('utf-8')

# Function to encrypt using Format-Preserving Encryption (FPE)
def fpe_encrypt(data):
    # Placeholder for FPE implementation
    # In a real implementation, you would use a library that supports FPE
    return base64.b64encode(data.encode('utf-8')).decode('utf-8')  # Simple base64 as a placeholder

# Function to convert data to different bases
def convert_to_base(data, base):
    if base == 2:
        return ''.join(format(ord(i), '08b') for i in data)
    elif base == 8:
        return ''.join(format(ord(i), '03o') for i in data)
    elif base == 16:
        return data.encode('utf-8').hex()
    elif base == 64:
        return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    return data

# Function to log messages with timestamps
def log_message(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Main encryption function
def encrypt_message(message, sequence):
    log_message(Fore.GREEN + "Starting encryption process..." + Style.RESET_ALL)
    total_steps = len(sequence)
    
    for i, step in enumerate(sequence):
        # Validate the algorithm name
        if step not in descriptions:
            log_message(Fore.RED + f"Invalid encryption algorithm: {step}. Please use one of the following: {', '.join(descriptions.keys())}" + Style.RESET_ALL)
            continue
        
        # Display a random tip during the encryption process
        if random.random() < 0.3:  # 30% chance to show a tip
            print(Fore.YELLOW + random.choice(tips) + Style.RESET_ALL)
        
        # Display a description of the algorithm being used
        print(Fore.CYAN + random.choice(descriptions[step]) + Style.RESET_ALL)

        # Simulate the encryption process with a more realistic progress update
        for progress in tqdm(range(100), desc=f"Encrypting {step}", leave=False):
            time.sleep(0.02)  # Simulate work being done

        if step == 'AES':
            key_size = int(input("Enter AES key size (128, 192, or 256): "))
            key = generate_aes_key(key_size // 8)  # Key size in bytes
            message = aes_encrypt(message, key)
            log_message(Fore.BLUE + f"AES encryption applied with key: {key}" + Style.RESET_ALL)
        elif step == 'DES':
            key = generate_des_key()
            message = des_encrypt(message, key)
            log_message(Fore.BLUE + f"DES encryption applied with key: {key}" + Style.RESET_ALL)
        elif step == 'Blowfish':
            key = generate_aes_key(16)  # Using 16 bytes for Blowfish
            message = blowfish_encrypt(message, key)
            log_message(Fore.BLUE + f"Blowfish encryption applied with key: {key}" + Style.RESET_ALL)
        elif step == 'TripleDES':
            key = generate_aes_key(24)  # Triple DES key must be 24 bytes
            message = triple_des_encrypt(message, key)
            log_message(Fore.BLUE + f"Triple DES encryption applied with key: {key}" + Style.RESET_ALL)
        elif step == 'FPE':
            message = fpe_encrypt(message)
            log_message(Fore.BLUE + "FPE encryption applied." + Style.RESET_ALL)
        
        time.sleep(1)  # Simulate time delay for encryption
    log_message(Fore.GREEN + "Encryption complete." + Style.RESET_ALL)
    return message

# User interface for encryption
def user_interface():
    print(Fore.CYAN + "Welcome to the Encryption Program!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: You can encrypt messages in different languages, symbols, and numbers!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Choose your encryption sequence wisely for better security!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Type 'exit' at any time to quit the program." + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + "Available encryption algorithms: AES, DES, Blowfish, TripleDES, FPE" + Style.RESET_ALL)
    
    while True:
        message = input("\nEnter the message to encrypt: ")
        if message.lower() == 'exit':
            log_message(Fore.RED + "Exiting the program." + Style.RESET_ALL)
            break
        sequence = input("Enter the encryption sequence (e.g., AES,DES,Blowfish,TripleDES,FPE): ").split(',')
        
        encrypted_message = encrypt_message(message, sequence)
        print(f"{Fore.YELLOW}Encrypted message: {encrypted_message}{Style.RESET_ALL}")

if __name__ == "__main__":
    user_interface()
