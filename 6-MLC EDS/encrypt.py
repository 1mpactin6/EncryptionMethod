import base64
import time
import random
import string
from Crypto.Cipher import AES, DES, Blowfish, DES3
from Crypto.Random import get_random_bytes
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
    'AES-128': [
        "AES-128 uses a 128-bit key length for encryption. It provides good security for most applications.",
        "AES-128 is fast and widely supported across different platforms and systems."
    ],
    'AES-192': [
        "AES-192 uses a 192-bit key length, providing stronger security than AES-128.",
        "AES-192 offers a good balance between security and performance."
    ],
    'AES-256': [
        "AES-256 uses a 256-bit key length, providing the highest level of security among AES variants.",
        "AES-256 is recommended for highly sensitive data requiring maximum security."
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
    ],
    'Base64': [
        "Base64 encoding converts binary data to ASCII text format.",
        "Base64 is commonly used for transmitting binary data in text-based systems."
    ],
    'Binary': [
        "Binary conversion represents data in base-2 format.",
        "Binary is the fundamental representation of data in computer systems."
    ],
    'Octal': [
        "Octal conversion represents data in base-8 format.",
        "Octal is sometimes used in computing and digital communications."
    ],
    'Hexadecimal': [
        "Hexadecimal conversion represents data in base-16 format.",
        "Hexadecimal is commonly used to represent binary data in a more readable format."
    ]
}

# Generate random keys for each algorithm
AES_KEY_128 = get_random_bytes(16)  # 16 bytes for AES-128
AES_KEY_192 = get_random_bytes(24)  # 24 bytes for AES-192
AES_KEY_256 = get_random_bytes(32)  # 32 bytes for AES-256
DES_KEY = get_random_bytes(8)  # 8 bytes for DES
BLOWFISH_KEY = get_random_bytes(16)  # 16 bytes
TRIPLE_DES_KEY = get_random_bytes(24)  # 24 bytes

# Function to encrypt using AES with variable key sizes
def aes_encrypt_128(data, key=AES_KEY_128):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

def aes_encrypt_192(data, key=AES_KEY_192):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

def aes_encrypt_256(data, key=AES_KEY_256):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

# Function to encrypt using DES
def des_encrypt(data, key=DES_KEY):
    cipher = DES.new(key, DES.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), DES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

# Function to encrypt using Blowfish
def blowfish_encrypt(data, key=BLOWFISH_KEY):
    cipher = Blowfish.new(key, Blowfish.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), Blowfish.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

# Function to encrypt using Triple DES
def triple_des_encrypt(data, key=TRIPLE_DES_KEY):
    cipher = DES3.new(key, DES3.MODE_CBC)
    padded_data = pad(data.encode('utf-8'), DES3.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

# Function to encrypt using Format-Preserving Encryption (FPE)
def fpe_encrypt(data):
    # Simple substitution cipher as FPE placeholder
    substitution = str.maketrans(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM9012345678'
    )
    return data.translate(substitution)

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
    
    for step in sequence:
        step = step.strip()  # Remove whitespace from each algorithm name
        if step not in descriptions:
            log_message(Fore.RED + f"Invalid encryption algorithm: {step}" + Style.RESET_ALL)
            continue
        
        print(Fore.CYAN + random.choice(descriptions[step]) + Style.RESET_ALL)

        for progress in tqdm(range(100), desc=f"Encrypting {step}", leave=False):
            time.sleep(0.02)

        if step == 'AES-128':
            message = aes_encrypt_128(message)
            log_message(Fore.BLUE + "AES-128 encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'AES-192':
            message = aes_encrypt_192(message)
            log_message(Fore.BLUE + "AES-192 encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'AES-256':
            message = aes_encrypt_256(message)
            log_message(Fore.BLUE + "AES-256 encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'DES':
            message = des_encrypt(message)
            log_message(Fore.BLUE + "DES encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'Blowfish':
            message = blowfish_encrypt(message)
            log_message(Fore.BLUE + "Blowfish encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'TripleDES':
            message = triple_des_encrypt(message)
            log_message(Fore.BLUE + "Triple DES encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'FPE':
            message = fpe_encrypt(message)
            log_message(Fore.BLUE + "FPE encryption applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'Base64':
            message = convert_to_base(message, 64)
            log_message(Fore.BLUE + "Base64 encoding applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'Binary':
            message = convert_to_base(message, 2)
            log_message(Fore.BLUE + "Binary conversion applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'Octal':
            message = convert_to_base(message, 8)
            log_message(Fore.BLUE + "Octal conversion applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        elif step == 'Hexadecimal':
            message = convert_to_base(message, 16)
            log_message(Fore.BLUE + "Hexadecimal conversion applied" + Style.RESET_ALL)
            print(f"Current output: {message}")
        
        time.sleep(1)
    log_message(Fore.GREEN + "Encryption complete." + Style.RESET_ALL)
    return message

# User interface for encryption
def user_interface():
    print(Fore.CYAN + "Welcome to the Encryption Program!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: You can encrypt messages in different languages, symbols, and numbers!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Choose your encryption sequence wisely for better security!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Type 'exit' at any time to quit the program." + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + "Available encryption algorithms: AES-128, AES-192, AES-256, DES, Blowfish, TripleDES, FPE, Base64, Binary, Octal, Hexadecimal" + Style.RESET_ALL)
    
    while True:
        message = input("\nEnter the message to encrypt (type 'exit' to quit) (^C to quit): ")
        if message.lower() == 'exit':
            log_message(Fore.RED + "Exiting the program." + Style.RESET_ALL)
            break
        sequence = input("Enter the encryption sequence (e.g., AES-128,AES-192,AES-256,DES,Blowfish,TripleDES,FPE,Base64,Binary,Octal,Hexadecimal): ").split(',')
        
        encrypted_message = encrypt_message(message, sequence)
        print(f"{Fore.YELLOW}Final encrypted message: {encrypted_message}{Style.RESET_ALL}")

if __name__ == "__main__":
    user_interface()
