import base64
import time
import random
import string
from Crypto.Cipher import AES, DES, Blowfish, DES3
from Crypto.Random import get_random_bytes
from tqdm import tqdm  # For progress bar
from colorama import Fore, Style  # For colored output
from Crypto.Util.Padding import pad, unpad

# List of random tips to display during decryption
tips = [
    "Tip: Always verify the key used for decryption matches the one used for encryption.",
    "Tip: Ensure the encrypted message is not altered before decryption.",
    "Tip: Use a secure method to share your decryption keys.",
    "Tip: Regularly update your decryption methods to stay secure.",
    "Tip: Test your decryption with known values to ensure accuracy.",
    "Tip: Keep your decryption keys secure and private.",
    "Tip: Use a secure channel to share your decrypted messages.",
    "Tip: Remember that decryption is only as strong as the key used.",
    "Tip: Consider using a password manager to store your keys securely.",
    "Tip: Always back up your decrypted data in a secure location."
]

# List of descriptions for each decryption algorithm
descriptions = {
    'AES': [
        "AES (Advanced Encryption Standard) is a symmetric encryption algorithm widely used across the globe. It operates on fixed block sizes of 128 bits and supports key sizes of 128, 192, or 256 bits.",
        "AES is known for its speed and security, making it suitable for decrypting sensitive data in various applications."
    ],
    'DES': [
        "DES (Data Encryption Standard) is a symmetric-key algorithm that encrypts data in 64-bit blocks using a 56-bit key. It is now considered insecure due to its short key length.",
        "Despite its vulnerabilities, DES is still studied for educational purposes."
    ],
    'Blowfish': [
        "Blowfish is a symmetric-key block cipher that encrypts data in 64-bit blocks and supports variable key lengths from 32 bits to 448 bits.",
        "Blowfish is often used in applications where speed is critical, such as in network protocols."
    ],
    'TripleDES': [
        "Triple DES (3DES) enhances the security of the original DES algorithm by applying the DES cipher three times to each data block.",
        "While Triple DES is more secure than its predecessor, it is slower and has been largely replaced by AES."
    ],
    'FPE': [
        "Format-Preserving Encryption (FPE) allows data to be decrypted while maintaining its original format.",
        "FPE ensures that the decrypted output has the same length and format as the input."
    ]
}

# Function to decrypt using AES
def aes_decrypt(data, key):
    # Decode the base64 encoded data
    decoded_data = base64.b64decode(data)
    # Extract the nonce (first 16 bytes)
    nonce = decoded_data[:16]
    # Create the cipher object
    cipher = AES.new(key.encode('utf-8'), AES.MODE_EAX, nonce=nonce)
    # Decrypt the data
    decrypted_data = cipher.decrypt(decoded_data[16:])
    return decrypted_data

# Function to decrypt using DES
def des_decrypt(data, key):
    cipher = DES.new(key.encode('utf-8'), DES.MODE_EAX)
    decrypted_data = cipher.decrypt(base64.b64decode(data))
    return decrypted_data

# Function to decrypt using Blowfish
def blowfish_decrypt(data, key):
    cipher = Blowfish.new(key.encode('utf-8'), Blowfish.MODE_EAX)
    decrypted_data = cipher.decrypt(base64.b64decode(data))
    return decrypted_data

# Function to decrypt using Triple DES
def triple_des_decrypt(data, key):
    cipher = DES3.new(key.encode('utf-8'), DES3.MODE_EAX)
    decrypted_data = cipher.decrypt(base64.b64decode(data))
    return decrypted_data

# Function to decrypt using Format-Preserving Encryption (FPE)
def fpe_decrypt(data):
    # Placeholder for FPE implementation
    return base64.b64decode(data).decode('utf-8')  # Simple base64 as a placeholder

# Function to log messages with timestamps
def log_message(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Function to brute force decryption
def brute_force_decrypt(encrypted_message, max_key_length=4, charset=string.ascii_letters + string.digits):
    # Attempt to brute force with a range of keys
    for key_length in range(1, max_key_length + 1):
        for key in (''.join(random.choices(charset, k=key_length)) for _ in range(1000)):
            try:
                decrypted_message = aes_decrypt(encrypted_message, key)
                return decrypted_message.decode('utf-8')
            except (UnicodeDecodeError, Exception):
                continue
    return None

# Function to detect input typos
def detect_typos(user_input, valid_options):
    for option in valid_options:
        if user_input.lower() in option.lower():
            return option
    return None

# Function to prompt for valid input
def prompt_for_input(prompt, valid_options=None):
    while True:
        user_input = input(prompt)
        if valid_options:
            detected = detect_typos(user_input, valid_options)
            if detected:
                print(f"Did you mean: {detected}?")
                user_input = detected
            else:
                print("Invalid input. Please try again.")
                continue
        return user_input

# Function to crack seeds (placeholder)
def crack_seed():
    # Placeholder for seed cracking logic
    print("Cracking seed functionality is not implemented yet.")

# Main decryption function
def decrypt_message(encrypted_message, sequence):
    log_message(Fore.GREEN + "Starting decryption process..." + Style.RESET_ALL)
    
    if not sequence:
        log_message(Fore.YELLOW + "No sequence provided, attempting brute force..." + Style.RESET_ALL)
        decrypted_message = brute_force_decrypt(encrypted_message)
        if decrypted_message:
            return decrypted_message
        else:
            log_message(Fore.RED + "Brute force failed. No valid key found." + Style.RESET_ALL)
            return None

    decrypted_message = None  # Initialize the variable

    for step in tqdm(sequence, desc="Decrypting", unit="step"):
        # Validate the algorithm name
        if step not in descriptions:
            log_message(Fore.RED + f"Invalid decryption algorithm: {step}. Please use one of the following: {', '.join(descriptions.keys())}" + Style.RESET_ALL)
            continue
        
        # Display a random tip during the decryption process
        if random.random() < 0.3:  # 30% chance to show a tip
            print(Fore.YELLOW + random.choice(tips) + Style.RESET_ALL)
        
        # Display a description of the algorithm being used
        print(Fore.CYAN + random.choice(descriptions[step]) + Style.RESET_ALL)

        # Simulate the decryption process with a more realistic progress update
        for progress in tqdm(range(100), desc=f"Decrypting {step}", leave=False):
            time.sleep(0.02)  # Simulate work being done

        try:
            if step == 'AES':
                key_length = prompt_for_input("Enter AES key length (128, 192, or 256): ", ['128', '192', '256'])
                key = input(f"Enter AES key ({int(key_length) // 8} bytes): ")
                while len(key) != int(key_length) // 8:
                    print(Fore.RED + f"Invalid AES key length. Must be {int(key_length) // 8} bytes." + Style.RESET_ALL)
                    key = input(f"Enter AES key ({int(key_length) // 8} bytes): ")
                decrypted_message = aes_decrypt(encrypted_message, key).decode('utf-8')
                log_message(Fore.BLUE + f"AES decryption applied with key: {key}" + Style.RESET_ALL)
            elif step == 'DES':
                key = prompt_for_input("Enter DES key (8 bytes): ", [string.ascii_letters + string.digits])
                while len(key) != 8:
                    print(Fore.RED + "Invalid DES key length. Must be 8 bytes." + Style.RESET_ALL)
                    key = input("Enter DES key (8 bytes): ")
                decrypted_message = des_decrypt(encrypted_message, key).decode('utf-8')
                log_message(Fore.BLUE + f"DES decryption applied with key: {key}" + Style.RESET_ALL)
            elif step == 'Blowfish':
                key = prompt_for_input("Enter Blowfish key (4 to 56 bytes): ", [string.ascii_letters + string.digits])
                decrypted_message = blowfish_decrypt(encrypted_message, key).decode('utf-8')
                log_message(Fore.BLUE + f"Blowfish decryption applied with key: {key}" + Style.RESET_ALL)
            elif step == 'TripleDES':
                key = prompt_for_input("Enter Triple DES key (24 bytes): ", [string.ascii_letters + string.digits])
                while len(key) != 24:
                    print(Fore.RED + "Invalid Triple DES key length. Must be 24 bytes." + Style.RESET_ALL)
                    key = input("Enter Triple DES key (24 bytes): ")
                decrypted_message = triple_des_decrypt(encrypted_message, key).decode('utf-8')
                log_message(Fore.BLUE + f"Triple DES decryption applied with key: {key}" + Style.RESET_ALL)
            elif step == 'FPE':
                decrypted_message = fpe_decrypt(encrypted_message)
                log_message(Fore.BLUE + "FPE decryption applied." + Style.RESET_ALL)
        except UnicodeDecodeError:
            log_message(Fore.RED + "Decryption failed. Invalid key or corrupted data." + Style.RESET_ALL)
            continue

    log_message(Fore.GREEN + "Decryption complete." + Style.RESET_ALL)
    return decrypted_message

# User interface for decryption
def user_interface():
    print(Fore.CYAN + "Welcome to the Decryption Program!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: You can decrypt messages in different languages, symbols, and numbers!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Choose your decryption sequence wisely for better security!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Tip: Type 'exit' at any time to quit the program." + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + "Available decryption algorithms: AES, DES, Blowfish, TripleDES, FPE" + Style.RESET_ALL)
    
    while True:
        encrypted_message = input("\nEnter the encrypted message: ")
        if encrypted_message.lower() == 'exit':
            log_message(Fore.RED + "Exiting the program." + Style.RESET_ALL)
            break
        sequence = input("Enter the decryption sequence (e.g., AES,DESC,Blowfish,TripleDES,FPE) or leave blank for brute force: ").split(',')
        
        decrypted_message = decrypt_message(encrypted_message, sequence)
        if decrypted_message:
            print(f"{Fore.YELLOW}Decrypted message: {decrypted_message}{Style.RESET_ALL}")
        else:
            print(Fore.RED + "Decryption failed. No valid key found." + Style.RESET_ALL)

if __name__ == "__main__":
    user_interface()
