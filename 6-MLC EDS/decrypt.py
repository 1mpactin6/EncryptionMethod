import base64
from Crypto.Cipher import AES, DES, Blowfish, DES3
from Crypto.Util.Padding import pad, unpad
from colorama import Fore, Style
from tqdm import tqdm
import time

# Use the same keys from encrypt.py
from encrypt import (AES_KEY_128, AES_KEY_192, AES_KEY_256, 
                    DES_KEY, BLOWFISH_KEY, TRIPLE_DES_KEY)

def log_message(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Decryption functions for each algorithm
def aes_decrypt_128(encrypted_data, key=AES_KEY_128):
    raw = base64.b64decode(encrypted_data.encode('utf-8'))
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')

def aes_decrypt_192(encrypted_data, key=AES_KEY_192):
    raw = base64.b64decode(encrypted_data.encode('utf-8'))
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')

def aes_decrypt_256(encrypted_data, key=AES_KEY_256):
    raw = base64.b64decode(encrypted_data.encode('utf-8'))
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')

def des_decrypt(encrypted_data, key=DES_KEY):
    raw = base64.b64decode(encrypted_data.encode('utf-8'))
    iv = raw[:8]
    ciphertext = raw[8:]
    cipher = DES.new(key, DES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), DES.block_size).decode('utf-8')

def blowfish_decrypt(encrypted_data, key=BLOWFISH_KEY):
    raw = base64.b64decode(encrypted_data.encode('utf-8'))
    iv = raw[:8]
    ciphertext = raw[8:]
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), Blowfish.block_size).decode('utf-8')

def triple_des_decrypt(encrypted_data, key=TRIPLE_DES_KEY):
    try:
        raw = base64.b64decode(encrypted_data.encode('utf-8'))
        iv = raw[:8]
        ciphertext = raw[8:]
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        decrypted = cipher.decrypt(ciphertext)
        # Add error handling for padding
        try:
            return unpad(decrypted, DES3.block_size).decode('utf-8')
        except ValueError:
            # If padding error occurs, try without unpadding
            return decrypted.decode('utf-8')
    except Exception as e:
        raise Exception(f"Triple DES decryption failed: {str(e)}")

def fpe_decrypt(data):
    # Reverse substitution cipher
    substitution = str.maketrans(
        'nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM9012345678',
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    return data.translate(substitution)

def convert_from_base(data, base):
    try:
        if base == 2:
            return ''.join(chr(int(data[i:i+8], 2)) for i in range(0, len(data), 8))
        elif base == 8:
            return ''.join(chr(int(data[i:i+3], 8)) for i in range(0, len(data), 3))
        elif base == 16:
            return bytes.fromhex(data).decode('utf-8')
        elif base == 64:
            return base64.b64decode(data.encode('utf-8')).decode('utf-8')
    except Exception as e:
        log_message(Fore.RED + f"Error in base conversion: {e}" + Style.RESET_ALL)
        return data
    return data

def decrypt_message(encrypted_message, sequence):
    log_message(Fore.GREEN + "Starting decryption process..." + Style.RESET_ALL)
    message = encrypted_message
    
    # Reverse the sequence for decryption
    sequence = sequence[::-1]
    
    for step in sequence:
        step = step.strip()
        
        for progress in tqdm(range(100), desc=f"Decrypting {step}", leave=False):
            time.sleep(0.02)
            
        try:
            if step == 'AES-128':
                message = aes_decrypt_128(message)
            elif step == 'AES-192':
                message = aes_decrypt_192(message)
            elif step == 'AES-256':
                message = aes_decrypt_256(message)
            elif step == 'DES':
                message = des_decrypt(message)
            elif step == 'Blowfish':
                message = blowfish_decrypt(message)
            elif step == 'TripleDES':
                message = triple_des_decrypt(message)
            elif step == 'FPE':
                message = fpe_decrypt(message)
            elif step == 'Base64':
                message = convert_from_base(message, 64)
            elif step == 'Binary':
                message = convert_from_base(message, 2)
            elif step == 'Octal':
                message = convert_from_base(message, 8)
            elif step == 'Hexadecimal':
                message = convert_from_base(message, 16)
                
            log_message(Fore.BLUE + f"{step} decryption applied" + Style.RESET_ALL)
            print(f"Current output: {message[:100]}..." if len(message) > 100 else f"Current output: {message}")
            
        except Exception as e:
            log_message(Fore.RED + f"Error decrypting with {step}: {e}" + Style.RESET_ALL)
            return None
            
        time.sleep(0.5)  # Reduced delay for better user experience
        
    log_message(Fore.GREEN + "Decryption complete." + Style.RESET_ALL)
    return message

def user_interface():
    print(Fore.CYAN + "Welcome to the Decryption Program!" + Style.RESET_ALL)
    print(Fore.YELLOW + "Note: Make sure to use the same encryption sequence used for encryption!" + Style.RESET_ALL)
    print("\n" + Fore.MAGENTA + "Available decryption algorithms: AES-128, AES-192, AES-256, DES, Blowfish, TripleDES, FPE, Base64, Binary, Octal, Hexadecimal" + Style.RESET_ALL)
    
    while True:
        encrypted_message = input("\nEnter the encrypted message (type 'exit' to quit): ")
        if encrypted_message.lower() == 'exit':
            log_message(Fore.RED + "Exiting the program." + Style.RESET_ALL)
            break
            
        sequence = input("Enter the original encryption sequence: ").split(',')
        
        decrypted_message = decrypt_message(encrypted_message, sequence)
        if decrypted_message:
            print(f"{Fore.GREEN}Decrypted message: {decrypted_message}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Decryption failed. Please check your input and sequence.{Style.RESET_ALL}")

if __name__ == "__main__":
    user_interface()
