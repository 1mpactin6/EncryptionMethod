# Add imports for additional encryption types
from Crypto.Cipher import AES, DES, Blowfish, DES3
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import unpad
import base64
import rsa
import time
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init()

def decrypt_message_basic(encrypted_message, private_key):
    """
    Decrypts a message that was encrypted using the basic encrypt function.
    
    Args:
        encrypted_message (str): The encrypted message as a string of numbers.
        private_key (int): The private key used for decryption.
    
    Returns:
        str: The original message or an error message if input is invalid.
    """
    decrypted_message = ''
    try:
        numbers = encrypted_message.split()
        for number in numbers:
            # Reverse the custom conversion
            binary = format(int(number) - 1000 - private_key, '08b')
            decrypted_message += chr(int(binary, 2))
        return decrypted_message
    except ValueError:
        return "Error: Invalid input. Please ensure you enter a valid encrypted message."

def decrypt_aes(encrypted_base64, key, mode=AES.MODE_ECB):
    """
    Decrypts AES-encrypted data.
    
    Args:
        encrypted_base64 (str): Base64 encoded encrypted data
        key (bytes): AES key
        mode (int): AES mode (default: ECB)
    
    Returns:
        str: Decrypted message
    """
    try:
        cipher = AES.new(key, mode)
        encrypted_bytes = base64.b64decode(encrypted_base64)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        return unpad(decrypted_bytes, AES.block_size).decode('utf-8')
    except Exception as e:
        return f"Error during AES decryption: {str(e)}"

def decrypt_des(encrypted_base64, key):
    """Decrypts DES-encrypted data."""
    try:
        cipher = DES.new(key, DES.MODE_ECB)
        encrypted_bytes = base64.b64decode(encrypted_base64)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        return unpad(decrypted_bytes, DES.block_size).decode('utf-8')
    except Exception as e:
        return f"Error during DES decryption: {str(e)}"

def decrypt_blowfish(encrypted_base64, key):
    """Decrypts Blowfish-encrypted data."""
    try:
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        encrypted_bytes = base64.b64decode(encrypted_base64)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        return unpad(decrypted_bytes, Blowfish.block_size).decode('utf-8')
    except Exception as e:
        return f"Error during Blowfish decryption: {str(e)}"

def decrypt_triple_des(encrypted_base64, key):
    """Decrypts Triple DES-encrypted data."""
    try:
        cipher = DES3.new(key, DES3.MODE_ECB)
        encrypted_bytes = base64.b64decode(encrypted_base64)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        return unpad(decrypted_bytes, DES3.block_size).decode('utf-8')
    except Exception as e:
        return f"Error during Triple DES decryption: {str(e)}"

def decrypt_rsa(encrypted_data, private_key):
    """Decrypts RSA-encrypted data."""
    try:
        decrypted_data = rsa.decrypt(encrypted_data, private_key)
        return decrypted_data.decode('utf-8')
    except Exception as e:
        return f"Error during RSA decryption: {str(e)}"

def multi_layer_decrypt(encrypted_data, keys):
    """
    Performs multi-layer decryption using multiple algorithms.
    
    Args:
        encrypted_data (dict): Dictionary containing encrypted data for each layer
        keys (dict): Dictionary containing keys for each encryption type
    
    Returns:
        str: Final decrypted message
    """
    try:
        # Start with RSA-encrypted AES key
        aes_key = base64.b64decode(
            decrypt_rsa(base64.b64decode(encrypted_data['rsa']), keys['rsa_private'])
        )
        
        # Decrypt AES layer
        message = decrypt_aes(encrypted_data['aes'], aes_key)
        
        # If there are additional layers, decrypt them
        if 'des' in encrypted_data:
            message = decrypt_des(message, keys['des'])
        if 'blowfish' in encrypted_data:
            message = decrypt_blowfish(message, keys['blowfish'])
        if 'triple_des' in encrypted_data:
            message = decrypt_triple_des(message, keys['triple_des'])
            
        return message
    except Exception as e:
        return f"Error during multi-layer decryption: {str(e)}"

def main():
    print(Fore.GREEN + "\nWelcome to the Advanced Multi-layer Decryption System!" + Style.RESET_ALL)
    print("================================================")
    
    while True:
        print("\nOptions:")
        print("1. Basic Decryption")
        print("2. AES Decryption")
        print("3. Multi-layer Decryption")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "4":
            print("Exiting the program.")
            break
            
        elif choice == "1":
            try:
                encrypted_message = input("Enter the encrypted message: ")
                private_key = int(input("Enter the private key: "))
                result = decrypt_message_basic(encrypted_message, private_key)
                print(f"{Fore.GREEN}Decrypted message: {result}{Style.RESET_ALL}")
            except ValueError as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                
        elif choice == "2":
            try:
                encrypted_base64 = input("Enter the AES encrypted message (base64): ")
                key = input("Enter the AES key (base64): ").encode('utf-8')
                key = base64.b64decode(key)
                result = decrypt_aes(encrypted_base64, key)
                print(f"{Fore.GREEN}Decrypted message: {result}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                
        elif choice == "3":
            try:
                # Get encrypted data
                encrypted_data = {
                    'aes': input("Enter AES encrypted message (base64): "),
                    'rsa': input("Enter RSA encrypted key (base64): ")
                }
                
                # Get keys
                private_key_input = input("Enter RSA private key components (n,e,d,p,q): ")
                n, e, d, p, q = map(int, private_key_input.split(','))
                rsa_private_key = rsa.PrivateKey(n, e, d, p, q)
                
                keys = {'rsa_private': rsa_private_key}
                
                # Optional additional layers
                if input("Include DES layer? (y/n): ").lower() == 'y':
                    encrypted_data['des'] = input("Enter DES encrypted data (base64): ")
                    keys['des'] = base64.b64decode(input("Enter DES key (base64): "))
                
                result = multi_layer_decrypt(encrypted_data, keys)
                print(f"{Fore.GREEN}Final decrypted message: {result}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error during multi-layer decryption: {str(e)}{Style.RESET_ALL}")
        
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
