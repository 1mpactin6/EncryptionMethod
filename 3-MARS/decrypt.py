import base64
from Crypto.Cipher import AES
import rsa

# ANSI escape codes for colored output and font styles
GREEN = "\033[92m"  # Green text
RED = "\033[91m"    # Red text
YELLOW = "\033[93m"  # Yellow text
RESET = "\033[0m"   # Reset to default color
BOLD = "\033[1m"    # Bold text
UNDERLINE = "\033[4m" # Underlined text

def decrypt_aes_binary_to_string(encrypted_base64, key):
    """Decrypts AES-encrypted data using the provided key."""
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_bytes = base64.b64decode(encrypted_base64)
    decrypted_bytes = cipher.decrypt(encrypted_bytes)
    return decrypted_bytes.decode('utf-8').rstrip()

def rsa_decrypt_string(encrypted_data, private_key):
    """Decrypts the data encrypted by rsa_encrypt_string using the provided private key."""
    decrypted_data = rsa.decrypt(encrypted_data, private_key)
    return decrypted_data.decode('utf-8')

def log_decryption():
    """Main function for logging and managing the decryption process."""
    print(f"{YELLOW}{BOLD}Welcome to the Decryption Tool!{RESET}")
    print(f"{YELLOW}This tool allows you to decrypt messages encrypted using AES and RSA.{RESET}")
    print(f"{YELLOW}{BOLD}Instructions:{RESET}")
    print(f"{YELLOW}- You will be prompted to enter the encrypted AES message (base64 encoded).")
    print(f"{YELLOW}- Then, provide the RSA-encrypted AES key (also base64).")
    print(f"{YELLOW}- Finally, provide the RSA private key in the required format.{RESET}")
    
    while True:
        print(f"\n{BOLD}{YELLOW}=========================={RESET}")
        print(f"{BOLD}{YELLOW}DECRYPTION PROCESS STARTED{RESET}")
        print(f"{BOLD}{YELLOW}=========================={RESET}")

        encrypted_aes_input = input(f"\n{BOLD}{GREEN}Enter the encrypted AES message (base64):{RESET} ")
        if encrypted_aes_input.lower() == 'exit':
            print(f"\n{YELLOW}Exiting the decryption tool. Goodbye!{RESET}")
            break
        
        encrypted_rsa_key_input = input(f"{BOLD}{GREEN}Enter the encrypted RSA key (base64):{RESET} ")
        if encrypted_rsa_key_input.lower() == 'exit':
            print(f"\n{YELLOW}Exiting the decryption tool. Goodbye!{RESET}")
            break
        
        private_key_input = input(f"{BOLD}{GREEN}Enter the private key (format: 'modulus,exponent,d,p,q'):{RESET} ")
        if private_key_input.lower() == 'exit':
            print(f"\n{YELLOW}Exiting the decryption tool. Goodbye!{RESET}")
            break
        
        # Create the private key directly from the provided values
        try:
            private_key_parts = private_key_input.split(',')
            private_key = rsa.PrivateKey(int(private_key_parts[0]), int(private_key_parts[1]), int(private_key_parts[2]), 
                                         int(private_key_parts[3]), int(private_key_parts[4]))
        except Exception as e:
            print(f"{RED}{BOLD}Error: Invalid private key format. Please enter it correctly.{RESET}")
            continue
        
        try:
            # Decrypt the RSA key (AES key)
            print(f"\n{BOLD}{YELLOW}Decrypting RSA-encrypted AES key...{RESET}")
            encrypted_rsa_key_bytes = base64.b64decode(encrypted_rsa_key_input)  # Decode from base64
            decrypted_aes_key = rsa_decrypt_string(encrypted_rsa_key_bytes, private_key)
            aes_key = base64.b64decode(decrypted_aes_key)  # Decode the AES key from base64
            
            # Decrypt the AES message
            print(f"{BOLD}{YELLOW}Decrypting AES-encrypted message...{RESET}")
            decrypted_aes = decrypt_aes_binary_to_string(encrypted_aes_input, aes_key)
            print(f"{GREEN}{BOLD}Decrypted AES Message: {RESET}{decrypted_aes}")
        
        except Exception as e:
            print(f"{RED}{BOLD}Error during decryption: {str(e)}{RESET}")
            continue

# Example usage
if __name__ == "__main__":
    log_decryption()
