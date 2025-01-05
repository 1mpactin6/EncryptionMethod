import time
from datetime import datetime
from Crypto.Random import get_random_bytes
import base64
import rsa
from Crypto.Cipher import AES

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

def main():
    # Generate RSA keys (Public and Private)
    (public_key, private_key) = rsa.newkeys(2048)  # Generate a new RSA key pair

    while True:
        try:
            user_input = input("Enter a string to encrypt (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting the program.")
                break
            
            log_message = get_formatted_log(f"Encrypting: {user_input}")
            print(log_message)
            
            print("In progress...")

            # Encrypt the input string using AES
            encrypted_aes, aes_key = encrypt_string_to_aes_binary(user_input)

            # Encrypt the AES key using RSA
            encrypted_rsa_key = rsa_encrypt_string(base64.b64encode(aes_key).decode('utf-8'), public_key)
            
            print(f"\033[92mEncrypted AES: {encrypted_aes}\033[0m")  # Green text for success
            print(f"Encrypted RSA Key: {base64.b64encode(encrypted_rsa_key).decode('utf-8')}")  # Encrypted AES key
            print(f"RSA Public Key: {public_key}")  # Public key for decryption later
            print(f"RSA Private Key: {private_key}")  # Private key for later decryption
        
        except Exception as e:
            print(f"\033[91mError: {str(e)}\033[0m")  # Red text for error
            continue  # Continue the loop to allow for new input

# Call the main function to start the program
if __name__ == "__main__":
    main()
