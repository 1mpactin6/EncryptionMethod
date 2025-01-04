import random

# ANSI escape codes for colored output
GREEN = "\033[92m"  # Green text
BLUE = "\033[94m"   # Blue text
RED = "\033[91m"    # Red text
YELLOW = "\033[93m"  # Yellow text
RESET = "\033[0m"   # Reset to default color

def decrypt(encrypted_message, private_key):
    """
    Decrypts a message that was encrypted using the encrypt function.
    
    Args:
        encrypted_message (str): The encrypted message as a string of numbers.
        private_key (int): The private key used for decryption.
    
    Returns:
        str: The original message or an error message if input is invalid.
    """
    decrypted_message = ''
    try:
        numbers = encrypted_message.split()  # Split the encrypted message into numbers
        for number in numbers:
            # Reverse the custom conversion (subtracting 1000 and private key)
            binary = format(int(number) - 1000 - private_key, '08b')  # Convert back to binary
            decrypted_message += chr(int(binary, 2))  # Convert binary to character
        return decrypted_message
    except ValueError:
        return "Error: Invalid input. Please ensure you enter a valid encrypted message."

def rsa_decrypt(encrypted_message, public_key):
    """
    Simulates RSA decryption by subtracting the public key from the encrypted ASCII values.
    
    Args:
        encrypted_message (str): The RSA encrypted message as a string of numbers.
        public_key (int): The public key used for decryption.
    
    Returns:
        str: The original message or an error message if input is invalid.
    """
    decrypted_message = ''
    try:
        numbers = encrypted_message.split()  # Split the encrypted message into numbers
        for number in numbers:
            decrypted_message += chr(int(number) - public_key)  # Reverse the RSA-like encryption
        return decrypted_message
    except ValueError:
        return "Error: Invalid input. Please ensure you enter a valid encrypted message."

def brute_force_decrypt(encrypted_message):
    """
    Attempts to brute-force decrypt the message by trying different private keys.
    
    Args:
        encrypted_message (str): The encrypted message as a string of numbers.
    
    Returns:
        list: A list of possible decrypted messages.
    """
    possible_decryptions = {}
    for key in range(1, 101):  # Try private keys from 1 to 100
        decrypted_message = decrypt(encrypted_message, key)
        if "Error" not in decrypted_message:
            if decrypted_message in possible_decryptions:
                possible_decryptions[decrypted_message].append(key)
            else:
                possible_decryptions[decrypted_message] = [key]
    
    return possible_decryptions

def log_decryption():
    """
    Continuously decrypts messages and logs the original message or error messages.
    """
    print(f"{YELLOW}Welcome to the Decryption Tool!{RESET}")
    print(f"{YELLOW}This tool allows you to decrypt messages that were encrypted using a custom method and RSA-like encryption.{RESET}")
    print(f"{YELLOW}You can enter an encrypted message and optionally provide a public key for decryption.{RESET}")
    
    while True:
        encrypted_message = input("\nEnter an encrypted message to decrypt (or 'exit' to quit): ")
        if encrypted_message.lower() == 'exit':
            print(f"{YELLOW}Exiting the decryption tool. Goodbye!{RESET}")
            break
        public_key_input = input("Enter the public key (or press Enter to brute force): ")
        public_key = int(public_key_input) if public_key_input else None
        
        # Attempt decryption with the public key if provided
        if public_key:
            private_key = public_key - random.randint(1, 100)  # Simulate private key derivation
            result_with_key = decrypt(encrypted_message, private_key)
            rsa_result_with_key = rsa_decrypt(encrypted_message, public_key)

            # Check and print results with color coding
            if "Error" in result_with_key:
                print(f"{RED}Decryption with custom key failed: {result_with_key}{RESET}")
                print(f"{YELLOW}Tip: Ensure the encrypted message is valid and the private key is correct.{RESET}")
            else:
                print(f"{BLUE}Decrypted with public key (Custom): {result_with_key}{RESET}")
                print(f"{GREEN}Explanation: The custom decryption process involves reversing the encryption by subtracting the private key and converting the result back to characters.{RESET}")

            if "Error" in rsa_result_with_key:
                print(f"{RED}Decryption with public key (RSA) failed: {rsa_result_with_key}{RESET}")
                print(f"{YELLOW}Tip: Ensure the encrypted message is valid and the public key is correct.{RESET}")
            else:
                print(f"{BLUE}Decrypted with public key (RSA): {rsa_result_with_key}{RESET}")
                print(f"{GREEN}Explanation: The RSA decryption process involves subtracting the public key from the ASCII values of the characters to retrieve the original message.{RESET}")
        
        # Attempt brute-force decryption
        print(f"{YELLOW}Attempting brute-force decryption...{RESET}")
        brute_force_results = brute_force_decrypt(encrypted_message)
        
        # Determine if outputs are mostly the same
        unique_outputs = {msg: keys for msg, keys in brute_force_results.items() if len(keys) > 1}
        varied_outputs = {msg: keys for msg, keys in brute_force_results.items() if len(keys) == 1}

        if unique_outputs:
            print(f"{YELLOW}Brute-force results found with unique outputs. Type 'show' to display them or 'skip' to continue.{RESET}")
            user_choice = input("Your choice: ").strip().lower()
            if user_choice == 'show':
                for message, keys in unique_outputs.items():
                    print(f"{BLUE}Brute-force successful with keys {keys}: {message}{RESET}")
                    print(f"{GREEN}Explanation: This message was successfully decrypted using a brute-force approach with keys {keys}.{RESET}")
            else:
                print(f"{YELLOW}Unique brute-force results skipped.{RESET}")
        elif varied_outputs:
            print(f"{YELLOW}Brute-force results found but are varied. Type 'show' to display them or 'skip' to continue.{RESET}")
            user_choice = input("Your choice: ").strip().lower()
            if user_choice == 'show':
                for message, keys in varied_outputs.items():
                    print(f"{BLUE}Brute-force successful with key {keys[0]}: {message}{RESET}")
                    print(f"{GREEN}Explanation: This message was successfully decrypted using a brute-force approach with key {keys[0]}.{RESET}")
            else:
                print(f"{YELLOW}Varied brute-force results skipped.{RESET}")
        else:
            print(f"{RED}Brute-force decryption failed. No valid messages found.{RESET}")
            print(f"{YELLOW}Tip: The encrypted message may be invalid or the key range may need adjustment.{RESET}")

# Example usage
if __name__ == "__main__":
    log_decryption()
