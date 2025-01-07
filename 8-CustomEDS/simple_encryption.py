import base64
import random
import string
from colorama import init, Fore, Style
import sys

# Initialize colorama for Windows
init()

class SimpleEncryption:
    def __init__(self):
        """Initialize the encryption system with a simpler approach"""
        self.key_length = 16
        self.char_set = string.ascii_letters + string.digits + "!@#$%^&*"
        
    def generate_key(self) -> str:
        """Generate a simple but secure key"""
        return ''.join(random.choices(self.char_set, k=self.key_length))
    
    def custom_encode(self, data: bytes) -> str:
        """Custom encoding to avoid bash-like output"""
        # Convert to base64 first
        b64 = base64.b64encode(data).decode('utf-8')
        # Replace characters that might look like commands
        replacements = {
            '+': '-plus-',
            '/': '-slash-',
            '=': '-equals-',
            '|': '-pipe-',
            '>': '-gt-',
            '<': '-lt-',
            ';': '-semi-',
            '&': '-and-',
            '$': '-dollar-'
        }
        for old, new in replacements.items():
            b64 = b64.replace(old, new)
        return b64
    
    def custom_decode(self, encoded: str) -> bytes:
        """Decode the custom encoded string back to bytes"""
        # Reverse the character replacements
        replacements = {
            '-plus-': '+',
            '-slash-': '/',
            '-equals-': '=',
            '-pipe-': '|',
            '-gt-': '>',
            '-lt-': '<',
            '-semi-': ';',
            '-and-': '&',
            '-dollar-': '$'
        }
        for old, new in replacements.items():
            encoded = encoded.replace(old, new)
        return base64.b64decode(encoded)
    
    def encrypt(self, message: str, key: str = None) -> tuple[str, str]:
        """
        Encrypt a message using a simple XOR-based encryption
        
        Args:
            message: The message to encrypt
            key: Optional encryption key (will be generated if not provided)
            
        Returns:
            tuple: (encrypted_message, key)
        """
        if key is None:
            key = self.generate_key()
            
        # Convert message to bytes
        message_bytes = message.encode('utf-8')
        key_bytes = key.encode('utf-8')
        
        # XOR each byte with the key (repeating the key if necessary)
        encrypted_bytes = bytearray()
        for i in range(len(message_bytes)):
            encrypted_bytes.append(message_bytes[i] ^ key_bytes[i % len(key_bytes)])
        
        # Use custom encoding to avoid bash-like output
        encrypted = self.custom_encode(encrypted_bytes)
        return encrypted, key
    
    def decrypt(self, encrypted_message: str, key: str) -> str:
        """
        Decrypt a message using the provided key
        
        Args:
            encrypted_message: The encrypted message
            key: The encryption key used to encrypt the message
            
        Returns:
            str: The decrypted message
        """
        try:
            # Decode using custom decoder
            encrypted_bytes = self.custom_decode(encrypted_message)
            key_bytes = key.encode('utf-8')
            
            # XOR each byte with the key (repeating the key if necessary)
            decrypted_bytes = bytearray()
            for i in range(len(encrypted_bytes)):
                decrypted_bytes.append(encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)])
            
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            return f"Decryption failed: Please check your key and encrypted message"

def print_panel(title: str, content: str = "", color: Fore = Fore.WHITE):
    """Print a simple panel with title and content"""
    width = 70
    print(f"\n{color}{'=' * width}")
    print(f"  {Style.BRIGHT}{title}{Style.RESET_ALL}")
    if content:
        print(f"{color}{content}")
    print(f"{'=' * width}{Style.RESET_ALL}\n")

def show_tips(topic: str):
    """Show helpful tips based on the context"""
    tips = {
        "encryption": [
            "Tips for secure encryption:",
            "1. Use a strong, unique key for each message",
            "2. Don't share your encryption key in the same channel as the encrypted message",
            "3. Longer messages are more secure than shorter ones",
            "4. The encrypted message will contain '-plus-', '-slash-', etc. - this is normal!"
        ],
        "decryption": [
            "Tips for decryption:",
            "1. Make sure to use the exact key that was used for encryption",
            "2. The key is case-sensitive",
            "3. Don't modify the encrypted message - copy and paste it exactly",
            "4. Keep your encryption keys in a secure place"
        ]
    }
    print(f"{Fore.YELLOW}")
    for tip in tips.get(topic, []):
        print(tip)
    print(f"{Style.RESET_ALL}")

def main():
    """Main interactive menu"""
    encryptor = SimpleEncryption()
    
    while True:
        print_panel("SIMPLE ENCRYPTION SYSTEM", 
                   "A straightforward and secure way to encrypt your messages",
                   Fore.CYAN)
        
        print(f"{Fore.GREEN}1. {Fore.WHITE}Encrypt a message")
        print(f"{Fore.GREEN}2. {Fore.WHITE}Decrypt a message")
        print(f"{Fore.GREEN}3. {Fore.WHITE}Exit")
        
        choice = input(f"\n{Fore.CYAN}Choose an option (1-3): {Style.RESET_ALL}")
        
        if choice == '1':
            print_panel("ENCRYPTION", "Enter your message below:", Fore.BLUE)
            show_tips("encryption")
            
            message = input(f"{Fore.WHITE}Message: {Style.RESET_ALL}")
            encrypted, key = encryptor.encrypt(message)
            
            print_panel("ENCRYPTION RESULT", Fore.GREEN)
            print(f"{Fore.CYAN}Encrypted Message: {Style.RESET_ALL}{encrypted}")
            print(f"{Fore.RED}Encryption Key: {Style.RESET_ALL}{key}")
            print(f"\n{Fore.YELLOW}Important: Save this key! You'll need it to decrypt your message.{Style.RESET_ALL}")
            
        elif choice == '2':
            print_panel("DECRYPTION", "Enter the encrypted message and key:", Fore.MAGENTA)
            show_tips("decryption")
            
            encrypted = input(f"{Fore.WHITE}Encrypted message: {Style.RESET_ALL}")
            key = input(f"{Fore.WHITE}Encryption key: {Style.RESET_ALL}")
            
            decrypted = encryptor.decrypt(encrypted, key)
            print_panel("DECRYPTION RESULT", Fore.GREEN)
            print(f"{Fore.CYAN}Decrypted Message: {Style.RESET_ALL}{decrypted}")
            
        elif choice == '3':
            print_panel("GOODBYE!", "Thank you for using Simple Encryption System", Fore.CYAN)
            break
            
        input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
        print("\n" * 2)

if __name__ == "__main__":
    # Configure console for UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    main()
