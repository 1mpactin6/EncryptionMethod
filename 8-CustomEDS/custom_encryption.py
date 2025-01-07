import base64, math, random, string, sys, time, os, codecs
from typing import Tuple, Union, List
from colorama import init, Fore, Back, Style

init()

TIPS = {
    "encryption": [
        "Use strong, unique keys for each encryption.",
        "Keep keys secure and private.",
        "Mix character types in your key.",
        "Avoid personal info in keys.",
        "Consider using passphrases.",
        "Longer keys generally provide better security.",
        "Use a combination of uppercase, lowercase, numbers, and symbols.",
        "Avoid common words or phrases.",
        "Don't reuse keys across different messages.",
        "Regularly update your encryption keys.",
    ],
    "decryption": [
        "Ensure correct key usage.",
        "Verify message integrity.",
        "Be cautious with unknown sources.",
        "Brute-forcing may be time-consuming.",
        "Verify decrypted message coherence.",
        "Double-check the key before decryption.",
        "Be aware of potential key transmission errors.",
        "Use secure channels to exchange keys.",
        "Consider using key derivation functions for added security.",
        "Implement multi-factor authentication for key access when possible.",
    ]
}

CONCEPTS = {
    "Base64": "Binary-to-text encoding scheme.",
    "Key": "Information used for encryption/decryption.",
    "Brute Force": "Guessing combinations via trial-and-error.",
    "UTF-8": "Unicode character encoding.",
    "Strong Key": "Complex, hard-to-guess encryption key.",
    "Passphrase": "Text sequence for access control.",
    "Key Length": "The number of bits in a key, affecting encryption strength.",
    "Key Entropy": "Measure of unpredictability in a key.",
    "Key Derivation": "Process of obtaining a key from a password or master key.",
    "Key Management": "Practices for secure handling of cryptographic keys.",
}

def print_panel(title: str, content: str, color: Fore = Fore.WHITE, width: int = 60):
    print(f"{color}{'=' * width}\n| {Style.BRIGHT}{title}{Style.RESET_ALL}{color}\n{'=' * width}")
    words = content.split()
    line = "| "
    for word in words:
        if len(line) + len(word) + 1 > width - 2:
            print(f"{color}{line:<{width-1}}|")
            line = "| " + word + " "
        else:
            line += word + " "
    if line:
        print(f"{color}{line:<{width-1}}|")
    print(f"{'=' * width}{Style.RESET_ALL}")

def animate_text(text: str, color: Fore = Fore.WHITE):
    for char in text:
        sys.stdout.write(f"{color}{char}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.02)
    print()

def print_random_tip(tips: List[str]):
    print(f"\n{Fore.YELLOW}Tip: {random.choice(tips)}{Style.RESET_ALL}")

def print_description(concept: str):
    if concept in CONCEPTS:
        print(f"\n{Fore.CYAN}What is {concept}?{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{CONCEPTS[concept]}{Style.RESET_ALL}")

class CustomEncryption:
    def __init__(self):
        self.compression_map = {}
        self.decompression_map = {}
        
    def generate_key(self, length: int = 16) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))
    
    def create_compression_maps(self, key: str) -> None:
        random.seed(key)
        chars = string.ascii_letters + string.digits + '+/='
        compressed = [''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 2))) for _ in range(len(chars))]
        self.compression_map = dict(zip(chars, compressed))
        self.decompression_map = {v: k for k, v in self.compression_map.items()}
    
    def encrypt(self, data: str, key: str = None) -> Tuple[str, str]:
        if key is None:
            key = self.generate_key()
        print_description("Base64")
        base64_encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        print_description("Key")
        self.create_compression_maps(key)
        compressed = ''.join(self.compression_map.get(c, c) for c in base64_encoded)
        bash_like = f"echo '{compressed}' | base64 -d | bash"
        print_random_tip(TIPS["encryption"])
        return bash_like, key
    
    def decrypt(self, encrypted_data: str, key: str = None) -> Union[str, List[str]]:
        if key is not None:
            return self._decrypt_with_key(encrypted_data, key)
        else:
            print_description("Brute Force")
            return self._brute_force_decrypt(encrypted_data)
    
    def _decrypt_with_key(self, encrypted_data: str, key: str) -> str:
        self.create_compression_maps(key)
        decompressed = ''
        i = 0
        while i < len(encrypted_data):
            for j in range(1, 3):
                chunk = encrypted_data[i:i+j]
                if chunk in self.decompression_map:
                    decompressed += self.decompression_map[chunk]
                    i += j
                    break
            else:
                decompressed += encrypted_data[i]
                i += 1
        try:
            decoded = base64.b64decode(decompressed).decode('utf-8')
            print_description("UTF-8")
            return decoded
        except:
            return "Invalid key or corrupted data"
    
    def _brute_force_decrypt(self, encrypted_data: str, max_attempts: int = 1000) -> List[str]:
        possible_decryptions = []
        for _ in range(max_attempts):
            key = self.generate_key()
            try:
                decrypted = self._decrypt_with_key(encrypted_data, key)
                if decrypted != "Invalid key or corrupted data":
                    possible_decryptions.append((decrypted, key))
            except:
                pass
        print_random_tip(TIPS["decryption"])
        return [f"Possible decryption (key={key}): {text}" for text, key in possible_decryptions]

def interactive_menu():
    while True:
        print_panel("CUSTOM ENCRYPTION SYSTEM", "Welcome! Choose an operation:", Fore.CYAN)
        print(f"{Fore.YELLOW}1. {Fore.WHITE}Encrypt\n{Fore.YELLOW}2. {Fore.WHITE}Decrypt\n{Fore.YELLOW}3. {Fore.WHITE}Brute Force Decrypt\n{Fore.YELLOW}4. {Fore.WHITE}Exit")
        choice = input(f"\n{Fore.GREEN}Choice (1-4): {Style.RESET_ALL}")
        encryptor = CustomEncryption()
        
        if choice == '1':
            print_panel("ENCRYPTION", "Enter message to encrypt:", Fore.BLUE)
            message = input(f"{Fore.WHITE}Message: {Style.RESET_ALL}")
            print_description("Strong Key")
            print_random_tip(TIPS["encryption"])
            use_custom_key = input(f"{Fore.YELLOW}Use custom key? (y/n): {Style.RESET_ALL}").lower() == 'y'
            if use_custom_key:
                print(f"\n{Fore.CYAN}Custom Key Guidelines:{Style.RESET_ALL}")
                print("1. Use a mix of uppercase, lowercase, numbers, and symbols.")
                print("2. Make it at least 16 characters long.")
                print("3. Avoid personal information or common words.")
                print("Example: 'P@ssw0rd' is weak. 'j8K#mL9$fR2@qX' is strong.")
                key = input(f"{Fore.WHITE}Enter your custom key: {Style.RESET_ALL}")
                while len(key) < 16:
                    print(f"{Fore.RED}Key too short. Please use at least 16 characters.{Style.RESET_ALL}")
                    key = input(f"{Fore.WHITE}Enter your custom key: {Style.RESET_ALL}")
            else:
                key = encryptor.generate_key()
            encrypted, used_key = encryptor.encrypt(message, key)
            print_panel("ENCRYPTION RESULT", "Encrypted message:", Fore.GREEN)
            print(f"{Fore.CYAN}Encrypted:{Style.RESET_ALL} {encrypted}")
            print(f"{Fore.RED}Key:{Style.RESET_ALL} {used_key}")
            print(f"\n{Fore.YELLOW}Keep this key safe for decryption!{Style.RESET_ALL}")
        elif choice == '2':
            print_panel("DECRYPTION", "Enter encrypted message and key:", Fore.MAGENTA)
            encrypted = input(f"{Fore.WHITE}Encrypted: {Style.RESET_ALL}")
            print_description("Key")
            print_random_tip(TIPS["decryption"])
            key = input(f"{Fore.WHITE}Key: {Style.RESET_ALL}")
            decrypted = encryptor.decrypt(encrypted, key)
            print_panel("DECRYPTION RESULT", "Decrypted message:", Fore.GREEN)
            print(f"{Fore.CYAN}Decrypted:{Style.RESET_ALL} {decrypted}")
        elif choice == '3':
            print_panel("BRUTE FORCE DECRYPTION", "Enter encrypted message:", Fore.RED)
            encrypted = input(f"{Fore.WHITE}Encrypted: {Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Attempting decryption... Please wait.{Style.RESET_ALL}")
            possible_decryptions = encryptor.decrypt(encrypted)
            print_panel("BRUTE FORCE RESULTS", "Possible decryptions:", Fore.RED)
            if possible_decryptions:
                for i, decryption in enumerate(possible_decryptions, 1):
                    print(f"\n{Fore.CYAN}Attempt {i}:{Style.RESET_ALL}\n{decryption}")
            else:
                print(f"{Fore.RED}No valid decryptions found.{Style.RESET_ALL}")
        elif choice == '4':
            animate_text("Thank you for using the Custom Encryption System!", Fore.CYAN)
            break
        else:
            print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")
            time.sleep(1)
        input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    interactive_menu()
