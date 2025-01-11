import base64, random, string, sys, time
from typing import Tuple, Union, List
from colorama import init, Fore, Style

init()

TIPS = {
    "encryption": [
        "Use strong, unique keys for each encryption.",
        "Keep keys secure and private.",
        "Mix character types in your key.",
        "Avoid personal info in keys.",
        "Consider using passphrases.",
    ],
    "decryption": [
        "Ensure correct key usage.",
        "Verify message integrity.",
        "Be cautious with unknown sources.",
        "Verify decrypted message coherence.",
        "Double-check the key before decryption.",
    ]
}

CONCEPTS = {
    "Base64": "Binary-to-text encoding scheme.",
    "Key": "Information used for encryption/decryption.",
    "Brute Force": "Guessing combinations via trial-and-error.",
    "Strong Key": "Complex, hard-to-guess encryption key.",
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
        self.frequency_map = {}
        
    def generate_key(self, length: int = 16) -> str:
        salt = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        key = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))
        return f"{salt}${key}"
    
    def create_compression_maps(self, key: str) -> None:
        if '$' in key:
            salt, main_key = key.split('$')
        else:
            main_key = key
        random.seed(main_key)
        chars = string.ascii_letters + string.digits + '+/='
        compressed = [''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 2))) for _ in range(len(chars))]
        self.compression_map = dict(zip(chars, compressed))
        self.decompression_map = {v: k for k, v in self.compression_map.items()}
    
    def run_length_encode(self, data: str) -> str:
        if not data:
            return ""
        encoded = []
        count = 1
        current = data[0]
        
        for char in data[1:]:
            if char == current:
                count += 1
            else:
                if count > 3:  # Only use RLE if it saves space
                    encoded.append(f"${count}{current}")
                else:
                    encoded.append(current * count)
                current = char
                count = 1
                
        if count > 3:
            encoded.append(f"${count}{current}")
        else:
            encoded.append(current * count)
            
        return ''.join(encoded)
    
    def run_length_decode(self, data: str) -> str:
        result = []
        i = 0
        while i < len(data):
            if data[i] == '$':
                # Find the end of the number
                j = i + 1
                while j < len(data) and data[j].isdigit():
                    j += 1
                if j < len(data):
                    count = int(data[i+1:j])
                    result.append(data[j] * count)
                    i = j + 1
                else:
                    result.append(data[i])
                    i += 1
            else:
                result.append(data[i])
                i += 1
        return ''.join(result)
    
    def encrypt(self, data: str, key: str = None) -> Tuple[str, str]:
        if key is None:
            key = self.generate_key()
            
        # Step 1: Base64 encode the input data
        base64_encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        
        # Step 2: Create compression maps with key
        self.create_compression_maps(key)
        
        # Step 3: Simple XOR encryption with key
        key_bytes = key.encode('utf-8')
        key_len = len(key_bytes)
        encoded_bytes = base64_encoded.encode('utf-8')
        encrypted = bytes(encoded_bytes[i] ^ key_bytes[i % key_len] for i in range(len(encoded_bytes)))
        
        # Step 4: Convert to base85 for shorter output
        compressed = base64.b85encode(encrypted).decode('utf-8')
        
        # Step 5: Run-length encoding for repeated characters
        final_output = self.run_length_encode(compressed)
        
        # Add checksum for integrity
        checksum = sum(ord(c) for c in final_output) % 256
        final_output = f"{checksum:02x}{final_output}"
        
        return final_output, key
    
    def decrypt(self, encrypted_data: str, key: str = None) -> Union[str, List[str]]:
        if key is not None:
            return self._decrypt_with_key(encrypted_data, key)
        else:
            return self._brute_force_decrypt(encrypted_data)
    
    def _decrypt_with_key(self, encrypted_data: str, key: str) -> str:
        try:
            # Verify checksum
            if len(encrypted_data) < 2:
                return "Invalid encrypted data"
            checksum_str = encrypted_data[:2]
            encrypted_data = encrypted_data[2:]
            
            calculated_checksum = sum(ord(c) for c in encrypted_data) % 256
            if format(calculated_checksum, '02x') != checksum_str:
                return "Data integrity check failed"
            
            # Decode run-length encoding
            decoded_rle = self.run_length_decode(encrypted_data)
            
            # Decode base85
            try:
                encrypted = base64.b85decode(decoded_rle.encode('utf-8'))
            except:
                return "Invalid base85 encoding"
            
            # XOR decrypt with key
            key_bytes = key.encode('utf-8')
            key_len = len(key_bytes)
            decrypted = bytes(encrypted[i] ^ key_bytes[i % key_len] for i in range(len(encrypted)))
            
            try:
                decoded_str = decrypted.decode('utf-8')
                final_decoded = base64.b64decode(decoded_str).decode('utf-8')
                return final_decoded
            except:
                return "Invalid base64 encoding or corrupted data"
        except Exception as e:
            return f"Decryption error: {str(e)}"
    
    def is_likely_text(self, text: str) -> bool:
        """Check if a string is likely to be a meaningful text message."""
        if not text:
            return False
            
        # Check if string contains only printable characters
        if not all(32 <= ord(c) <= 126 for c in text):
            return False
            
        # Check character distribution
        char_count = len(text)
        if char_count < 2:  # Too short to be meaningful
            return False
            
        # Count different types of characters
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        punctuation = sum(c in string.punctuation for c in text)
        
        # Calculate ratios
        letter_ratio = letters / char_count
        digit_ratio = digits / char_count
        space_ratio = spaces / char_count
        punct_ratio = punctuation / char_count
        
        # Heuristics for text-like content
        if letter_ratio > 0.9:  # Mostly letters - likely a word
            return True
        if digit_ratio > 0.9:  # Mostly digits - likely a number
            return True
        if letter_ratio + digit_ratio > 0.8:  # Mix of letters and numbers
            return True
        if (letter_ratio > 0.5 and space_ratio > 0 and space_ratio < 0.3):  # Looks like a sentence
            return True
            
        # Check for common patterns
        if any(pattern in text.lower() for pattern in ['the', 'and', 'ing', 'ion']):
            return True
            
        # Check for repeated characters (might be garbage)
        from collections import Counter
        char_freq = Counter(text)
        most_common_ratio = char_freq.most_common(1)[0][1] / char_count
        if most_common_ratio > 0.5:  # Too many repeated characters
            return False
            
        return False
    
    def score_text(self, text: str) -> float:
        """Score how likely a text is to be a valid message."""
        if not text:
            return 0.0
            
        score = 0.0
        char_count = len(text)
        
        # Basic character type scoring
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        
        letter_ratio = letters / char_count
        digit_ratio = digits / char_count
        space_ratio = spaces / char_count
        
        # Score based on character distribution
        score += letter_ratio * 0.4  # Letters are good
        score += digit_ratio * 0.3   # Numbers are okay
        score += space_ratio * 0.2   # Some spaces are good
        
        # Penalize non-printable characters
        non_printable = sum(not (32 <= ord(c) <= 126) for c in text)
        score -= (non_printable / char_count) * 0.5
        
        # Bonus for common word patterns
        common_patterns = ['the', 'and', 'ing', 'ion', 'ed', 'er', 'es', 'or', 'th']
        text_lower = text.lower()
        pattern_matches = sum(pattern in text_lower for pattern in common_patterns)
        score += pattern_matches * 0.1
        
        # Bonus for balanced character distribution
        char_freq = Counter(text)
        unique_chars = len(char_freq)
        if unique_chars > 3:  # More variety is better
            score += min(0.2, unique_chars / char_count)
            
        return min(1.0, max(0.0, score))  # Normalize between 0 and 1
    
    def _brute_force_decrypt(self, encrypted_data: str, max_attempts: int = 1000) -> List[str]:
        possible_decryptions = []
        attempts = 0
        
        while attempts < max_attempts:
            key = self.generate_key()
            try:
                decrypted = self._decrypt_with_key(encrypted_data, key)
                if isinstance(decrypted, str) and not decrypted.startswith("Invalid") and not decrypted.startswith("Decryption error"):
                    # Score the decrypted text
                    if self.is_likely_text(decrypted):
                        score = self.score_text(decrypted)
                        if score > 0.3:  # Only keep results with decent scores
                            possible_decryptions.append((decrypted, key, score))
            except:
                pass
            attempts += 1
        
        # Sort results by score
        possible_decryptions.sort(key=lambda x: x[2], reverse=True)
        
        # Format results with confidence levels
        results = []
        for decrypted, key, score in possible_decryptions[:5]:  # Show top 5 results
            confidence = "High" if score > 0.6 else "Medium" if score > 0.4 else "Low"
            results.append(f"Possible decryption (Confidence: {confidence})\nKey: {key}\nMessage: {decrypted}\nScore: {score:.2f}")
        
        print_random_tip(TIPS["decryption"])
        return results if results else ["No likely messages found in brute force attempt"]

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
                key = input(f"{Fore.WHITE}Enter your custom key (min 16 chars): {Style.RESET_ALL}")
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
            
            # Test decryption
            decrypted = encryptor.decrypt(encrypted, used_key)
            print("\nTesting decryption...")
            print(f"Original message: {message}")
            print(f"Decrypted message: {decrypted}")
            if message == decrypted:
                print(f"{Fore.GREEN}Decryption successful!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Decryption failed.{Style.RESET_ALL}")
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
