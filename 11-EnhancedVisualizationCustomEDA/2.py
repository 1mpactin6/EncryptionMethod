# Import required modules
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
import base64
import json
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from tqdm import tqdm
import os

# Function to load or generate RSA keys
def load_or_generate_keys():
    """
    Loads existing RSA keys from files or generates new ones if they don't exist.
    Saves generated keys to 'private_key.pem' and 'public_key.pem'.
    """
    if os.path.exists("private_key.pem") and os.path.exists("public_key.pem"):
        with open("private_key.pem", "rb") as f:
            private_key = RSA.import_key(f.read())
        with open("public_key.pem", "rb") as f:
            public_key = RSA.import_key(f.read())
    else:
        key = RSA.generate(2048)  # Generate a 2048-bit RSA key pair
        private_key = key
        public_key = key.publickey()
        with open("private_key.pem", "wb") as f:
            f.write(private_key.export_key())
        with open("public_key.pem", "wb") as f:
            f.write(public_key.export_key())
    return private_key, public_key

# Function to encrypt a message
def encrypt_message(message):
    steps = 3
    with tqdm(total=steps, desc="Encrypting", ncols=100) as pbar:
        private_key, public_key = load_or_generate_keys()
        pbar.update(1)
        
        aes_key = get_random_bytes(32)
        rsa_cipher = PKCS1_OAEP.new(public_key)
        encrypted_aes_key = rsa_cipher.encrypt(aes_key)
        pbar.update(1)
        
        aes_cipher = AES.new(aes_key, AES.MODE_GCM)
        ciphertext, tag = aes_cipher.encrypt_and_digest(message.encode())
        nonce = aes_cipher.nonce
        pbar.update(1)
        
        concatenated = encrypted_aes_key + nonce + tag + ciphertext
        encoded_data = base64.b64encode(concatenated).decode()
    return encoded_data

# Function to decrypt a message
def decrypt_message(encoded_data):
    steps = 3
    with tqdm(total=steps, desc="Decrypting", ncols=100) as pbar:
        private_key, _ = load_or_generate_keys()
        pbar.update(1)
        
        concatenated = base64.b64decode(encoded_data)
        if len(concatenated) < 288:
            raise ValueError("Invalid encrypted data: too short")
        encrypted_aes_key = concatenated[:256]
        nonce = concatenated[256:272]
        tag = concatenated[272:288]
        ciphertext = concatenated[288:]
        pbar.update(1)
        
        rsa_cipher = PKCS1_OAEP.new(private_key)
        aes_key = rsa_cipher.decrypt(encrypted_aes_key)
        aes_cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        plaintext = aes_cipher.decrypt_and_verify(ciphertext, tag)
        pbar.update(1)
    return plaintext.decode()

# Main function to run the program
def main():
    console = Console()
    console.print(Panel(
        "Welcome to the Two-Layer Encryption/Decryption Tool!\n"
        "This program uses AES and RSA-OAEP to secure your messages.",
        title="Welcome",
        style="bold green"
    ))
    
    while True:
        console.print("\n[bold cyan]Menu:[/bold cyan]")
        console.print("1. Encrypt a message")
        console.print("2. Decrypt a message")
        console.print("3. Exit")
        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3"], default="1")
        
        if choice == "1":
            console.print(Panel(
                "Your message will be encrypted using AES-GCM with a randomly generated key.\n"
                "This key is then encrypted with RSA-OAEP using a public key.\n"
                "Tip: Keep the encrypted output safe to decrypt later!",
                title="Encryption Tip",
                style="bold blue"
            ))
            message = Prompt.ask("Enter the message to encrypt")
            encrypted_data = encrypt_message(message)
            console.print("[green]Encrypted data:[/green]")
            console.print(encrypted_data)
        
        elif choice == "2":
            console.print(Panel(
                "Provide the base64-encoded string from the encryption process.\n"
                "The program will decrypt it using the private key.\n"
                "Tip: Ensure you paste the full encrypted string accurately!",
                title="Decryption Tip",
                style="bold blue"
            ))
            encoded_data = Prompt.ask("Enter the encrypted data")
            try:
                decrypted_message = decrypt_message(encoded_data)
                console.print("[green]Decrypted message:[/green]")
                console.print(decrypted_message)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        elif choice == "3":
            console.print("[yellow]Goodbye! Your keys are saved for future use.[/yellow]")
            break

if __name__ == "__main__":
    main()