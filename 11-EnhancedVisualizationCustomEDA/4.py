from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
import os
import shutil

console = Console()

def main():
    while True:
        console.print(Panel("In-Place Encryption/Decryption Tool", style="bold blue"))
        choice = console.input("[1] Encrypt file\n[2] Decrypt file\n[3] Exit\nEnter choice: ")
        if choice == '1':
            process_in_place(encrypt=True)
        elif choice == '2':
            process_in_place(encrypt=False)
        elif choice == '3':
            break
        else:
            console.print("Invalid choice", style="bold red")

def process_in_place(encrypt=True):
    file_path = console.input("Enter file path: ")
    temp_path = f"{file_path}.tmp"
    
    try:
        if encrypt:
            console.print("Starting encryption...", style="yellow")
            generate_and_save_keys()
            
            with open(file_path, "rb") as f_in, open(temp_path, "wb") as f_out:
                encrypt_data(f_in, f_out)
                
        else:
            console.print("Starting decryption...", style="yellow")
            private_key = load_private_key()
            
            with open(file_path, "rb") as f_in, open(temp_path, "wb") as f_out:
                decrypt_data(f_in, f_out, private_key)
        
        # Atomic replacement
        shutil.copystat(file_path, temp_path)
        os.remove(file_path)
        os.rename(temp_path, file_path)
        console.print(f"Process complete: [bold]{file_path}[/bold]", style="bold green")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        console.print(f"Process failed: {str(e)}", style="bold red")
        raise

def generate_and_save_keys():
    console.print("Generating RSA keys...", style="yellow")
    key = RSA.generate(2048)
    private_key = key.export_key()
    
    with open("private.pem", "wb") as f:
        f.write(private_key)
    console.print("Private key saved to [bold]private.pem[/bold]. Keep this safe!", style="green")

def load_private_key():
    key_path = console.input("Enter private key path [private.pem]: ") or "private.pem"
    with open(key_path, "rb") as f:
        return RSA.import_key(f.read())

def encrypt_data(f_in, f_out):
    # Generate AES key and IV
    aes_key = get_random_bytes(32)
    iv = get_random_bytes(16)
    
    # Encrypt AES key with RSA
    rsa_cipher = PKCS1_OAEP.new(RSA.import_key(open("private.pem").read()))
    encrypted_aes_key = rsa_cipher.encrypt(aes_key)
    
    # Write header
    f_out.write(encrypted_aes_key)
    f_out.write(iv)
    
    # Encrypt data
    aes_cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    file_size = os.fstat(f_in.fileno()).st_size
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Encrypting") as pbar:
        while True:
            chunk = f_in.read(1024 * AES.block_size)
            if not chunk:
                break
            chunk = pad(chunk, AES.block_size) if len(chunk) % AES.block_size else chunk
            encrypted_chunk = aes_cipher.encrypt(chunk)
            f_out.write(encrypted_chunk)
            pbar.update(len(chunk))

def decrypt_data(f_in, f_out, private_key):
    # Read header
    rsa_cipher = PKCS1_OAEP.new(private_key)
    encrypted_aes_key = f_in.read(private_key.size_in_bytes())
    iv = f_in.read(16)
    
    # Decrypt AES key
    aes_key = rsa_cipher.decrypt(encrypted_aes_key)
    
    # Decrypt data
    aes_cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    file_size = os.fstat(f_in.fileno()).st_size - private_key.size_in_bytes() - 16
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Decrypting") as pbar:
        while True:
            chunk = f_in.read(1024 * AES.block_size)
            if not chunk:
                break
            decrypted_chunk = aes_cipher.decrypt(chunk)
            f_out.write(decrypted_chunk)
            pbar.update(len(chunk))
    
    # Remove padding
    f_out.seek(0)
    padded_data = f_out.getvalue()
    unpadded_data = unpad(padded_data, AES.block_size)
    f_out.seek(0)
    f_out.write(unpadded_data)
    f_out.truncate()

if __name__ == "__main__":
    main()