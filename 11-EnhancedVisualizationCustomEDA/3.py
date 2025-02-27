from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
import os

console = Console()

def main():
    while True:
        console.print(Panel("Encryption/Decryption Tool", style="bold blue"))
        choice = console.input("[1] Encrypt\n[2] Decrypt\n[3] Exit\nEnter choice: ")
        if choice == '1':
            encrypt_file()
        elif choice == '2':
            decrypt_file()
        elif choice == '3':
            break
        else:
            console.print("Invalid choice", style="bold red")

def encrypt_file():
    input_file = console.input("Enter file to encrypt: ")
    output_file = console.input("Enter encrypted output path: ")

    console.print("Generating RSA keys...", style="yellow")
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    
    priv_key_path = "private.pem"
    with open(priv_key_path, "wb") as f:
        f.write(private_key)
    console.print(f"Private key saved to [bold]{priv_key_path}[/bold]. Keep this safe!", style="green")

    aes_key = get_random_bytes(32)
    iv = get_random_bytes(16)
    
    rsa_cipher = PKCS1_OAEP.new(RSA.import_key(public_key))
    encrypted_aes_key = rsa_cipher.encrypt(aes_key)

    filesize = os.path.getsize(input_file)
    aes_cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(encrypted_aes_key)
        f_out.write(iv)
        
        with tqdm(total=filesize, unit='B', unit_scale=True, desc="Encrypting") as pbar:
            while True:
                chunk = f_in.read(1024 * AES.block_size)
                if not chunk:
                    break
                if len(chunk) % AES.block_size != 0:
                    chunk = pad(chunk, AES.block_size)
                encrypted_chunk = aes_cipher.encrypt(chunk)
                f_out.write(encrypted_chunk)
                pbar.update(len(chunk))
    
    console.print(f"Encryption complete: [bold]{output_file}[/bold]", style="bold green")

def decrypt_file():
    input_file = console.input("Enter encrypted file: ")
    output_file = console.input("Enter decrypted output path: ")
    private_key_path = console.input("Enter private key file: ")

    try:
        with open(private_key_path, "rb") as f:
            private_key = RSA.import_key(f.read())
    except Exception as e:
        console.print(f"Key load error: {e}", style="bold red")
        return

    try:
        with open(input_file, "rb") as f_in:
            encrypted_aes_key = f_in.read(private_key.size_in_bytes())
            iv = f_in.read(16)
            encrypted_data_size = os.path.getsize(input_file) - len(encrypted_aes_key) - len(iv)
            
            rsa_cipher = PKCS1_OAEP.new(private_key)
            aes_key = rsa_cipher.decrypt(encrypted_aes_key)
            
            aes_cipher = AES.new(aes_key, AES.MODE_CBC, iv)
            chunk_size = 1024 * AES.block_size
            decrypted_data = b''
            
            with tqdm(total=encrypted_data_size, unit='B', unit_scale=True, desc="Decrypting") as pbar:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    decrypted_chunk = aes_cipher.decrypt(chunk)
                    decrypted_data += decrypted_chunk
                    pbar.update(len(chunk))
            
            decrypted_data = unpad(decrypted_data, AES.block_size)
            
            with open(output_file, "wb") as f_out:
                f_out.write(decrypted_data)
            
            console.print(f"Decryption complete: [bold]{output_file}[/bold]", style="bold green")
    
    except Exception as e:
        console.print(f"Decryption failed: {e}", style="bold red")

if __name__ == "__main__":
    main()