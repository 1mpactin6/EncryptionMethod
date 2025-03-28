import os
import base64
import json
from pathlib import Path
from typing import Union, Optional
from getpass import getpass
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

console = Console()

class Vaultic:
    def __init__(self):
        self.salt = os.urandom(16)
        self.iterations = 100000
        self.backend = default_backend()
        
    def _derive_key(self, password: str, key_length: int = 32) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=self.salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return kdf.derive(password.encode())
    
    def _encrypt_aes(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-CTR"""
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(key),
            modes.CTR(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + ciphertext
    
    def _decrypt_aes(self, data: bytes, key: bytes) -> bytes:
        """Decrypt AES-256-CTR encrypted data"""
        iv = data[:16]
        ciphertext = data[16:]
        cipher = Cipher(
            algorithms.AES(key),
            modes.CTR(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _encrypt_chacha(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using ChaCha20"""
        nonce = os.urandom(16)
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return nonce + ciphertext
    
    def _decrypt_chacha(self, data: bytes, key: bytes) -> bytes:
        """Decrypt ChaCha20 encrypted data"""
        nonce = data[:16]
        ciphertext = data[16:]
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_data(self, data: Union[str, bytes], method: str = "aes") -> str:
        """Encrypt data with selected method"""
        if isinstance(data, str):
            data = data.encode()
            
        password = getpass("Enter encryption password: ")
        key = self._derive_key(password)
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Encrypting...", total=1)
            
            if method.lower() == "aes":
                encrypted = self._encrypt_aes(data, key)
            elif method.lower() == "chacha":
                encrypted = self._encrypt_chacha(data, key)
            else:
                raise ValueError("Invalid encryption method")
            
            progress.update(task, advance=1)
        
        # Store salt + method + encrypted data
        result = {
            'salt': base64.b64encode(self.salt).decode(),
            'method': method,
            'data': base64.b64encode(encrypted).decode()
        }
        return base64.b64encode(json.dumps(result).encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, bytes]:
        """Decrypt data"""
        try:
            result = json.loads(base64.b64decode(encrypted_data).decode())
            self.salt = base64.b64decode(result['salt'])
            method = result['method']
            data = base64.b64decode(result['data'])
            
            password = getpass("Enter decryption password: ")
            key = self._derive_key(password)
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Decrypting...", total=1)
                
                if method == "aes":
                    decrypted = self._decrypt_aes(data, key)
                elif method == "chacha":
                    decrypted = self._decrypt_chacha(data, key)
                else:
                    raise ValueError("Unknown encryption method")
                
                progress.update(task, advance=1)
            
            try:
                return decrypted.decode()
            except UnicodeDecodeError:
                return decrypted
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return b""
    
    def encrypt_file(self, file_path: str, method: str = "aes") -> str:
        """Encrypt a file"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.encrypt_data(data, method)
        
        output_path = f"{file_path}.vaultic"
        with open(output_path, 'w') as f:
            f.write(encrypted)
        
        return output_path
    
    def decrypt_file(self, file_path: str) -> Optional[str]:
        """Decrypt a file"""
        with open(file_path, 'r') as f:
            encrypted_data = f.read()
        
        decrypted = self.decrypt_data(encrypted_data)
        
        if not decrypted:
            return None
            
        if file_path.endswith('.vaultic'):
            output_path = file_path[:-8]  # Remove .vaultic extension
        else:
            output_path = f"{file_path}.decrypted"
        
        with open(output_path, 'wb') as f:
            if isinstance(decrypted, str):
                f.write(decrypted.encode())
            else:
                f.write(decrypted)
        
        return output_path
    
    def encrypt_folder(self, folder_path: str, method: str = "aes") -> None:
        """Encrypt all files in a folder"""
        folder = Path(folder_path)
        files = [f for f in folder.glob('**/*') if f.is_file()]
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Encrypting folder...", total=len(files))
            
            for file in files:
                if not file.name.endswith('.vaultic'):
                    self.encrypt_file(str(file), method)
                progress.update(task, advance=1)
    
    def decrypt_folder(self, folder_path: str) -> None:
        """Decrypt all .vaultic files in a folder"""
        folder = Path(folder_path)
        files = [f for f in folder.glob('**/*.vaultic') if f.is_file()]
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Decrypting folder...", total=len(files))
            
            for file in files:
                self.decrypt_file(str(file))
                progress.update(task, advance=1)

def show_menu():
    """Display the main menu"""
    console.print(Panel.fit("Vaultic - Secure Encryption System", style="bold blue"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="dim")
    table.add_column("Description")
    
    table.add_row("1", "Encrypt text/data")
    table.add_row("2", "Decrypt text/data")
    table.add_row("3", "Encrypt file")
    table.add_row("4", "Decrypt file")
    table.add_row("5", "Encrypt folder")
    table.add_row("6", "Decrypt folder")
    table.add_row("7", "Exit")
    
    console.print(table)

def demo():
    """Run a demo of the encryption/decryption process"""
    vaultic = Vaultic()
    console.print("\n[bold]Running Vaultic Demo:[/bold]")
    
    # Test data
    test_data = "This is a secret message!"
    test_file = "test.txt"
    test_folder = "test_folder"
    
    # Create test files
    os.makedirs(test_folder, exist_ok=True)
    with open(test_file, 'w') as f:
        f.write(test_data)
    with open(f"{test_folder}/file1.txt", 'w') as f:
        f.write("File 1 content")
    with open(f"{test_folder}/file2.txt", 'w') as f:
        f.write("File 2 content")
    
    # Demo 1: Text encryption
    console.print("\n[bold cyan]1. Text Encryption/Decryption:[/bold cyan]")
    encrypted = vaultic.encrypt_data(test_data, "aes")
    console.print(f"[green]Encrypted:[/green] {encrypted[:50]}...")
    decrypted = vaultic.decrypt_data(encrypted)
    console.print(f"[green]Decrypted:[/green] {decrypted}")
    
    # Demo 2: File encryption
    console.print("\n[bold cyan]2. File Encryption/Decryption:[/bold cyan]")
    encrypted_file = vaultic.encrypt_file(test_file, "chacha")
    console.print(f"[green]Encrypted file created:[/green] {encrypted_file}")
    decrypted_file = vaultic.decrypt_file(encrypted_file)
    console.print(f"[green]Decrypted file created:[/green] {decrypted_file}")
    
    # Demo 3: Folder encryption
    console.print("\n[bold cyan]3. Folder Encryption/Decryption:[/bold cyan]")
    vaultic.encrypt_folder(test_folder, "aes")
    console.print("[green]Folder encrypted successfully[/green]")
    vaultic.decrypt_folder(test_folder)
    console.print("[green]Folder decrypted successfully[/green]")
    
    # Cleanup
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    for f in Path(test_folder).glob('*'):
        f.unlink()
    os.rmdir(test_folder)

if __name__ == "__main__":
    demo()
