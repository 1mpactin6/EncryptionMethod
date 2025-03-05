# Import necessary libraries
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import json
import os
import logging
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

# Configure logging for detailed tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='encryption_tool.log'
)
logger = logging.getLogger(__name__)

# Global configuration dictionary for settings
config = {
    "aes_key_size": 16,  # Options: 16 (AES-128), 24 (AES-192), 32 (AES-256)
    "progress_bar": True,
    "progress_accuracy": 2,  # Decimal places for progress bar
    "delete_original": False,  # Privacy control: delete original files
    "default_key_dir": "./keys",
    "animation_speed": 0.1,  # Seconds per animation frame
}

# Hardcoded key and IV for login encryption (16 bytes each)
LOGIN_KEY = b'sixteen byte key'
LOGIN_IV = b'sixteen byte ivv'

# Encrypt the credentials "1mpactin9" (username and password) - precomputed
# To generate these, run the encryption once and hardcode the results
ENCRYPTED_USERNAME = "Q2L8Z5Kj9pXvM7nB2sT4wA=="  # Example base64 ciphertext
ENCRYPTED_PASSWORD = ""  # Example base64 ciphertext

# Initialize Rich console
console = Console()

# --- Utility Functions ---

def encrypt_credentials(input_str):
    """
    Encrypt input string using AES-CBC with hardcoded key and IV.
    
    Args:
        input_str (str): The string to encrypt
    
    Returns:
        str: Base64-encoded encrypted string
    """
    cipher = AES.new(LOGIN_KEY, AES.MODE_CBC, LOGIN_IV)
    input_bytes = input_str.encode('utf-8')
    padded = pad(input_bytes, AES.block_size)
    encrypted = cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode('utf-8')

def show_spinner(message, duration=1):
    """
    Display a spinner animation for a given duration.
    
    Args:
        message (str): Message to display with spinner
        duration (float): Duration in seconds
    """
    with Live(Spinner("dots", text=Text(message, style="green")), refresh_per_second=20):
        time.sleep(duration)

# --- Login System ---

def login():
    """Handle user login with encrypted credential comparison."""
    console.print(Panel.fit("[bold blue]Welcome to the Encryption Tool[/bold blue]", border_style="cyan"))
    console.print("Please log in to access the tool.")
    console.print("Hint: Both username and password are '1mpactin9'.\n")
    
    show_spinner("Initializing login system...", 1)
    
    input_username = Prompt.ask("Enter username")
    input_password = Prompt.ask("Enter password", password=True)
    
    encrypted_input_username = encrypt_credentials(input_username)
    encrypted_input_password = encrypt_credentials(input_password)
    
    if encrypted_input_username == ENCRYPTED_USERNAME and encrypted_input_password == ENCRYPTED_PASSWORD:
        console.print("\n[green]Login successful![/green]")
        show_spinner("Loading main menu...", 1)
        return True
    else:
        console.print("\n[red]Invalid credentials. Access denied.[/red]")
        return False

# --- Key Management ---

def generate_rsa_keys():
    """Generate and save RSA key pair with user-specified settings."""
    console.print(Panel.fit("[bold blue]Generate RSA Key Pair[/bold blue]", border_style="cyan"))
    console.print("Generate secure RSA keys for encryption and decryption.")
    console.print("Tip: Larger key sizes (e.g., 4096) offer more security but are slower.\n")
    
    key_size = Prompt.ask(
        "Enter key size (1024, 2048, 4096)",
        choices=["1024", "2048", "4096"],
        default="2048"
    )
    key_size = int(key_size)
    
    private_key_path = Prompt.ask(
        "Enter path to save private key",
        default=os.path.join(config["default_key_dir"], "private_key.pem")
    )
    public_key_path = Prompt.ask(
        "Enter path to save public key",
        default=os.path.join(config["default_key_dir"], "public_key.pem")
    )
    
    password = Prompt.ask("Enter password to encrypt private key (optional)", password=True, default="")
    
    # Ensure key directory exists
    os.makedirs(os.path.dirname(private_key_path), exist_ok=True)
    
    console.print("Generating RSA key pair...")
    show_spinner("Generating keys...", 2)
    
    key = RSA.generate(key_size)
    private_key = key.export_key(
        passphrase=password if password else None,
        pkcs=8,
        protection="PBKDF2WithHMAC-SHA1AndAES256-CBC" if password else None
    )
    public_key = key.publickey().export_key()
    
    with open(private_key_path, "wb") as f:
        f.write(private_key)
    with open(public_key_path, "wb") as f:
        f.write(public_key)
    
    console.print(f"[green]Keys saved to {private_key_path} and {public_key_path}[/green]")
    logger.info(f"RSA keys generated: {private_key_path}, {public_key_path}")

# --- Encryption and Decryption Functions ---

def aes_encrypt(data, key):
    """
    Encrypt data with AES in CBC mode.
    
    Args:
        data (bytes): Data to encrypt
        key (bytes): AES key
    
    Returns:
        bytes: IV + encrypted data
    """
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data, AES.block_size)
    encrypted_data = cipher.encrypt(padded_data)
    return iv + encrypted_data

def aes_decrypt(encrypted_data, key):
    """
    Decrypt data with AES in CBC mode.
    
    Args:
        encrypted_data (bytes): IV + encrypted data
        key (bytes): AES key
    
    Returns:
        bytes: Decrypted data
    """
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = cipher.decrypt(ciphertext)
    return unpad(padded_data, AES.block_size)

def hybrid_encrypt(data, public_key_path):
    """Encrypt data using hybrid RSA + AES encryption."""
    with open(public_key_path, "rb") as f:
        public_key = RSA.import_key(f.read())
    
    aes_key = get_random_bytes(config["aes_key_size"])
    encrypted_data = aes_encrypt(data, aes_key)
    
    cipher_rsa = PKCS1_OAEP.new(public_key)
    encrypted_aes_key = cipher_rsa.encrypt(aes_key)
    
    return encrypted_aes_key + encrypted_data

def hybrid_decrypt(combined, private_key_path, password):
    """Decrypt data using hybrid RSA + AES decryption."""
    with open(private_key_path, "rb") as f:
        private_key = RSA.import_key(f.read(), passphrase=password or None)
    
    key_size_bytes = private_key.size_in_bytes()
    encrypted_aes_key = combined[:key_size_bytes]
    encrypted_data = combined[key_size_bytes:]
    
    cipher_rsa = PKCS1_OAEP.new(private_key)
    aes_key = cipher_rsa.decrypt(encrypted_aes_key)
    return aes_decrypt(encrypted_data, aes_key)

def aes_encrypt_file(input_file, output_file, aes_key):
    """Encrypt a file with AES, showing progress."""
    iv = get_random_bytes(16)
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    file_size = os.path.getsize(input_file)
    
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(iv)
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Encrypting file",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                padded_chunk = pad(chunk, AES.block_size) if len(chunk) < 1024 * 1024 else chunk
                encrypted_chunk = cipher.encrypt(padded_chunk)
                f_out.write(encrypted_chunk)
                pbar.update(len(chunk))

def aes_decrypt_file(input_file, output_file, aes_key):
    """Decrypt a file with AES."""
    with open(input_file, "rb") as f:
        encrypted_data = f.read()
    data = aes_decrypt(encrypted_data, aes_key)
    with open(output_file, "wb") as f:
        f.write(data)

def encrypt_file(input_file, output_file, public_key_path):
    """Encrypt a file using hybrid encryption."""
    logger.info(f"Encrypting file: {input_file}")
    data = open(input_file, "rb").read()
    encrypted_data = hybrid_encrypt(data, public_key_path)
    with open(output_file, "wb") as f:
        f.write(encrypted_data)
    if config["delete_original"]:
        os.remove(input_file)
        logger.info(f"Deleted original file: {input_file}")

def decrypt_file(input_file, output_file, private_key_path, password):
    """Decrypt a file using hybrid decryption."""
    logger.info(f"Decrypting file: {input_file}")
    combined = open(input_file, "rb").read()
    data = hybrid_decrypt(combined, private_key_path, password)
    with open(output_file, "wb") as f:
        f.write(data)

def encrypt_folder(input_folder, output_folder, public_key_path):
    """Encrypt all files in a folder with a single AES key."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    aes_key = get_random_bytes(config["aes_key_size"])
    with open(public_key_path, "rb") as f:
        public_key = RSA.import_key(f.read())
    cipher_rsa = PKCS1_OAEP.new(public_key)
    encrypted_aes_key = cipher_rsa.encrypt(aes_key)
    
    key_file = os.path.join(output_folder, "key.enc")
    with open(key_file, "wb") as f:
        f.write(encrypted_aes_key)
    
    file_count = sum(len(files) for _, _, files in os.walk(input_folder))
    with tqdm(total=file_count, desc="Encrypting folder", unit="file") as pbar:
        for root, _, files in os.walk(input_folder):
            for file in files:
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_folder)
                output_file = os.path.join(output_folder, relative_path + ".enc")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                aes_encrypt_file(input_file, output_file, aes_key)
                if config["delete_original"]:
                    os.remove(input_file)
                pbar.update(1)
    logger.info(f"Folder encrypted: {input_folder} -> {output_folder}")

def decrypt_folder(input_folder, output_folder, private_key_path, password):
    """Decrypt all files in a folder using the shared AES key."""
    key_file = os.path.join(input_folder, "key.enc")
    with open(key_file, "rb") as f:
        encrypted_aes_key = f.read()
    
    with open(private_key_path, "rb") as f:
        private_key = RSA.import_key(f.read(), passphrase=password or None)
    cipher_rsa = PKCS1_OAEP.new(private_key)
    aes_key = cipher_rsa.decrypt(encrypted_aes_key)
    
    file_count = sum(1 for _, _, files in os.walk(input_folder) for f in files if f.endswith(".enc"))
    with tqdm(total=file_count, desc="Decrypting folder", unit="file") as pbar:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".enc"):
                    input_file = os.path.join(root, file)
                    relative_path = os.path.relpath(input_file, input_folder)
                    output_file = os.path.join(output_folder, relative_path[:-4])
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    aes_decrypt_file(input_file, output_file, aes_key)
                    pbar.update(1)
    logger.info(f"Folder decrypted: {input_folder} -> {output_folder}")

# --- User Interface Functions ---

def encrypt_data_ui():
    """Encrypt direct data input."""
    console.print(Panel.fit("[bold blue]Encrypt Data[/bold blue]", border_style="cyan"))
    console.print("Enter text to encrypt securely.")
    console.print("Tip: Use a public key generated from the 'Generate Keys' option.\n")
    
    data = Prompt.ask("Enter the data to encrypt")
    public_key_path = Prompt.ask("Enter public key path", default=os.path.join(config["default_key_dir"], "public_key.pem"))
    
    try:
        show_spinner("Encrypting data...", 1)
        encrypted_data = hybrid_encrypt(data.encode('utf-8'), public_key_path)
        encrypted_str = base64.b64encode(encrypted_data).decode('utf-8')
        console.print("\n[green]Encrypted data (base64):[/green]")
        console.print(encrypted_str)
        logger.info("Data encrypted successfully")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Data encryption failed: {e}")

def decrypt_data_ui():
    """Decrypt direct data input."""
    console.print(Panel.fit("[bold blue]Decrypt Data[/bold blue]", border_style="cyan"))
    console.print("Enter base64-encoded encrypted data to decrypt.")
    console.print("Tip: Ensure you have the correct private key and password.\n")
    
    encrypted_str = Prompt.ask("Enter encrypted data (base64)")
    private_key_path = Prompt.ask("Enter private key path", default=os.path.join(config["default_key_dir"], "private_key.pem"))
    password = Prompt.ask("Enter private key password (if any)", password=True, default="")
    
    try:
        show_spinner("Decrypting data...", 1)
        encrypted_data = base64.b64decode(encrypted_str)
        data = hybrid_decrypt(encrypted_data, private_key_path, password)
        console.print("\n[green]Decrypted data:[/green]")
        console.print(data.decode('utf-8'))
        logger.info("Data decrypted successfully")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Data decryption failed: {e}")

def encrypt_file_ui():
    """Encrypt a single file."""
    console.print(Panel.fit("[bold blue]Encrypt File[/bold blue]", border_style="cyan"))
    console.print("Encrypt a file with hybrid encryption.")
    console.print("Tip: The encrypted file will include the AES key encrypted with RSA.\n")
    
    input_file = Prompt.ask("Enter input file path")
    output_file = Prompt.ask("Enter output file path", default=input_file + ".enc")
    public_key_path = Prompt.ask("Enter public key path", default=os.path.join(config["default_key_dir"], "public_key.pem"))
    
    try:
        show_spinner("Preparing file encryption...", 1)
        encrypt_file(input_file, output_file, public_key_path)
        console.print(f"\n[green]File encrypted to {output_file}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"File encryption failed: {e}")

def decrypt_file_ui():
    """Decrypt a single file."""
    console.print(Panel.fit("[bold blue]Decrypt File[/bold blue]", border_style="cyan"))
    console.print("Decrypt an encrypted file.")
    console.print("Tip: Provide the private key used during encryption.\n")
    
    input_file = Prompt.ask("Enter input file path")
    output_file = Prompt.ask("Enter output file path", default=input_file.rstrip(".enc"))
    private_key_path = Prompt.ask("Enter private key path", default=os.path.join(config["default_key_dir"], "private_key.pem"))
    password = Prompt.ask("Enter private key password (if any)", password=True, default="")
    
    try:
        show_spinner("Preparing file decryption...", 1)
        decrypt_file(input_file, output_file, private_key_path, password)
        console.print(f"\n[green]File decrypted to {output_file}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"File decryption failed: {e}")

def encrypt_folder_ui():
    """Encrypt an entire folder."""
    console.print(Panel.fit("[bold blue]Encrypt Folder[/bold blue]", border_style="cyan"))
    console.print("Encrypt all files in a folder with a single AES key.")
    console.print("Tip: A 'key.enc' file will be created with the encrypted AES key.\n")
    
    input_folder = Prompt.ask("Enter input folder path")
    output_folder = Prompt.ask("Enter output folder path", default=input_folder + "_encrypted")
    public_key_path = Prompt.ask("Enter public key path", default=os.path.join(config["default_key_dir"], "public_key.pem"))
    
    try:
        show_spinner("Preparing folder encryption...", 1)
        encrypt_folder(input_folder, output_folder, public_key_path)
        console.print(f"\n[green]Folder encrypted to {output_folder}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Folder encryption failed: {e}")

def decrypt_folder_ui():
    """Decrypt an entire folder."""
    console.print(Panel.fit("[bold blue]Decrypt Folder[/bold blue]", border_style="cyan"))
    console.print("Decrypt all files in a folder using the shared AES key.")
    console.print("Tip: Ensure 'key.enc' is present in the input folder.\n")
    
    input_folder = Prompt.ask("Enter input folder path")
    output_folder = Prompt.ask("Enter output folder path", default=input_folder.rstrip("_encrypted"))
    private_key_path = Prompt.ask("Enter private key path", default=os.path.join(config["default_key_dir"], "private_key.pem"))
    password = Prompt.ask("Enter private key password (if any)", password=True, default="")
    
    try:
        show_spinner("Preparing folder decryption...", 1)
        decrypt_folder(input_folder, output_folder, private_key_path, password)
        console.print(f"\n[green]Folder decrypted to {output_folder}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Folder decryption failed: {e}")

# --- Menu Functions ---

def encrypt_menu():
    """Display encryption options."""
    console.print(Panel.fit("[bold blue]Encryption Menu[/bold blue]", border_style="cyan"))
    console.print("Choose what to encrypt:")
    table = Table()
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    table.add_row("1", "Encrypt Data - Encrypt text input")
    table.add_row("2", "Encrypt File - Encrypt a single file")
    table.add_row("3", "Encrypt Folder - Encrypt all files in a folder")
    console.print(table)
    
    choice = Prompt.ask("Select an option", choices=["1", "2", "3"])
    if choice == "1":
        encrypt_data_ui()
    elif choice == "2":
        encrypt_file_ui()
    elif choice == "3":
        encrypt_folder_ui()

def decrypt_menu():
    """Display decryption options."""
    console.print(Panel.fit("[bold blue]Decryption Menu[/bold blue]", border_style="cyan"))
    console.print("Choose what to decrypt:")
    table = Table()
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    table.add_row("1", "Decrypt Data - Decrypt text input")
    table.add_row("2", "Decrypt File - Decrypt a single file")
    table.add_row("3", "Decrypt Folder - Decrypt all files in a folder")
    console.print(table)
    
    choice = Prompt.ask("Select an option", choices=["1", "2", "3"])
    if choice == "1":
        decrypt_data_ui()
    elif choice == "2":
        decrypt_file_ui()
    elif choice == "3":
        decrypt_folder_ui()

def settings_menu():
    """Display and manage settings."""
    console.print(Panel.fit("[bold blue]Settings Menu[/bold blue]", border_style="cyan"))
    console.print("Configure the encryption tool settings.\n")
    
    while True:
        table = Table(title="Current Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("AES Key Size", f"{config['aes_key_size']} bytes")
        table.add_row("Progress Bar", "Enabled" if config["progress_bar"] else "Disabled")
        table.add_row("Progress Accuracy", f"{config['progress_accuracy']} decimals")
        table.add_row("Delete Original Files", "Yes" if config["delete_original"] else "No")
        table.add_row("Default Key Directory", config["default_key_dir"])
        table.add_row("Animation Speed", f"{config['animation_speed']} seconds")
        console.print(table)
        
        console.print("\nOptions:")
        console.print("1. Change AES key size")
        console.print("2. Toggle progress bar")
        console.print("3. Set progress accuracy")
        console.print("4. Toggle delete original files")
        console.print("5. Set default key directory")
        console.print("6. Set animation speed")
        console.print("7. Back to main menu")
        
        choice = Prompt.ask("Select an option", choices=[str(i) for i in range(1, 8)])
        
        if choice == "1":
            size = Prompt.ask("Enter AES key size (16, 24, 32)", choices=["16", "24", "32"], default=str(config["aes_key_size"]))
            config["aes_key_size"] = int(size)
            console.print(f"[green]AES key size set to {size} bytes[/green]")
        elif choice == "2":
            config["progress_bar"] = not config["progress_bar"]
            console.print(f"[green]Progress bar {'enabled' if config['progress_bar'] else 'disabled'}[/green]")
        elif choice == "3":
            accuracy = Prompt.ask("Enter progress accuracy (0-5 decimals)", default=str(config["progress_accuracy"]))
            config["progress_accuracy"] = max(0, min(int(accuracy), 5))
            console.print(f"[green]Progress accuracy set to {config['progress_accuracy']} decimals[/green]")
        elif choice == "4":
            config["delete_original"] = not config["delete_original"]
            console.print(f"[green]Delete original files {'enabled' if config['delete_original'] else 'disabled'}[/green]")
        elif choice == "5":
            dir_path = Prompt.ask("Enter default key directory", default=config["default_key_dir"])
            config["default_key_dir"] = dir_path
            console.print(f"[green]Default key directory set to {dir_path}[/green]")
        elif choice == "6":
            speed = float(Prompt.ask("Enter animation speed (seconds)", default=str(config["animation_speed"])))
            config["animation_speed"] = max(0.05, min(speed, 2.0))
            console.print(f"[green]Animation speed set to {config['animation_speed']} seconds[/green]")
        elif choice == "7":
            break

def main_menu():
    """Display the main menu."""
    console.print(Panel.fit("[bold blue]Main Menu[/bold blue]", border_style="cyan"))
    console.print("Welcome to the Encryption Tool!")
    console.print("Select an option below:\n")
    
    table = Table()
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    table.add_row("1", "Encrypt - Encrypt data, files, or folders")
    table.add_row("2", "Decrypt - Decrypt data, files, or folders")
    table.add_row("3", "Generate Keys - Create RSA key pairs")
    table.add_row("4", "Settings - Configure tool options")
    table.add_row("5", "Exit - Close the application")
    console.print(table)
    
    return Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])

# --- Main Application ---

def main():
    """Run the encryption and decryption tool."""
    if not login():
        return
    
    console.print("\n[bold green]Access granted. Starting the tool...[/bold green]")
    time.sleep(1)
    
    while True:
        choice = main_menu()
        show_spinner("Processing your selection...", config["animation_speed"])
        
        if choice == "1":
            encrypt_menu()
        elif choice == "2":
            decrypt_menu()
        elif choice == "3":
            generate_rsa_keys()
        elif choice == "4":
            settings_menu()
        elif choice == "5":
            console.print("\n[bold yellow]Exiting the tool. Goodbye![/bold yellow]")
            show_spinner("Shutting down...", 1)
            break
        console.print("\n")

if __name__ == "__main__":
    main()