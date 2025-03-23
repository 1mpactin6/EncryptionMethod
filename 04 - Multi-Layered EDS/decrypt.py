import base64
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

# Colors for visual clarity
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

# AES Decryption
def decrypt_aes(aes_iv, aes_encrypted, aes_key):
    cipher = AES.new(aes_key, AES.MODE_CBC, base64.b64decode(aes_iv))
    decrypted = cipher.decrypt(base64.b64decode(aes_encrypted))
    return unpad(decrypted).decode('utf-8', errors='ignore')  # Ignore errors for non-UTF-8 content

# DES Decryption
def decrypt_des(des_encrypted, des_key):
    cipher = DES.new(des_key, DES.MODE_ECB)
    decrypted = cipher.decrypt(base64.b64decode(des_encrypted))
    return unpad(decrypted).decode('utf-8', errors='ignore')  # Ignore errors for non-UTF-8 content

# RSA Decryption
def decrypt_rsa(rsa_encrypted, rsa_private_key):
    try:
        cipher = PKCS1_OAEP.new(rsa_private_key)
        decrypted = cipher.decrypt(base64.b64decode(rsa_encrypted))
        return decrypted.decode('utf-8', errors='ignore')  # Ignore errors for non-UTF-8 content
    except Exception as e:
        print(f"Error during RSA decryption: {e}")
        return None

# MARS Decryption (using AES as placeholder)
def decrypt_mars(mars_encrypted, mars_key):
    cipher = AES.new(mars_key, AES.MODE_ECB)  # Simulating MARS with AES
    decrypted = cipher.decrypt(base64.b64decode(mars_encrypted))
    return unpad(decrypted).decode('utf-8', errors='ignore')  # Ignore errors for non-UTF-8 content

# Base64 Decoding
def base64_decode(base64_encoded):
    try:
        return base64.b64decode(base64_encoded).decode('utf-8', errors='ignore')  # Ignore errors for non-UTF-8 content
    except Exception as e:
        print(f"Error during Base64 decoding: {e}")
        return None

# FPE Decryption (simulated: reverse the transformation)
def decrypt_fpe(fpe_encrypted, fpe_key):
    return fpe_encrypted[len(fpe_key):][::-1]  # Reverse the string from the encryption

# Unpadding for AES and DES (removes padding after decryption)
def unpad(data):
    padding_length = data[-1]
    return data[:-padding_length]

# Multi-Layer Decryption Function
def multi_layer_decryption(encrypted_data, rsa_private_key, aes_key, des_key, mars_key, fpe_key):
    print(GREEN + "\nStarting multi-layer decryption..." + RESET)

    # Unpack encrypted data
    aes_iv, aes_encrypted, des_encrypted, rsa_encrypted, mars_encrypted, base64_encoded, fpe_encrypted = encrypted_data

    # Decrypt AES
    aes_decrypted = decrypt_aes(aes_iv, aes_encrypted, aes_key)
    print(YELLOW + f"AES Decrypted: {aes_decrypted}" + RESET)

    # Decrypt DES
    des_decrypted = decrypt_des(des_encrypted, des_key)
    print(YELLOW + f"DES Decrypted: {des_decrypted}" + RESET)

    # Decrypt RSA
    rsa_decrypted = decrypt_rsa(rsa_encrypted, rsa_private_key)
    if rsa_decrypted:
        print(YELLOW + f"RSA Decrypted: {rsa_decrypted}" + RESET)
    else:
        print("RSA Decryption failed.")

    # Decrypt MARS (simulated with AES)
    mars_decrypted = decrypt_mars(mars_encrypted, mars_key)
    print(YELLOW + f"MARS Decrypted (simulated with AES): {mars_decrypted}" + RESET)

    # Base64 Decoding
    base64_decoded = base64_decode(base64_encoded)
    if base64_decoded:
        print(YELLOW + f"Base64 Decoded: {base64_decoded}" + RESET)
    else:
        print("Base64 Decoding failed.")

    # Decrypt FPE
    fpe_decrypted = decrypt_fpe(fpe_encrypted, fpe_key)
    print(YELLOW + f"FPE Decrypted: {fpe_decrypted}" + RESET)

    return fpe_decrypted  # The final decrypted output

# Main function
def main():
    print(GREEN + "\nWelcome to the Multi-layer Decryption System!" + RESET)
    print("===============================================")
    print("Options:")
    print("1. Decrypt Data")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        # Assuming these keys were generated during encryption
        key = RSA.generate(2048)
        rsa_private_key = key

        # Generate random keys for AES, DES, and MARS (same as encryption)
        aes_key = get_random_bytes(16)  # AES 128-bit key
        des_key = get_random_bytes(8)   # DES 64-bit key
        mars_key = get_random_bytes(16)  # MARS 128-bit key (simulated with AES)
        fpe_key = "FPE1234"  # A simple FPE key (same as during encryption)

        # Example encrypted data (replace with real data after encryption)
        encrypted_data = (
            "cBODvCql2mB8qql7G4GbUg==",  # aes_iv
            "A1GqT5Z7gUdwkRjfl0gsDA==",  # aes_encrypted
            "Y6J2xIu4n9g3mVzH2mOkCg==",  # des_encrypted
            "RlgfhX49V56udfj2faBOg==",  # rsa_encrypted (make sure this is a valid base64 string)
            "2s7YhxOqFOowihFB7gN4dw==",  # mars_encrypted
            "b4W8vGE5LhDmlkjfhg5Frw==",  # base64_encoded
            "FPE1234wqoG5TfWtxycsA=="  # fpe_encrypted
        )

        # Perform multi-layer decryption
        decrypted_data = multi_layer_decryption(encrypted_data, rsa_private_key, aes_key, des_key, mars_key, fpe_key)

if __name__ == "__main__":
    main()
