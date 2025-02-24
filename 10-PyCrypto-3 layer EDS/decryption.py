import sys
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256

# Somehow, the receiver securely get aes_key and hmac_key
# encrypted.bin can be sent over an unsecure channel

with open("encrypted.bin", "rb") as f:
    tag = f.read(32)
    nonce = f.read(8)
    ciphertext = f.read()

try:
    hmac = HMAC.new(hmac_key, digestmod=SHA256)
    tag = hmac.update(nonce + ciphertext).verify(tag)
except ValueError:
    print("The message was modified!")
    sys.exit(1)

cipher = AES.new(aes_key, AES.MODE_CTR, nonce=nonce)
message = cipher.decrypt(ciphertext)
print("Message:", message.decode())

from Crypto.PublicKey import RSA

secret_code = "Unguessable"
encoded_key = open("rsa_key.bin", "rb").read()
key = RSA.import_key(encoded_key, passphrase=secret_code)

print(key.publickey().export_key())

from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

private_key = RSA.import_key(open("private.pem").read())

with open("encrypted_data.bin", "rb") as f:
    enc_session_key = f.read(private_key.size_in_bytes())
    nonce = f.read(16)
    tag = f.read(16)
    ciphertext = f.read()

# Decrypt the session key with the private RSA key
cipher_rsa = PKCS1_OAEP.new(private_key)
session_key = cipher_rsa.decrypt(enc_session_key)

# Decrypt the data with the AES session key
cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
data = cipher_aes.decrypt_and_verify(ciphertext, tag)
print(data.decode("utf-8"))