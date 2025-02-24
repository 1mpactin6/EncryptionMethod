from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes

data = 'secret data to transmit'.encode()

aes_key = get_random_bytes(16)
hmac_key = get_random_bytes(16)

cipher = AES.new(aes_key, AES.MODE_CTR)
ciphertext = cipher.encrypt(data)

hmac = HMAC.new(hmac_key, digestmod=SHA256)
tag = hmac.update(cipher.nonce + ciphertext).digest()

with open("encrypted.bin", "wb") as f:
    f.write(tag)
    f.write(cipher.nonce)
    f.write(ciphertext)

# Share securely aes_key and hmac_key with the receiver
# encrypted.bin can be sent over an unsecure channel


from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

data = 'secret data to transmit'.encode()

aes_key = get_random_bytes(16)

cipher = AES.new(aes_key, AES.MODE_OCB)
ciphertext, tag = cipher.encrypt_and_digest(data)
assert len(cipher.nonce) == 15

with open("encrypted.bin", "wb") as f:
    f.write(tag)
    f.write(cipher.nonce)
    f.write(ciphertext)

# Share securely aes_key with the receiver
# encrypted.bin can be sent over an unsecure channel

from Crypto.PublicKey import RSA

secret_code = "Unguessable"
key = RSA.generate(2048)
encrypted_key = key.export_key(passphrase=secret_code, pkcs=8,
                              protection="scryptAndAES128-CBC",
                              prot_params={'iteration_count':131072})

with open("rsa_key.bin", "wb") as f:
    f.write(encrypted_key)

print(key.publickey().export_key())

from Crypto.PublicKey import RSA

key = RSA.generate(2048)
private_key = key.export_key()
with open("private.pem", "wb") as f:
    f.write(private_key)

public_key = key.publickey().export_key()
with open("receiver.pem", "wb") as f:
    f.write(public_key)

from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP

data = "I met aliens in UFO. Here is the map.".encode("utf-8")

recipient_key = RSA.import_key(open("receiver.pem").read())
session_key = get_random_bytes(16)

# Encrypt the session key with the public RSA key

cipher_rsa = PKCS1_OAEP.new(recipient_key)
enc_session_key = cipher_rsa.encrypt(session_key)

# Encrypt the data with the AES session key

cipher_aes = AES.new(session_key, AES.MODE_EAX)
ciphertext, tag = cipher_aes.encrypt_and_digest(data)

with open("encrypted_data.bin", "wb") as f:
    f.write(enc_session_key)
    f.write(cipher_aes.nonce)
    f.write(tag)
    f.write(ciphertext)