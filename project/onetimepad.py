import os

class OneTimePad:
    
    def xor(a, b):
        res = bytes([x^y for (x,y) in zip(a,b)])
        return res
    
    def encrypt(plaintext, key):
        ciphertext = OneTimePad.xor(plaintext, key)
        return ciphertext
    
    def decrypt(ciphertext, key):
        plaintext = OneTimePad.xor(ciphertext, key)
        return plaintext