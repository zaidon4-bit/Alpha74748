"""Versioned, backwards-compatible API-key encryption for Alpha AI.

New values use AES-256-GCM with a fresh 96-bit nonce and authenticated context.
Legacy Fernet values are decrypt-only and remain readable without data migration.
"""

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class SecretDecryptionError(ValueError):
    """Raised when a stored API secret cannot be authenticated or decoded."""


class APIKeyCipher:
    PREFIX = "a256gcm:v1:"
    NONCE_BYTES = 12
    TAG_BYTES = 16
    HKDF_SALT = b"alpha-ai/api-key-encryption/v1"
    HKDF_INFO = b"alpha-ai storage key | AES-256-GCM | v1"
    AAD = b"alpha-ai:users-and-model-api-routes:api-key:v1"

    def __init__(self, fernet_key):
        if isinstance(fernet_key, str):
            fernet_key = fernet_key.encode("ascii")
        # Keep the original Fernet key solely for decrypting pre-v1 records.
        self._legacy = Fernet(fernet_key)
        try:
            key_material = base64.urlsafe_b64decode(fernet_key)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid API encryption root key") from exc
        if len(key_material) != 32:
            raise ValueError("API encryption root key must decode to exactly 32 bytes")
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.HKDF_SALT,
            info=self.HKDF_INFO,
        ).derive(key_material)
        self._aead = AESGCM(derived_key)

    def encrypt(self, plaintext):
        if not isinstance(plaintext, str):
            raise TypeError("API key must be text")
        nonce = os.urandom(self.NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, plaintext.encode("utf-8"), self.AAD)
        token = base64.urlsafe_b64encode(nonce + sealed).decode("ascii")
        return self.PREFIX + token

    def decrypt(self, token):
        if not isinstance(token, str):
            raise SecretDecryptionError("Encrypted API key must be text")
        if token.startswith(self.PREFIX):
            encoded = token[len(self.PREFIX):]
            try:
                packed = base64.b64decode(
                    encoded.encode("ascii"), altchars=b"-_", validate=True
                )
                if len(packed) < self.NONCE_BYTES + self.TAG_BYTES:
                    raise ValueError("Encrypted value is too short")
                nonce = packed[:self.NONCE_BYTES]
                sealed = packed[self.NONCE_BYTES:]
                plaintext = self._aead.decrypt(nonce, sealed, self.AAD)
                return plaintext.decode("utf-8")
            except (InvalidTag, ValueError, UnicodeError, binascii.Error) as exc:
                raise SecretDecryptionError("AES-256-GCM authentication failed") from exc

        # Backward compatibility only: new secrets are never written as Fernet.
        try:
            return self._legacy.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError, UnicodeError, binascii.Error) as exc:
            raise SecretDecryptionError("Legacy Fernet decryption failed") from exc
