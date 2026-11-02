import base64
import unittest

from cryptography.fernet import Fernet
from secret_crypto import APIKeyCipher, SecretDecryptionError


class APIKeyCipherTests(unittest.TestCase):
    def setUp(self):
        self.root_key = Fernet.generate_key()
        self.cipher = APIKeyCipher(self.root_key)

    def test_aes_gcm_round_trip(self):
        secret = "test-provider-key-123"
        token = self.cipher.encrypt(secret)
        self.assertTrue(token.startswith(APIKeyCipher.PREFIX))
        self.assertEqual(self.cipher.decrypt(token), secret)

    def test_each_encryption_uses_a_fresh_nonce(self):
        one = self.cipher.encrypt("same-secret")
        two = self.cipher.encrypt("same-secret")
        self.assertNotEqual(one, two)
        self.assertEqual(self.cipher.decrypt(one), "same-secret")
        self.assertEqual(self.cipher.decrypt(two), "same-secret")

    def test_legacy_fernet_values_remain_readable(self):
        legacy_token = Fernet(self.root_key).encrypt(b"legacy-api-key").decode("ascii")
        self.assertEqual(self.cipher.decrypt(legacy_token), "legacy-api-key")

    def test_tampered_gcm_payload_is_rejected(self):
        token = self.cipher.encrypt("secret")
        packed = bytearray(base64.urlsafe_b64decode(token[len(APIKeyCipher.PREFIX):]))
        packed[-1] ^= 0x01
        tampered = APIKeyCipher.PREFIX + base64.urlsafe_b64encode(packed).decode("ascii")
        with self.assertRaises(SecretDecryptionError):
            self.cipher.decrypt(tampered)

    def test_wrong_root_key_is_rejected(self):
        token = self.cipher.encrypt("secret")
        other = APIKeyCipher(Fernet.generate_key())
        with self.assertRaises(SecretDecryptionError):
            other.decrypt(token)


if __name__ == "__main__":
    unittest.main()
