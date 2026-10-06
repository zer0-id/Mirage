import os
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tpm2_pytss import ESAPI, TPM2B_PRIVATE, TPM2B_PUBLIC, TPM2B_SENSITIVE_CREATE, TPMS_SENSITIVE_CREATE
from tpm2_pytss.constants import TPMA_OBJECT
from mirage.errors import VaultFormatError, VaultIntegrityError
from mirage.storage.context import build_context
from mirage.storage.tpmutil import primary
from . import MODE_SEAL
from .base import Backend, Entry, b64e, get_bytes, get_len

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
SEAL_ATTRS = TPMA_OBJECT.FIXEDTPM | TPMA_OBJECT.FIXEDPARENT | TPMA_OBJECT.USERWITHAUTH


class SealBackend(Backend):
    mode = MODE_SEAL

    def encrypt(self, username: str, data: bytes) -> Entry:
        key = os.urandom(KEY_BYTES)
        nonce = os.urandom(NONCE_BYTES)
        pub = TPM2B_PUBLIC.parse("keyedhash", objectAttributes=SEAL_ATTRS)
        sens = TPM2B_SENSITIVE_CREATE(sensitive=TPMS_SENSITIVE_CREATE(data=key))

        with ESAPI(self._tcti) as ectx, primary(ectx) as prim:
            private, public = ectx.create(prim, sens, pub)[0:2]

        private_b, public_b = private.marshal(), public.marshal()
        ctx = build_context(MODE_SEAL, username, len(data), sealed_private=private_b, sealed_public=public_b)
        ct = AESGCM(key).encrypt(nonce, data, ctx)

        return {
            "mode": MODE_SEAL,
            "len": len(data),
            "sealed_private": b64e(private_b),
            "sealed_public": b64e(public_b),
            "nonce": b64e(nonce),
            "ct": b64e(ct),
        }

    def decrypt(self, username: str, entry: Entry) -> bytes:
        self._require_mode(entry)

        original_len = get_len(entry)
        private_b = get_bytes(entry, "sealed_private")
        public_b = get_bytes(entry, "sealed_public")
        nonce = get_bytes(entry, "nonce")
        ct = get_bytes(entry, "ct")

        if original_len <= 0 or original_len % 4 != 0 or len(nonce) != NONCE_BYTES or len(ct) != original_len + TAG_BYTES:
            raise VaultFormatError("vault entry has an invalid length")

        try:
            private, _ = TPM2B_PRIVATE.unmarshal(private_b)
            public, _ = TPM2B_PUBLIC.unmarshal(public_b)
        except Exception:
            raise VaultFormatError("malformed vault entry") from None

        ctx = build_context(MODE_SEAL, username, original_len, sealed_private=private_b, sealed_public=public_b)

        with ESAPI(self._tcti) as ectx, primary(ectx) as prim:
            handle = ectx.load(prim, private, public)
            try:
                key = bytes(ectx.unseal(handle))
            finally:
                ectx.flush_context(handle)

        if len(key) != KEY_BYTES:
            raise VaultFormatError("malformed vault entry")
        try:
            return AESGCM(key).decrypt(nonce, ct, ctx)
        except InvalidTag:
            raise VaultIntegrityError("vault entry failed integrity check, possible tampering or corruption!") from None
