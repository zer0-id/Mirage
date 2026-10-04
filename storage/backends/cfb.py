import base64
import hmac
import os
from tpm2_pytss import ESAPI, TPM2B_PRIVATE, TPM2B_PUBLIC, TSS2_Exception
from tpm2_pytss.constants import ESYS_TR, TPM2_ALG, TPM2_RC, TSS2_RC
from errors import VaultFormatError, VaultIntegrityError
from storage.context import build_context
from storage.tpmutil import primary
from . import MODE_CFB
from .base import Backend, Entry, get_bytes, b64e, get_len

CHUNK_SIZE = 1024
AES_TEMPLATE = "aes128cfb"
HMAC_TEMPLATE = "keyedhash"


class CFBBackend(Backend):
    mode = MODE_CFB

    def _create_aes_key(self, ectx: ESAPI, primary_handle: ESYS_TR) -> tuple[TPM2B_PRIVATE, TPM2B_PUBLIC]:
        private, public = ectx.create(primary_handle, None, AES_TEMPLATE)[0:2]
        return private, public

    def _create_hmac_key(self, ectx: ESAPI, primary_handle: ESYS_TR) -> tuple[TPM2B_PRIVATE, TPM2B_PUBLIC]:
        private, public = ectx.create(primary_handle, None, HMAC_TEMPLATE)[0:2]
        return private, public

    def _compute_hmac(self, ectx: ESAPI, key_handle: ESYS_TR, data: bytes) -> bytes:
        seq = ectx.hmac_start(key_handle, b"", TPM2_ALG(TPM2_ALG.SHA256))
        chunks = [data[i : i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)] or [b""]
        for chunk in chunks[:-1]:
            ectx.sequence_update(seq, chunk)
        digest, _ = ectx.sequence_complete(seq, chunks[-1])
        return bytes(digest)

    def _encrypt_decrypt(self, ectx: ESAPI, key_handle: ESYS_TR, decrypt: bool, iv_in: bytes, in_data: bytes):
        mode = TPM2_ALG(TPM2_ALG.CFB)
        try:
            return ectx.encrypt_decrypt_2(key_handle, decrypt=decrypt, mode=mode, iv_in=iv_in, in_data=in_data)
        except TSS2_Exception as e:
            # TCG deprecated EncryptDecrypt (no "2"): fall back only when it is unimplemented.
            if (e.error & ~TSS2_RC.RC_LAYER_MASK) != TPM2_RC.COMMAND_CODE:
                raise
        return ectx.encrypt_decrypt(key_handle, decrypt=decrypt, mode=mode, iv_in=iv_in, in_data=in_data)

    def encrypt(self, username: str, data: bytes) -> Entry:
        initial_iv = iv = os.urandom(16)
        ciphertext = b""

        with ESAPI(self._tcti) as ectx, primary(ectx) as prim:
            aes_private, aes_public = self._create_aes_key(ectx, prim)
            aes = ectx.load(prim, aes_private, aes_public)
            try:
                for i in range(0, len(data), CHUNK_SIZE):
                    out, iv_out = self._encrypt_decrypt(ectx, aes, False, iv, data[i : i + CHUNK_SIZE])
                    ciphertext += bytes(out)
                    iv = bytes(iv_out)
            finally:
                ectx.flush_context(aes)

            hmac_private, hmac_public = self._create_hmac_key(ectx, prim)
            hmac_key = ectx.load(prim, hmac_private, hmac_public)
            try:
                ctx = build_context(
                    MODE_CFB,
                    username,
                    len(data),
                    private=aes_private.marshal(),
                    public=aes_public.marshal(),
                    iv=initial_iv,
                    ct=ciphertext,
                )
                mac = self._compute_hmac(ectx, hmac_key, ctx)
            finally:
                ectx.flush_context(hmac_key)

        return {
            "mode": MODE_CFB,
            "len": len(data),
            "private": b64e(aes_private.marshal()),
            "public": b64e(aes_public.marshal()),
            "iv": b64e(initial_iv),
            "ct": b64e(ciphertext),
            "hmac_private": b64e(hmac_private.marshal()),
            "hmac_public": b64e(hmac_public.marshal()),
            "mac": b64e(mac),
        }

    def decrypt(self, username: str, entry: Entry) -> bytes:
        self._require_mode(entry)

        original_len = entry.get("len")
        if not isinstance(original_len, int) or isinstance(original_len, bool):
            raise VaultFormatError("malformed vault entry")
        private_b = get_bytes(entry, "private")
        public_b = get_bytes(entry, "public")
        iv = get_bytes(entry, "iv")
        ciphertext = get_bytes(entry, "ct")
        hmac_private_b = get_bytes(entry, "hmac_private")
        hmac_public_b = get_bytes(entry, "hmac_public")
        expected_mac = get_bytes(entry, "mac")

        try:
            private, _ = TPM2B_PRIVATE.unmarshal(private_b)
            public, _ = TPM2B_PUBLIC.unmarshal(public_b)
            hmac_private, _ = TPM2B_PRIVATE.unmarshal(hmac_private_b)
            hmac_public, _ = TPM2B_PUBLIC.unmarshal(hmac_public_b)
        except Exception:
            raise VaultFormatError("malformed vault entry") from None

        ctx = build_context(
            MODE_CFB,
            username,
            original_len,
            private=private_b,
            public=public_b,
            iv=iv,
            ct=ciphertext,
        )

        plaintext = b""
        with ESAPI(self._tcti) as ectx, primary(ectx) as prim:
            hmac_key = ectx.load(prim, hmac_private, hmac_public)
            try:
                actual_mac = self._compute_hmac(ectx, hmac_key, ctx)
            finally:
                ectx.flush_context(hmac_key)

            if not hmac.compare_digest(actual_mac, expected_mac):
                raise VaultIntegrityError("vault entry failed integrity check, possible tampering or corruption!")

            if original_len <= 0 or original_len % 4 != 0 or original_len > len(ciphertext):
                raise VaultFormatError("vault entry has an invalid length")

            aes = ectx.load(prim, private, public)
            try:
                for i in range(0, len(ciphertext), CHUNK_SIZE):
                    out, iv_out = self._encrypt_decrypt(ectx, aes, True, iv, ciphertext[i : i + CHUNK_SIZE])
                    plaintext += bytes(out)
                    iv = bytes(iv_out)
            finally:
                ectx.flush_context(aes)

        return plaintext[:original_len]
