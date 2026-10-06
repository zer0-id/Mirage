"""Vault entry formats, one per storage mode.

Every entry is a flat dict of JSON-safe values. Binary fields are base64 strings.

| Field                            | tpm-cfb                  | tpm-seal                       |
|----------------------------------|--------------------------|--------------------------------|
| mode                             | "tpm-cfb"                | "tpm-seal"                     |
| len                              | plaintext byte count     | plaintext byte count           |
| private, public                  | AES key blobs            | not used                       |
| sealed_private, sealed_public    | not used                 | sealed-key blobs               |
| iv                               | CFB initial IV           | not used                       |
| nonce                            | not used                 | AEAD nonce                     |
| ct                               | ciphertext               | AEAD ciphertext (tag appended) |
| hmac_private, hmac_public, mac   | HMAC key blobs and MAC   | not used                       |

Rules:
- The backend is chosen from the entry's "mode", never from the config.
- The mode id is part of the authenticated data in every backend.
- A missing or unknown mode raises VaultFormatError. There is no default.
"""

MODE_CFB = "tpm-cfb"
MODE_SEAL = "tpm-seal"

from mirage.errors import VaultFormatError

from .base import Backend, Entry

__all__ = ["ALL_MODES", "MODE_CFB", "MODE_SEAL", "Backend", "Entry", "get_backend"]

ALL_MODES = (MODE_CFB, MODE_SEAL)


def get_backend(mode: object, tcti: str | None = None) -> Backend:
    """Return the backend for a mode id.

    The modules are imported lazily, so ``tpm2_pytss`` and ``cryptography``
    load only for a mode that needs them, and no import cycle forms.

    Args:
        mode: The mode id from a vault entry. Typed ``object`` because it
            comes from JSON and could be anything.
        tcti: TCTI string passed to the backend, or ``None`` for the default.

    Returns:
        A new backend instance for that mode.

    Raises:
        VaultFormatError: If the mode is unknown or missing.
    """
    if mode == MODE_CFB:
        from .cfb import CFBBackend

        return CFBBackend(tcti)
    if mode == MODE_SEAL:
        from .seal import SealBackend

        return SealBackend(tcti)
    raise VaultFormatError("unknown or missing vault mode")
