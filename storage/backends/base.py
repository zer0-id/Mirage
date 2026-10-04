from abc import ABC, abstractmethod
from errors import VaultFormatError
import base64

Entry = dict[str, str | int]


def b64e(data: bytes) -> str:
    """Base64-encode bytes for storage in a JSON entry."""
    return base64.b64encode(data).decode()


def get_bytes(entry: Entry, key: str) -> bytes:
    """Read a base64 field from an entry, or raise VaultFormatError."""
    value = entry.get(key)
    if not isinstance(value, str):
        raise VaultFormatError("malformed vault entry")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError:
        raise VaultFormatError("malformed vault entry") from None


def get_len(entry: Entry) -> int:
    """Read the plaintext length from an entry, or raise VaultFormatError."""
    value = entry.get("len")
    if not isinstance(value, int) or isinstance(value, bool):
        raise VaultFormatError("malformed vault entry")
    return value


class Backend(ABC):
    """Encrypts and decrypts one embedding with a specific TPM scheme.

    Rules every backend follows:

    * Never touch the filesystem. FaceVault owns ``vault.json`` and its locking.
    * Accept and return JSON-safe dicts only (base64 strings and ints).
    * Put the mode id, username and plaintext length in the authenticated data.
    * Raise ``VaultIntegrityError`` when authentication fails and
      ``VaultFormatError`` when an entry is malformed.
    * Never put secrets or field values in error messages.
    """

    mode: str

    def __init__(self, tcti: str | None = None) -> None:
        """Create a backend.

        Args:
            tcti: TCTI string such as ``"swtpm:path=..."``, or ``None`` to use
                the system default.
        """
        self._tcti = tcti

    def supported(self) -> bool:
        probe = bytes(16)
        try:
            return self.decrypt("__probe__", self.encrypt("__probe__", probe)) == probe
        except Exception:
            return False

    @abstractmethod
    def encrypt(self, username: str, data: bytes) -> Entry:
        """Encrypt data for a user.

        Args:
            username: The user the entry belongs to. Bound into the authenticated data.
            data: The plaintext bytes (a float32 embedding).

        Returns:
            A new vault entry in this mode's format.

        Raises:
            TSS2_Exception: If a TPM command fails.
        """

    @abstractmethod
    def decrypt(self, username: str, entry: Entry) -> bytes:
        """Verify an entry and return its plaintext.

        Args:
            username: The user the entry is expected to belong to.
            entry: A vault entry written by this mode's ``encrypt``.

        Returns:
            The plaintext bytes.

        Raises:
            VaultFormatError: If the entry has the wrong mode or an invalid length.
            VaultIntegrityError: If authentication fails (tampering or corruption).
            TSS2_Exception: If a TPM command fails.
        """

    def _require_mode(self, entry: Entry) -> None:
        """Refuse entries written by a different mode (second line of defense).

        Raises:
            VaultFormatError: If the entry's mode is not this backend's mode.
        """
        if entry.get("mode") != self.mode:
            raise VaultFormatError("vault entry was written by a different mode")
