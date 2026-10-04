import struct

FORMAT_VER = 1
TAG = f"vault-mac-v{FORMAT_VER}".encode()


def _field(data: bytes) -> bytes:
    """Prefix data with its 4-byte big-endian length."""
    return struct.pack(">I", len(data)) + data


def build_context(mode: str, username: str, original_len: int, **fields: bytes) -> bytes:
    """Build the bytes a backend authenticates (a MAC input or AEAD associated data).

    Args:
        mode: The backend's mode id.
        username: The user the entry belongs to.
        original_len: The plaintext byte count.
        **fields: Named binary fields to bind in. Sorted by name, so call order is irrelevant.
    """
    out = TAG + _field(mode.encode("utf-8")) + _field(username.encode("utf-8")) + struct.pack(">Q", original_len) + struct.pack(">H", len(fields))
    for name in sorted(fields):
        out += _field(name.encode("utf-8")) + _field(fields[name])
    return out
