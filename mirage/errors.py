class MirageError(Exception):
    """Base class for every error Mirage raises on purpose."""


class VaultError(MirageError, ValueError):
    """Base class for vault storage errors. Subclasses ValueError for old handlers."""


class VaultIntegrityError(VaultError):
    """A vault entry failed its MAC check: tampering or corruption."""


class VaultFormatError(VaultError):
    """A vault entry or file has the wrong format version or an invalid length."""

class ConfigError(MirageError):
    """The configuration file is untrusted or invalid."""
