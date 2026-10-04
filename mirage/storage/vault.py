import json
import stat
import fcntl
from contextlib import contextmanager
import tempfile
from pathlib import Path
import os
import numpy as np
from errors import VaultFormatError
from storage.backends import ALL_MODES, get_backend, MODE_SEAL
from storage.context import FORMAT_VER


DEFAULT_VAULT_DIR = Path("/var/lib/mirage")
TPM_DEVICE = Path("/dev/tpmrm0")
EMBEDDING_DIM = 512
EMBEDDING_BYTES = EMBEDDING_DIM * 4


class FaceVault:
    def __init__(self, vault_dir: Path = DEFAULT_VAULT_DIR, tcti: str | None = None, enroll_mode: str = MODE_SEAL) -> None:
        if enroll_mode not in ALL_MODES:
            raise ValueError(f"unknown enroll mode: {enroll_mode}")
        self.vault_dir = vault_dir
        self.vault_json = vault_dir / "vault.json"
        self._tcti = tcti
        self._enroll_mode = enroll_mode
        self._ensure_vault_dir()
        self.vault = self._load_vault()

    def _ensure_vault_dir(self) -> None:
        try:
            lst = os.lstat(self.vault_dir)
            if stat.S_ISLNK(lst.st_mode):
                raise RuntimeError(f"{self.vault_dir} is a symlink, refusing to follow...")
        except FileNotFoundError:
            pass

        self.vault_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.vault_dir, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchmod(fd, 0o700)
            st = os.fstat(fd)
            if st.st_uid != os.geteuid():
                raise RuntimeError(f"{self.vault_dir} owned by uid {st.st_uid}, expected {os.geteuid()}")
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self):
        fd = os.open(self.vault_dir / "vault.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _sweep_temp_files(self) -> None:
        for p in self.vault_dir.glob("vault.*.tmp"):
            p.unlink(missing_ok=True)

    def _load_vault(self) -> dict:
        if self.vault_json.exists():
            with open(self.vault_json, "r") as f:
                data = json.load(f)
                format_version = data.get("version")
                if format_version == FORMAT_VER:
                    return data
                else:
                    raise VaultFormatError(f"vault format v{format_version}, this version expects {FORMAT_VER}, re-enroll or migrate")
        return {"version": FORMAT_VER, "users": {}}

    def _save_vault(self, vault: dict) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=self.vault_dir, prefix="vault.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(vault, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self.vault_json)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        dir_fd = os.open(self.vault_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @classmethod
    def check_prerequisites(cls, vault_dir: Path = DEFAULT_VAULT_DIR) -> dict[str, bool]:
        vault_json = vault_dir / "vault.json"
        checks = {
            "tpm_accessible": os.access(TPM_DEVICE, os.R_OK | os.W_OK),
            "vault_readable": os.access(vault_json, os.R_OK) if vault_json.exists() else True,
            "vault_dir_writable": os.access(vault_dir if vault_dir.exists() else vault_dir.parent, os.W_OK),
        }
        checks["ready"] = all(checks.values())
        return checks

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        raw = embedding.astype(np.float32).tobytes()
        if len(raw) != EMBEDDING_BYTES:
            raise ValueError(f"embedding must have {EMBEDDING_DIM} float32 values")

        with self._locked():
            self._sweep_temp_files()
            current = self._load_vault()
            if name in current.get("users", {}):
                raise ValueError(f"user '{name}' is already enrolled")

            backend = get_backend(self._enroll_mode, self._tcti)
            entry = backend.encrypt(name, raw)

            current.setdefault("users", {})[name] = {"embeddings": [entry]}
            self._save_vault(current)
            self.vault = current

    def get_embeddings(self, username: str) -> list[np.ndarray]:
        user = self.vault.get("users", {}).get(username)
        if user is None:
            raise KeyError(f"no such user: {username}")

        embeddings = []
        for entry in user["embeddings"]:
            if not isinstance(entry, dict):
                raise VaultFormatError("malformed vault entry")
            backend = get_backend(entry.get("mode"), self._tcti)
            plaintext = backend.decrypt(username, entry)
            if len(plaintext) != EMBEDDING_BYTES:
                raise VaultFormatError("vault entry has an unexpected embedding size")
            embeddings.append(np.frombuffer(plaintext, dtype=np.float32))
        return embeddings

    def remove_user(self, username: str) -> None:
        with self._locked():
            self._sweep_temp_files()
            current = self._load_vault()
            if username not in current.get("users", {}):
                raise KeyError(f"no such user: {username}")
            del current["users"][username]
            self._save_vault(current)
            self.vault = current

    def list_users(self) -> list[str]:
        return list(self.vault.get("users", {}).keys())
