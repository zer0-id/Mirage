import os
import stat
import tempfile
import tomllib
from typing import get_args
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePosixPath
from mirage.errors import ConfigError
from mirage.storage.backends import ALL_MODES

CONFIG_PATH = Path("/etc/mirage/config.toml")
RECOMMENDED_MAX_DISTANCE = 0.60
RECOMMENDED_MIN_FRAMES_REQUIRED = 2
RECOMMENDED_MIN_ENROLL_FRAMES = 3
RECOMMENDED_MAX_FAILURES = 10
RECOMMENDED_MIN_LOCKOUT_S = 30
MAX_SANE_TIMEOUT_S = 30


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


def _clean_abs(p: str) -> bool:
    path = PurePosixPath(p)
    return path.is_absolute() and ".." not in path.parts and str(path) == p


class _Section:
    def validate(self) -> None:
        """Override to enforce ranges. Raise ConfigError on a bad value."""

    def warnings(self) -> list[str]:
        """Return messages for legal values that weaken security."""
        return []


@dataclass(frozen=True)
class StorageConfig(_Section):
    vault_dir: str = "/var/lib/mirage"
    enroll_mode: str | None = None

    def validate(self) -> None:
        _require(_clean_abs(self.vault_dir), "storage.vault_dir must be a clean absolute path")
        _require(self.enroll_mode is None or self.enroll_mode in ALL_MODES, "storage.enroll_mode is not a known mode")


@dataclass(frozen=True)
class MatchConfig(_Section):
    max_distance: float = 0.40
    frames_required: int = 3
    timeout_s: float = 4.0

    def validate(self) -> None:
        _require(0 < self.max_distance <= 2, "match.max_distance must be in (0, 2]")
        _require(self.frames_required >= 1, "match.frames_required must be at least 1")
        _require(0 < self.timeout_s and math.isfinite(self.timeout_s), "match.timeout_s must be positive and finite")

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.max_distance > RECOMMENDED_MAX_DISTANCE:
            out.append(f"match.max_distance is above {RECOMMENDED_MAX_DISTANCE}: look-alikes and photos are more likely to pass")
        if self.frames_required < RECOMMENDED_MIN_FRAMES_REQUIRED:
            out.append("match.frames_required is 1: a single lucky frame can unlock")
        if self.timeout_s > MAX_SANE_TIMEOUT_S:
            out.append(f"match.timeout_s is above {MAX_SANE_TIMEOUT_S}: a login can hang for a long time")
        return out


@dataclass(frozen=True)
class EnrollConfig(_Section):
    frames: int = 5

    def validate(self) -> None:
        _require(self.frames >= 1, "enroll.frames must be at least 1")

    def warnings(self) -> list[str]:
        if self.frames < RECOMMENDED_MIN_ENROLL_FRAMES:
            return [f"enroll.frames is below {RECOMMENDED_MIN_ENROLL_FRAMES}: the stored template will be less stable"]
        return []


@dataclass(frozen=True)
class LivenessConfig(_Section):
    require: bool = True
    require_eyes_open: bool = True

    def warnings(self) -> list[str]:
        if not self.require:
            return ["liveness.require is off: a printed photo or a screen can unlock"]
        return []


@dataclass(frozen=True)
class CameraConfig(_Section):
    ir: bool = True
    ir_device: str = "/dev/video2"
    rgb_device: str = "/dev/video0"

    @property
    def device(self) -> str:
        """The device the camera module should open."""
        return self.ir_device if self.ir else self.rgb_device

    def validate(self) -> None:
        for name in ("ir_device", "rgb_device"):
            value = getattr(self, name)
            _require(_clean_abs(value) and value.startswith("/dev/"), f"camera.{name} must be a clean path under /dev/")


@dataclass(frozen=True)
class LimitsConfig(_Section):
    max_failures: int = 5
    lockout_s: int = 300

    def validate(self) -> None:
        _require(self.max_failures >= 1, "limits.max_failures must be at least 1")
        _require(self.lockout_s >= 0, "limits.lockout_s cannot be negative")

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.max_failures > RECOMMENDED_MAX_FAILURES:
            out.append(f"limits.max_failures is above {RECOMMENDED_MAX_FAILURES}: brute-forcing gets easier")
        if self.lockout_s < RECOMMENDED_MIN_LOCKOUT_S:
            out.append(f"limits.lockout_s is below {RECOMMENDED_MIN_LOCKOUT_S}: lockouts barely slow an attacker")
        return out


@dataclass(frozen=True)
class Config:
    storage: StorageConfig = field(default_factory=StorageConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    enroll: EnrollConfig = field(default_factory=EnrollConfig)
    liveness: LivenessConfig = field(default_factory=LivenessConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)

    def validate(self) -> None:
        for f in fields(self):
            getattr(self, f.name).validate()

    def warnings(self) -> list[str]:
        out: list[str] = []
        for f in fields(self):
            out.extend(getattr(self, f.name).warnings())
        if not self.camera.ir and not self.liveness.require:
            out.append("camera.ir is false and liveness.require is off: a photo or a screen can unlock")
        return out


def _coerce(where: str, value: object, expected: object) -> object:
    options = get_args(expected) or (expected,)
    for opt in options:
        if opt is bool and isinstance(value, bool):
            return value
        if opt is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if opt is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if opt is str and isinstance(value, str):
            return value
    raise ConfigError(f"{where} has the wrong type")


def _build_section(cls, raw: object, name: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    known = {f.name: f.type for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    return cls(**{key: _coerce(f"{name}.{key}", value, known[key]) for key, value in raw.items()})


def _from_dict(data: dict) -> Config:
    sections = {f.name: f.type for f in fields(Config)}
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    cfg = Config(**{name: _build_section(sections[name], data[name], name) for name in data})
    cfg.validate()
    return cfg


def _check_trusted(st: os.stat_result, what: str, trusted_uid: int) -> None:
    if st.st_uid != trusted_uid:
        raise ConfigError(f"{what} is not owned by the trusted user")
    if st.st_mode & 0o022:
        raise ConfigError(f"{what} is writable by group or others")


def _read_trusted(path: Path, trusted_uid: int) -> bytes:
    """Open path with all the trust checks and return its bytes. A missing file is an error."""
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        raise ConfigError("config directory not found") from None
    except OSError:
        raise ConfigError("cannot open the config directory") from None

    try:
        _check_trusted(os.fstat(dir_fd), "config directory", trusted_uid)
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
        except FileNotFoundError:
            raise ConfigError("config file not found") from None
        except OSError:
            raise ConfigError("cannot open the config file") from None

        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            _require(stat.S_ISREG(st.st_mode), "config is not a regular file")
            _check_trusted(st, "config file", trusted_uid)
            return f.read()
    finally:
        os.close(dir_fd)


def load_config(path: Path = CONFIG_PATH, trusted_uid: int = 0) -> Config:
    """Load the config, failing closed. A missing file means the built-in defaults."""
    blob = _read_trusted(path, trusted_uid)
    try:
        data = tomllib.loads(blob.decode("utf-8"))
    except UnicodeDecodeError, tomllib.TOMLDecodeError:
        raise ConfigError("config is not valid TOML") from None
    return _from_dict(data)


def set_enroll_mode(mode: str, path: Path = CONFIG_PATH, trusted_uid: int = 0) -> None:
    """Set storage.enroll_mode in place."""
    import tomlkit
    from tomlkit.exceptions import TOMLKitError

    _require(mode in ALL_MODES, "unknown enroll mode")
    blob = _read_trusted(path, trusted_uid)
    try:
        doc = tomlkit.parse(blob.decode("utf-8"))
    except UnicodeDecodeError, TOMLKitError:
        raise ConfigError("config is not valid TOML") from None

    if "storage" not in doc:
        doc["storage"] = tomlkit.table()
    doc["storage"]["enroll_mode"] = mode
    text = tomlkit.dumps(doc)

    _from_dict(tomllib.loads(text))

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix="config.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    dir_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
