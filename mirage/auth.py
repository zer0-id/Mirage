import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from mirage.config import Config
from mirage.core.camera import CameraHandler
from mirage.core.detect import FaceDetector
from mirage.core.embed import FaceEmbedder
from mirage.core.liveness import LivenessDetector
from mirage.core.match import is_match
from mirage.storage.vault import FaceVault
from collections import deque

log = logging.getLogger(__name__)

# Some IR emitters pulse on alternate frames. Remember the last few brightness
# levels and skip a frame that is far dimmer than the recent peak: it is an
# emitter-off frame.
_BRIGHTNESS_WINDOW = 4
_EMITTER_OFF_RATIO = 0.5  # skip a frame below this fraction of the recent peak

def _best_match(stored: Sequence[NDArray[np.float32]], live: NDArray[np.float32], threshold: float) -> tuple[bool, float]:
    """True if the live embedding matches any stored one, also the smallest distance."""
    matched = False
    best = math.inf
    for ref in stored:
        ok, dist = is_match(ref, live, threshold)
        matched = matched or ok
        best = min(best, dist)
    return matched, best


def authenticate(username: str, cfg: Config) -> bool:
    """Return True only after cfg.match.frames_required consecutive matching frames.
    Raises KeyError for an unknown user, and lets camera, vault and model errors
    propagate. Returns False on timeout.
    """
    for warning in cfg.warnings():
        log.warning(warning)

    vault = FaceVault(vault_dir=Path(cfg.storage.vault_dir))
    stored = vault.get_embeddings(username)
    if not stored:
        return False

    detector = FaceDetector(str(_DEV_MODELS_DIR / _DETECTOR_MODEL), score_threshold=cfg.camera.min_face_score)
    embedder = FaceEmbedder(str(_DEV_MODELS_DIR / _EMBEDDER_MODEL))
    liveness = LivenessDetector(str(_DEV_MODELS_DIR / _LIVENESS_MODEL)) if cfg.liveness.require else None

    streak = 0
    recent: deque[float] = deque(maxlen=_BRIGHTNESS_WINDOW)
    with CameraHandler(cfg.camera.device) as cam:
        deadline = time.monotonic() + cfg.match.timeout_s
        while time.monotonic() < deadline:
            frame = cam.read_frame()
            level = float(frame.mean())
            recent.append(level)
            if level < _EMITTER_OFF_RATIO * max(recent):
                continue

            face = detector.detect(frame)
            if face is None:
                log.debug("no face detected")
                streak = 0
                continue

            crop = detector.crop_face(frame, face)
            if crop is None:
                log.debug("degenerate crop")
                streak = 0
                continue

            if liveness is not None:
                is_live, score = liveness.is_live(crop, cfg.liveness.threshold)
                log.debug("liveness score=%.3f live=%s", score, is_live)
                if not is_live:
                    streak = 0
                    continue

            matched, dist = _best_match(stored, embedder.embed(crop), cfg.match.max_distance)
            streak = streak + 1 if matched else 0
            log.debug("distance=%.3f matched=%s streak=%d/%d", dist, matched, streak, cfg.match.frames_required)

            if streak >= cfg.match.frames_required:
                return True

    return False
