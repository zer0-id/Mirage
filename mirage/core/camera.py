import os
import stat
import cv2
import numpy as np
from typing import Self, cast
from numpy.typing import NDArray


class CameraHandler:
    def __init__(self, device: str, width: int = 640, height: int = 480) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.cap = None

    @classmethod
    def check_prerequisites(cls, device: str) -> dict[str, bool]:
        """Read-only probe: never opens the camera."""
        result = {
            "device_in_dev": False,
            "device_exists": False,
            "device_is_chardev": False,
            "device_accessible": False,
        }
        if isinstance(device, str):
            real = os.path.realpath(device)
            result["device_in_dev"] = real.startswith("/dev/")
            if result["device_in_dev"]:
                try:
                    mode = os.stat(real).st_mode
                    result["device_exists"] = True
                    result["device_is_chardev"] = stat.S_ISCHR(mode)
                    result["device_accessible"] = os.access(real, os.R_OK | os.W_OK)
                except OSError:
                    pass
        result["ready"] = all(result.values())
        return result

    def open(self) -> None:
        prereq = self.check_prerequisites(self.device)
        if not prereq["ready"]:
            failed = [k for k, v in prereq.items() if not v and k != "ready"]
            raise RuntimeError(f"Camera {self.device} not usable: {', '.join(failed)}")
        cap = cv2.VideoCapture(os.path.realpath(self.device), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera {self.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap

    def read_frame(self) -> NDArray[np.uint8]:
        if self.cap is None:
            raise RuntimeError("Camera is not open")
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0 or frame.ndim not in (2, 3):
            raise RuntimeError("Could not capture a valid frame from camera")
        img = cast(NDArray[np.uint8], frame)
        if img.ndim == 2:
            return cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        return img

    def release(self) -> None:
        if self.cap is None:
            return
        self.cap.release()
        self.cap = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()
