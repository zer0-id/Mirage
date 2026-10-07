import cv2
from typing import cast
import numpy as np
from numpy.typing import NDArray


class FaceDetector:
    def __init__(self, model_path, score_threshold: float, nms_threshold=0.3, top_k=1) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (640, 480), score_threshold, nms_threshold, top_k)

    def detect(self, frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return None
        return faces[0]

    def crop_face(self, frame: NDArray[np.uint8], face: NDArray[np.float32], size: tuple[int, int] = (112, 112)) -> NDArray[np.uint8] | None:
        fh, fw = frame.shape[:2]
        x, y, w, h = map(int, face[:4])
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, fw), min(y + h, fh)
        if x1 <= x0 or y1 <= y0:
            return None
        return cast(NDArray[np.uint8], cv2.resize(frame[y0:y1, x0:x1], size))
