import onnxruntime as ort
import numpy as np
from numpy.typing import NDArray
from typing import cast
import cv2


class LivenessDetector:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        if not (isinstance(h, int) and isinstance(w, int)):
            raise ValueError("liveness model must have a fixed input size")
        self.input_size = (w, h)

    def preprocess(self, face_img: NDArray[np.uint8]) -> NDArray[np.float32]:
        img = cv2.resize(face_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img -= [104.0, 117.0, 123.0]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def is_live(self, face_img: NDArray[np.uint8], threshold: float) -> tuple[bool, float]:
        img = self.preprocess(face_img)
        output = cast(list[NDArray[np.float32]], self.session.run(None, {self.input_name: img}))
        x = output[0][0]
        e_x = np.exp(x - np.max(x))
        softmax = e_x / e_x.sum()
        liveness_score = softmax[0]
        return liveness_score > threshold, liveness_score
