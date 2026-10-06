import onnxruntime as ort
import numpy as np
import cv2


class LivenessDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img):
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img -= [104.0, 117.0, 123.0]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def is_live(self, face_img, threshold=0.6):
        img = self.preprocess(face_img)
        output = self.session.run(None, {self.input_name: img})
        x = output[0][0]
        e_x = np.exp(x - np.max(x))
        softmax = e_x / e_x.sum()
        liveness_score = softmax[0]
        return liveness_score > threshold, liveness_score
