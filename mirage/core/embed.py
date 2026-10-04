import numpy as np
import onnxruntime as ort


class FaceEmbedder:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img):
        img = face_img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def embed(self, face_img):
        img = self.preprocess(face_img)
        output = self.session.run(None, {self.input_name: img})
        embedding = output[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
