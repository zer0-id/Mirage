import cv2


class FaceDetector:
    def __init__(self, model_path, score_threshold=0.8, nms_threshold=0.3, top_k=1):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (640, 480), score_threshold, nms_threshold, top_k)

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return None
        return faces[0]

    def crop_face(self, frame, face):
        x, y, w, h = map(int, face[:4])
        crop = frame[y : y + h, x : x + w]
        return cv2.resize(crop, (112, 112))
