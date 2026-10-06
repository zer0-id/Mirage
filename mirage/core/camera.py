import cv2


class CameraHandler:
    def __init__(self, device_index=2, fallback_index=0, width=640, height=480):
        self.device_index = device_index
        self.fallback_index = fallback_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.fallback_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open any camera (tried {self.device_index}, {self.fallback_index})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not capture the frame from camera")
        if len(frame.shape) == 2:
            frame_3channel = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame_3channel
        return frame

    def release(self):
        if self.cap is None:
            return
        self.cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
