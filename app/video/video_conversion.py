import cv2
import numpy as np


class Video:
    """
    video iteration and frame-by-frame recording
    """
    def __init__(self, path_in, path_out):
        """
        initialize input and output video objects
        """
        self.path_in = path_in
        self.path_out = path_out
        self.cap = cv2.VideoCapture(path_in)
        self.buffer_size = 250
        self.frame_buffer = np.array([None] * self.buffer_size, dtype=object)
        self.buffer_i = 0
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {path_in}")

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out = cv2.VideoWriter(path_out, self.fourcc, self.fps, (self.width, self.height))

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.close()
            raise StopIteration

        self.frame_buffer[self.buffer_i] = frame
        self.buffer_i = (self.buffer_i + 1) % self.buffer_size
        return frame

    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, index):
        if index < 0 or index >= self.buffer_size:
            raise IndexError(f"Index out of range: {index}")
        return self.frame_buffer[index]
    
    def clear_buffer(self):
        self.frame_buffer[:] = None
        self.buffer_i = 0

    def write(self, frame):
        self.out.write(frame)

    def close(self):
        self.cap.release()
        self.out.release()
