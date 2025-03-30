import cv2
import numpy as np

def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer_size = 250
    num_columns = 5
    numpy_array = np.empty((buffer_size, num_columns), dtype=object)

    frame_number = 0
    while(True):
        ret, frame = cap.read()

        if not ret:
            break

        numpy_array[frame_number % buffer_size, 0] = frame

        for i in range(1, num_columns):
            numpy_array[frame_number % buffer_size, i] = None

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "your_video.mp4"
    num_columns = 5
    video_to_numpy(video_file, num_columns)
