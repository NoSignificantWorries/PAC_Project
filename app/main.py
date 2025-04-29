import time

from app.parser import ArgParser
from app.video import Video
import numpy as np
from app.pipeline import Pipeline
from lgbt import lgbt
import cv2


def main(kwargs: dict):
    video = Video(kwargs["input"], kwargs["output"])
    
    stop_line = cv2.imread("app/detector/data/masks/stop_line_1.1.png", cv2.IMREAD_GRAYSCALE)

    pipeline = Pipeline(kwargs["models"])
    
    buffer_size = 250
    cluster_size = 5
    frame_buffer = np.array([[None] * cluster_size] * buffer_size, dtype=object)

    tmp = 0
    for frame in lgbt(video, desc="Processing frames"):
        frame_buffer[1:] = frame_buffer[:-1]
        frame_buffer[0][0] = frame

        for mid, dep, model in pipeline:
            # frame_buffer[0][mid] = model.predict(frame_buffer[0][0], "green", np.zeros_like(frame_buffer[0][0]))
            # frame_buffer[0][mid] = model.predict(frame_buffer[0][0], *tuple(frame_buffer[0][np.array(dep, dtype=np.int32)]))
            frame_buffer[0][mid] = model.predict(frame_buffer[0][0], "green", stop_line)
            print(frame_buffer[0][mid])

        mask = pipeline.apply_masks(frame_buffer[0][0])
        video.write(mask)
        
        tmp += 1

        if tmp > 60:        
            break
        time.sleep(0.5)


if __name__ == "__main__":
    parser = ArgParser()

    kwargs = parser.parse_args()

    main(kwargs)
