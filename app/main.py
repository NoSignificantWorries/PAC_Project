from app.parser import ArgParser
from app.video import Video
import numpy as np
from app.pipeline import Pipeline

from lgbt import lgbt


def main(kwargs: dict):
    video = Video(kwargs["input"], kwargs["output"])

    pipeline = Pipeline(kwargs["models"])

    buffer_size = 250
    cluster_size = 5
    frame_buffer = np.array([[None] * cluster_size] * buffer_size, dtype=object)

    for frame in lgbt(video, desc="Processing frames"):
        frame_buffer[1:] = frame_buffer[:-1]
        frame_buffer[0][0] = frame

        for mid, dep, model in pipeline:
            print(model)
            # model.predict(frame_buffer[0][0], *tuple(frame_buffer[0][np.array(dep, dtype=np.int32)]))
            model.predict(frame_buffer[0][0], None, None)

        mask = pipeline.apply_masks(frame_buffer[0][0])
        video.write(mask)


if __name__ == "__main__":
    parser = ArgParser()

    kwargs = parser.parse_args()

    main(kwargs)
