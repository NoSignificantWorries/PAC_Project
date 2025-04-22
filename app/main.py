from parser import ArgParser
from video import Video
import numpy as np
from pipeline import Pipeline

from lgbt import lgbt


def main(kwargs: dict):
    video = Video(kwargs["input"], kwargs["output"])

    pipeline = Pipeline(kwargs["models"])

    for cluster in lgbt(video, desc="Processing frames"):
        for mid, dep, model in pipeline:
            model.predict(cluster[0], *cluster[np.array(dep)])

        mask = pipeline.apply_masks(cluster[0])
        video.write(mask)


if __name__ == "__main__":
    parser = ArgParser()

    kwargs = parser.parse_args()

    main(kwargs)
