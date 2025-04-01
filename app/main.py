from parser import ArgParser
from video import Video
from pipeline import Pipeline

from lgbt import lgbt


def main(kwargs: dict):
    video = Video(kwargs["input"], kwargs["output"])

    pipeline = Pipeline(kwargs["models"])

    for frame in lgbt(video, desc="Processing frames"):
        for model in pipeline:
            model.predict(frame)

        mask =  pipeline.apply_masks(frame)
        video.write(mask)


if __name__ == "__main__":
    parser = ArgParser()

    kwargs = parser.parse_args()

    main(kwargs)
