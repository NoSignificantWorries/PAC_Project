import argparse

from app.detector import detectors


class ArgParser:
    """
    command line argument parsing
    """
    def __init__(self):
        parser_obj = argparse.ArgumentParser()

        parser_obj.add_argument("-i", "--input", help="Path to your video")
        parser_obj.add_argument("-o", "--output", help="Path to output log")
        parser_obj.add_argument("-m", "--models", help="Models to predict")

        self.args = parser_obj.parse_args()
        
    def parse_args(self) -> dict:
        video_path = self.args.input
        output_path = self.args.output
        models = self.args.models

        if video_path is None:
            raise "Missing input path."
        if output_path is None:
            raise "Missing output path."
        
        models_res = []
        if models is None:
            models_res = detectors.keys()
        else:
            for model_name in models.split(","):
                if model_name not in detectors.keys():
                    print("WARN: Unknown model name:", model_name)
                else:
                    models_res.append(model_name)

        return {"input": video_path, "output": output_path, "models": models_res}

    ...
