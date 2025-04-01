from app.detector import detectors


class Pipeline:
    """
    creates a pipeline for executing models based on their dependencies
    """
    def __init__(self, models):
        self.pipeline = []
        self.current = 0

        self.parse(models)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        """
        return the next model in the queue
        """
        if self.current == self.__len__():
            raise StopIteration

        model = self.pipeline[self.current]
        self.current += 1

        return model

    def __len__(self):
        return len(self.pipeline)

    def parse(self, models):
        """
        make ordered list of models by they dependencies
        """
        added = set()

        def add_model(model_name):
            if model_name in added:
                return
            if model_name not in detectors:
                raise ValueError(f"Model '{model_name}' not found in detectors.")

            model_info = detectors[model_name]
            dependencies = model_info.get("depend", [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]

            for dep in dependencies:
                add_model(dep)  # Рекурсивное добавление зависимостей

            self.pipeline.append(model_info["detector"])
            added.add(model_name)

        for model in models:
            add_model(model)

    def apply_masks(self, frame):
        """
        :param frame: image to apply masks on
        :return: masked frame
        """
        for model in self.pipeline:
            model.apply_mask(frame)

        return frame
