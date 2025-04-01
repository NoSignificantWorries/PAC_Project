from app.detector import detectors

"""
базовый класс детоктора будет лежать в файле BaseDetector.py в одной папке с вашим детектором
ваш детектор должен наследоваться от этого класса
"""

class Detector:
    def __init__(self, device):
        self.device = device

        self.__name__ = ""

        self.prediction = None
        ...

    def predict(self, frame):
        """
        update self.prediction
        return self.prediction
        """
        detectors["exampleDetector"] # получить актуальный предикт модели из списка зависимостей.
        # Гарантируется что зависимая модель уже обработала данный frame
        ...

    def apply_mask(self, frame):
        ...

    def __del__(self):
        ...
