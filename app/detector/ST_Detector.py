import numpy as np
from keras.models import load_model # Don`t forget to install this shit
from keras.preprocessing.image import load_img, img_to_array

import os # This is for test, can be removed
from cv2 import imread, resize, INTER_LINEAR

from BaseDetector import Detector

"""
ST Detector

Download weights here: https://drive.google.com/file/d/1l7TiIWSRCsTeDQolKyVFVTS6r8L2BRG8/view?usp=sharing

Note:
Weights are trained for 2 classes: 0 - Cars
                                   1 - Special Transport
"""

PATH_TO_WEIGHTS = './trained.keras'

class ST_Detector(Detector):
    def __init__(self, device):
        self.device = device

        self.__name__ = "ST_Detector"

        self.prediction = None

        self._model = load_model(PATH_TO_WEIGHTS)
    
    def __predict_image__(self, img):
        input_img = img_to_array(img)
        input_img = np.expand_dims(input_img, axis=0)
        predict_img = self._model.predict(input_img)
        return predict_img
    
    def predict(self, frame):
        self.prediction = self.__predict_image__(frame)
        return self.prediction


# Usage example for class (Also can be deleted)
if __name__ == "__main__":
    # Loading checkpoint
    det = ST_Detector("./trained.keras")
    # Loading image
    path = './dataset/SynthCrops/test/'
    class_index = 0
    TP  = [0,0]
    All = [0,0]
    
    print(os.listdir(path))
    for class_folder in os.listdir(path):
        for filename in os.listdir(os.path.join(path, class_folder)):
            file_path = os.path.join(path, class_folder, filename)
            img = imread(os.path.join(path, class_folder, filename))
            if img is not None:
                img = resize(img, (244,244), interpolation = INTER_LINEAR)
                res = det.__predict_image__(img)[0]
                print(res)
                if np.argmax(res) == class_index:
                    TP[np.argmax(res)] += 1
                All[class_index] += 1
        class_index += 1

    print(TP)
    print(All)
    print(f'Class 0: TP={TP[0]}, Acc={TP[0] / All[0] * 100}%')
    print(f'Class 1: TP={TP[1]}, Acc={TP[1] / All[1] * 100}%')
    print(f'Total Acc: {sum(TP) / sum(All)}%')
    
