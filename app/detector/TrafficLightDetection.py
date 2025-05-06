from app.detector.BaseDetector import Detector
from ultralytics import YOLO
import cv2
import numpy as np

class TrafficLightDetection(Detector):
    def __init__(self, confidence=0.7):
        self.path_to_model = "app/detector/weights/yolov8m_traffic_light.pt"
        self.model = YOLO(self.path_to_model)
        self.confidence = confidence
        self.isgreen = False
        self.isred = False
        self.isyellow = False
        self.isinvisible = False
        self.boxes = None


    def predict(self, frame) -> str:
        """
        Checks if seeable traffic light is green, red, yellow or invisible.

        :param frame: Frame (BGR, np.ndarray)
        :return: green/red/yellow/None
        :self.prediction = 0 ~ green
        :self.prediction = 1 ~ red
        :self.prediction = 2 ~ yellow
        """


        result = self.model(frame)
        
        pred_color = None

        boxes = result[0].boxes
        
        if not bool(boxes.conf.tolist()):
            self.boxes = None
            return None
        
        print(boxes)
        
        """
        сделать проверку на найденные классы

        вывод для примера:
            cls: tensor([0.], device='cuda:0')
            conf: tensor([0.3172], device='cuda:0')
            data: tensor([[6.8941e+02, 2.6723e+01, 7.1862e+02, 7.5892e+01, 3.1718e-01, 0.0000e+00]], device='cuda:0')
            id: None
            is_track: False
            orig_shape: (1080, 1920)
            shape: torch.Size([1, 6])
            xywh: tensor([[704.0143,  51.3075,  29.2186,  49.1697]], device='cuda:0')
            xywhn: tensor([[0.3667, 0.0475, 0.0152, 0.0455]], device='cuda:0')
            xyxy: tensor([[689.4050,  26.7226, 718.6236,  75.8924]], device='cuda:0')
            xyxyn: tensor([[0.3591, 0.0247, 0.3743, 0.0703]], device='cuda:0')

        ошибка:
              File "/home/dmitry/Projects/PAC_Project/app/main.py", line 32, in main
                frame_buffer[0][mid] = model.predict(frame_buffer[0][0])
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              File "/home/dmitry/Projects/PAC_Project/app/detector/TrafficLightDetection.py", line 48, in predict
                if boxes.conf[2] > self.confidence:
                   ~~~~~~~~~~^^^
            IndexError: index 2 is out of bounds for dimension 0 with size 1
        
        Запуск:
            python3 -m app.main -i ~/Videos/Bikes.mp4 -o ~/Videos/res.mp4

        """

        if boxes.conf[0] > self.confidence:
            self.isgreen = True
            pred_color = 'green'
        else:
            self.isgreen = False

        if boxes.conf[2] > self.confidence:
            self.isred = True
            pred_color = 'red'
        else:
            self.isred = 0

        if boxes.conf[3] > self.confidence:
            self.isyellow = True
            pred_color = 'yellow'
        else:
            self.isyellow = 0
        
        self.boxes = boxes
        
        return pred_color
    

def apply_mask(self, frame):
    """
    Draws rectangles on image
    if there are no boxes do nothing
    """

    image = frame
    
    if self.boxes is None:
        return
    
    for rects in np.array(self.boxes.xyxy):
        left_top = (round(rects[0]), round(rects[1]))
        right_low = (round(rects[2]), round(rects[3]))
        cv2.rectangle(image, left_top, right_low, (255, 0, 0), 5)
        