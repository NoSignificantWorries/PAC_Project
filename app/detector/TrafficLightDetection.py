from app.detector.BaseDetector import Detector
from ultralytics import YOLO

class TrafficLightDetection(Detector):
    def __init__(self, confidence=0.7):
        self.path_to_model = "app/detector/weights/yolov8m_traffic_light.pt"
        self.model = YOLO(self.path_to_model)
        self.confidence = confidence
        self.isgreen = False
        self.isred = False
        self.isyellow = False
        self.isinvisible = False



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

        
        boxes = result.boxes
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
        
        return pred_color
        