from app.detector.BaseDetector import Detector
from ultralytics import YOLO
import cv2
import numpy as np

class CarDetector:
    def __init__(self, model_path, conf_threshold=0.3):

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold   
    def detect(self, frame):
        """
        Detects cars in the image.

        :param frame: Frame (BGR, np.ndarray)
        :return: List of bounding boxes: [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame)[0]
        boxes = []

        for box in results.boxes:
            conf = box.conf.item()
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))

        return boxes
    

class TrafficLightViolation(Detector):
    def __init__(self):
        self.car_detector = CarDetector(model_path="app/detector/weights/yolov8n_car.pt", conf_threshold=0.3)
        pass

    def predict(self, frame, light_color, stop_line_mask):
        """
        Checks if the car violated the rules by running a red traffic light.

        :param frame: Frame (BGR, np.ndarray), light_color, stop_line_mask
        :return: True if violation, False if not
        """

        if light_color == "green":
            return False

        car_boxes = self.car_detector.detect(frame)

        if not car_boxes:
            return False
        
        stop_line = cv2.imread(stop_line_mask, cv2.IMREAD_GRAYSCALE)
        _, stop_line_bin = cv2.threshold(stop_line, 1, 255, cv2.THRESH_BINARY)
        
        for (x1, y1, x2, y2) in car_boxes:
            
            car_roi = frame[y1:y2, x1:x2]
            car_mask = cv2.inRange(car_roi, (0, 0, 0), (255, 255, 255)) 

            
            intersection = cv2.bitwise_and(car_mask, stop_line_bin[y1:y2, x1:x2])  
            if np.sum(intersection) > 0: 
                return True
        
        return False

            



