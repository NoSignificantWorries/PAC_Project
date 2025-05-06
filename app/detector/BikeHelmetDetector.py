from app.detector.BaseDetector import Detector
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class BikeHelmetDetector(Detector):
    def __init__(self): 
        super().__init__(device="cuda")

        # Модель для детекции велосипедистов
        self.bike_detector = YOLO("weights/bike_detector.pt")

        # Модель классификатора касок
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = YOLO("weights/helmet_classifier.pt")
        self.classifier.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # размер под классификатор
            transforms.ToTensor(),
        ])

    def predict(self, frame):
        """
        Метод для обработки одного кадра:
        - Детектируем велосипедистов
        - Вырезаем каждый bbox
        - Классифицируем каждый кроп: в каске или нет
        """
        results = self.bike_detector(frame)  # Получаем результат детекции

        bikes = []

        # Проверяем, что есть хотя бы один результат
        if len(results) > 0:
            # Пробегаемся по всем найденным объектам (bicyclists)
            for box in results[0].boxes:  # 'results[0]' — это детекции для первого изображения
                conf = box.conf[0].item()  # Доверие (confidence)

                if conf < 0.5:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Получаем координаты бокса

                # Вырезаем область велосипедиста
                cropped_bike = frame[y1:y2, x1:x2]

                # Проверяем размер кропа
                if cropped_bike.size == 0:
                    continue

                # Подготовка кропа для классификатора
                crop_resized = cv2.resize(cropped_bike, (224, 224))
                crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                crop_tensor = transforms.ToTensor()(crop_resized).unsqueeze(0).to(self.device)

                # Классификация
                cls_results = self.classifier(crop_tensor)
                helmet_label = cls_results[0].probs.top1  # Предполагаем, что top1: 0 - с каской, 1 - без каски

                # Используем инвертированную логику:
                helmet_status = "with_helmet" if helmet_label == 0 else "without_helmet"

                bikes.append({
                    "bbox": [x1, y1, x2, y2],
                    "helmet_status": helmet_status
                })

        return bikes



    def apply_mask(self, frame, bikes):
        """
        Метод для визуализации:
        - Рисует рамки вокруг велосипедистов
        - Разные цвета для "с каской" и "без каски"
        """

        for bike in bikes:
            x1, y1, x2, y2 = bike["bbox"]
            helmet_status = bike["helmet_status"]

            color = (0, 255, 0) if helmet_status == "with_helmet" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, helmet_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame
