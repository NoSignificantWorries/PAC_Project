from app.detector.BaseDetector import Detector
import cv2
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms

class PedestrianRoadDetector(Detector):
    def __init__(self):
        super().__init__(device="cuda" if torch.cuda.is_available() else "cpu")

        # Модели
        self.seg_model = self.load_segmentation_model()
        self.yolo_model = self.load_yolo_model()

        # Постпроцессинг
        self.seg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask = None

    def load_segmentation_model(self):
        model = deeplabv3_resnet50(num_classes=3)
        model_path = "../weights/deeplabv3_road_seg.pth"
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        return model

    def load_yolo_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', path='../weights/yolov5s_human.pt', force_reload=False).eval()

    def predict(self, frame):
        """
        Детекция пешеходов, проверка стоят ли они на проезжей части (но не на переходе)
        """
        rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Сегментация
        with torch.no_grad():
            input_tensor = self.seg_transform(rgb).unsqueeze(0).to(self.device)
            seg_output = self.seg_model(input_tensor)['out']
        seg_mask = torch.argmax(seg_output.squeeze(), dim=0).cpu().numpy()
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        road_mask = (seg_mask == 1)
        crosswalk_mask = (seg_mask == 2)

        results = self.yolo_model(rgb)
        detections = results.xyxy[0].cpu().numpy()

        person_cls = 0
        pedestrians = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            if int(cls) != person_cls or conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            box_height = y2 - y1
            strip_height = max(1, box_height // 6)
            strip_top = y2 - strip_height

            bottom_strip = np.zeros((h, w), dtype=np.uint8)
            bottom_strip[strip_top:y2, x1:x2] = 1

            on_road = np.any(bottom_strip & road_mask)
            on_crosswalk = np.any(bottom_strip & crosswalk_mask)

            label = "SAFE" if not (on_road and not on_crosswalk) else "WARNING"

            pedestrians.append({
                "bbox": [x1, y1, x2, y2],
                "status": label
            })

        self.mask = pedestrians
        return pedestrians

    def apply_mask(self, frame):
        if self.mask is None:
            raise RuntimeError("Call predict() before apply_mask()")

        for ped in self.mask:
            x1, y1, x2, y2 = ped["bbox"]
            status = ped["status"]
            color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
