from app.detector.TrafficLightViolation import TrafficLightViolation
from app.detector.TrafficLightDetection import TrafficLightDetection

# пример иерархии моделей
detectors = {
    "TLD": {
        "id": 1,
        "detector": TrafficLightDetection,
        "depend": []
    },
    "TLV": {
        "id": 3,
        "detector": TrafficLightViolation,
        "depend": [1]
    },
}

__all__ = ["detectors"]
