from app.detector.TrafficLightViolation import TrafficLightViolation
from app.detector.BikeHelmetDetector import BikeHelmetDetector

# пример иерархии моделей
detectors = {
    "TLV": {
        "id": 1,
        "detector": TrafficLightViolation,
        "depend": []
    },
    "BHD": {
        "id": 2,
        "detector": BikeHelmetDetector,
        "depend": []
    }
}

__all__ = ["detectors"]
