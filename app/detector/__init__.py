from app.detector.TrafficLightViolation import TrafficLightViolation

# пример иерархии моделей
detectors = {
    "TLV": {
        "id": 1,
        "detector": TrafficLightViolation,
        "depend": []
    },
}

__all__ = ["detectors"]
