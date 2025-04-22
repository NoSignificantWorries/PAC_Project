from exampleDetector import exampleDetector
from TrafficLightViolation import TrafficLightViolation
from exampleDetector2 import exampleDetector2
from exampleDetector3 import exampleDetector3
from exampleDetector4 import exampleDetector4

# пример иерархии моделей
detectors = {
    "TLV": {
        "id": 1,
        "detector": TrafficLightViolation,
        "depend": []
    },
}

__all__ = ["detectors"]
