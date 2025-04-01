from exampleDetector import exampleDetector
from exampleDetector2 import exampleDetector2
from exampleDetector3 import exampleDetector3
from exampleDetector4 import exampleDetector4

# пример иерархии моделей
detectors = {
    "exampleDetector": {
        "detector": exampleDetector,
        "depend": [exampleDetector2, exampleDetector3]
    },
    "exampleDetector2": {
        "detector": exampleDetector2,
        "depend": exampleDetector4
    },
    "exampleDetector3": {
        "detector": exampleDetector3
    },
    "exampleDetector4": {
        "detector": exampleDetector4
    },
}

__all__ = ["detectors"]
