from ultralytics import YOLO
from ultralytics.models import yolo     # module, not class
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from typing import override, Any

from .head import DetectDistribution
from .predictor import DetectionDistributionPredictor


class DistYOLO(YOLO):
    def __init__(self, model = "yolo11n.pt", task = None, verbose = False):
        super().__init__(model, task, verbose)

        # Swap out the model head
        detect: Detect = self.model.model[-1]
        self.model.model[-1] = DetectDistribution.from_detect(detect)

    @override
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        t_map = super().task_map
        t_map["detect"] = {
            "model": DetectionModel,
            "trainer": yolo.detect.DetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": DetectionDistributionPredictor,
        }

        return t_map