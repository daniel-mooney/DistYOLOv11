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

        # Grab device/dtype from the existing model
        base_param = next(self.model.parameters())
        device = base_param.device
        dtype = base_param.dtype

        # Swap out the model head
        detect: Detect = self.model.model[-1]
        new_head = DetectDistribution.from_detect(detect)
        new_head.to(device=device, dtype=dtype)
        self.model.model[-1] = new_head
        self.model.eval()

    @override
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        t_map = dict(super().task_map)
        t_map["detect"] = {
            "model": DetectionModel,
            "trainer": yolo.detect.DetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": DetectionDistributionPredictor,
        }

        return t_map