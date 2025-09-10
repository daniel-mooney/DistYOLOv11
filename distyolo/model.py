from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from .head import DetectDistribution


def load_model(model_cfg: str) -> YOLO:
    """Loads a DistYOLO model

    Args:
        model_cfg (str): yaml or pt file specifying the model architecture.

    Returns:
        YOLO: The configured model
    """
    model = YOLO(model_cfg)

    # Swap out the model head
    detect: Detect = model.model.model[-1]
    model.model.model[-1] = DetectDistribution.from_detect(detect)

    return model