from typing import override
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class DetectionDistributionPredictor(DetectionPredictor):
    @override
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        return super().postprocess(preds, img, orig_imgs, **kwargs)
    
    @override
    def construct_result(self, pred, img, orig_img, img_path):
        return super().construct_result(pred, img, orig_img, img_path)