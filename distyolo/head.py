from __future__ import annotations
from typing import override

import torch
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors


class DetectDistribution(Detect):
    @override
    def _inference(self, x):
        """
        Decode predicted bounding boxes, class probabilities and bounding box distribution
        based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes, class probabilities
            and bounding box distribution.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box_dist = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box_dist, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box_dist.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box_dist) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box_dist), self.anchors.unsqueeze(0)) * self.strides
        if self.export and self.format == "imx":
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        
        # dbox is mean of box distribution
        return torch.cat((dbox, cls.sigmoid(), box_dist), 1)

    @classmethod
    def from_detect(cls, detect: Detect) -> DetectDistribution:
        """ Create a DetectDistribution head from a Detect head 

        Args:
            d (Detect): The detect head

        Returns:
            DetectDistribution: New DetectDistribution head
        """
        new = cls.__new__(cls)
        new.__dict__ = detect.__dict__.copy()

        new.load_state_dict(detect.state_dict(), strict=False)
        return new