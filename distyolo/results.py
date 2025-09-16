from typing import override, Tuple

from ultralytics.engine.results import Results, Boxes, BaseTensor
import numpy as np
import torch


class ResultsDistribution(Results):
    """Extension of ultralytics Results class which fascilitates bounding box distributions.

    Note: only works for a Detection network.
    """
    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (np.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor): A 2D tensor of bounding box coordinates for each detection.
                Should be in the xyxy format. Note that Ultralytics performs the conversion from xywh to xyxy
                in the non-max suppression function.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            speed (dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).
        """
        super().__init__(
            orig_img,
            path,
            names,
            probs=probs,
            keypoints=keypoints,
            speed=speed
        )
        self.boxes = BoxDistribution(boxes, self.orig_shape)  # Overwrite boxes with BoxDistribution
    
    @override
    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        im_gpu: torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Plot detection results on an input RGB image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Specify the RGB text color for classification task.

        Returns:
            (np.ndarray): Annotated image as a numpy array.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            >>>     im = result.plot()
            >>>     im.show()
        """
        ...
    
class BoxDistribution(Boxes):
    @override
    def __init__(
        self,
        bbox_dist: torch.Tensor | np.ndarray,
        orig_shape: tuple[int, int],
        max_reg: int = 16
    ) -> None:
        """
        Initialise the BoxDistribution class with detection box data.

        Args:
            bbox_dist (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6 + 4 * reg_max) or (num_boxes, 7+ 4 * reg_max). Columns should contain
                [x1, y1, x2, y2, (optional) track_id, confidence, class, stride, l_dist, t_dist, r_dist, b_dist].
                Each of the bbox side distributions have a length equal to the number of DFL bins i.e. reg_max
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
            max_reg (int): DFL channels

        Attributes:
            data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
            orig_shape (tuple[int, int]): The original image size, used for normalization.
            is_track (bool): Indicates whether tracking IDs are included in the box data.
        """
        # Note: Boxes constructor is written too rigidly to extend
        if bbox_dist.ndim == 1:
            bbox_dist = bbox_dist[None, :]
        
        BaseTensor.__init__(self, bbox_dist, orig_shape)      # Grandparent constructor

        n_box_vars = bbox_dist.shape[-1]
        self.is_track = n_box_vars % 2 == 0    # Even when track 
        self.orig_shape = orig_shape
        self.max_reg = max_reg
    
    @override
    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """
        Return the confidence scores for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -4*self.max_reg - 3]

    @override
    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """
        Return the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -4*self.max_reg - 2]
    
    @override
    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """
        Return the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -4 * self.max_reg - 4]

    @property
    def distribution(self) -> torch.Tensor | np.ndarray:
        """
        Return the bounding box distribution parameters for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 2D tensor or array containing the bounding box distribution parameters.
                The shape is (N, 4 * reg_max), where N is the number of boxes and reg_max is the number of DFL bins.
                The distribution parameters are ordered as [l_dist, t_dist, r_dist, b_dist] for each box, where each
                side distribution has a length equal to reg_max.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> bbox_dists = boxes.distribution
            >>> print(bbox_dists.shape)  # e.g., torch.Size([3, 64]) for reg_max=16
            >>> top_dist = bbox_dists[0, 1*16: 2*16]  # Top side distribution for the first box
        """
        return self.data[:, -4 * self.max_reg:]
    
    @property
    def stride(self) -> torch.Tensor | np.ndarray:
        """
        Return the stride values for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing the stride values for each detection box.
                The shape is (N,), where N is the number of boxes.
        """
        return self.data[:, -4 * self.max_reg - 1]