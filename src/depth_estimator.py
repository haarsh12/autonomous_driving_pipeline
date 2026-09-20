"""
depth_estimator.py
Step 2: Monocular depth estimation using MiDaS.

MiDaS outputs a relative inverse-depth map: higher pixel value = closer
to the camera, lower value = farther away.
"""

import numpy as np
import cv2
import torch
import torch.hub

torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True


class DepthEstimator:
    """
    Wrapper around MiDaS depth estimation models (MiDaS_small, DPT_Hybrid, DPT_Large).
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[INFO] CUDA requested for MiDaS depth estimator but PyTorch CUDA is not available. Running on CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.model_type = model_type

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        if self.device.type == "cuda":
            self.model = self.model.half()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a 2D float32 array (H x W) where higher values = closer to camera.
        """
        H, W = frame_bgr.shape[:2]
        # Downsample large frames for 4x depth speedup with identical relative depth quality
        if W > 448:
            scale = 384.0 / W
            small_w, small_h = 384, max(128, int(H * scale))
            small_bgr = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_bgr = frame_bgr

        img_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        if self.device.type == "cuda":
            input_batch = input_batch.half()

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.float().cpu().numpy()
        return depth_map

    @staticmethod
    def normalize_for_display(depth_map: np.ndarray) -> np.ndarray:
        """Normalize a raw depth map to 0-255 uint8 for visualization."""
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_map, dtype=np.uint8)
        norm = (depth_map - d_min) / (d_max - d_min)
        return (norm * 255).astype(np.uint8)
