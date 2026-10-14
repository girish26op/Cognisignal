"""
OpenCV Computer Vision Feature Extraction Pipeline.
Demonstrates 6-stage classical image processing for vehicle candidate extraction:
1. Raw BGR Input Frame
2. Grayscale Conversion
3. Gaussian Blur
4. Canny Edge Detection
5. Morphological Dilation
6. Contour Extraction & Bounding Box Proposals
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


class FeatureExtractor:
    """
    Step-by-step computer vision feature extraction engine using OpenCV.
    """

    def __init__(self, min_contour_area: int = 600, blur_ksize: Tuple[int, int] = (5, 5),
                 canny_thresh1: int = 50, canny_thresh2: int = 150):
        self.min_contour_area = min_contour_area
        self.blur_ksize = blur_ksize
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def process_pipeline(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full 6-stage feature extraction pipeline on an input image.

        Args:
            frame_bgr: Input BGR image (numpy ndarray).

        Returns:
            Dictionary containing images for all 6 stages, extracted bounding boxes,
            and descriptive statistics.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input image frame is empty or invalid.")

        # Stage 1: Raw Input BGR Frame
        stage1_raw = frame_bgr.copy()

        # Stage 2: Grayscale Conversion
        stage2_gray = cv2.cvtColor(stage1_raw, cv2.COLOR_BGR2GRAY)

        # Stage 3: Gaussian Blur (Noise Reduction)
        stage3_blur = cv2.GaussianBlur(stage2_gray, self.blur_ksize, 0)

        # Stage 4: Canny Edge Detection
        stage4_edges = cv2.Canny(stage3_blur, self.canny_thresh1, self.canny_thresh2)

        # Stage 5: Morphological Dilation (Bridge Edges into Connected Blobs)
        stage5_dilated = cv2.dilate(stage4_edges, self.dilation_kernel, iterations=2)

        # Stage 6: Contour Extraction & Bounding Box Proposals
        contours, _ = cv2.findContours(stage5_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        proposals = []
        stage6_annotated = stage1_raw.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                proposals.append({
                    "box": [x, y, x + w, y + h],
                    "rect": [x, y, w, h],
                    "area": float(area)
                })
                # Draw proposal bounding box in vibrant amber/cyan
                cv2.rectangle(stage6_annotated, (x, y), (x + w, y + h), (0, 215, 255), 2)
                cv2.putText(stage6_annotated, f"Prop {int(area)}px", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        return {
            "stages": {
                "1_raw": stage1_raw,
                "2_gray": stage2_gray,
                "3_blur": stage3_blur,
                "4_canny": stage4_edges,
                "5_dilated": stage5_dilated,
                "6_contours": stage6_annotated
            },
            "proposals": proposals,
            "proposal_count": len(proposals),
            "total_contours": len(contours)
        }
