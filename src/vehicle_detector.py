"""
Unified Vehicle Detector Factory Engine.
Routes detection requests dynamically to:
- Custom CNN (Scratch dual-head PyTorch model)
- Faster R-CNN (MobileNetV3 FPN)
- YOLOv8 (Ultralytics Real-Time anchor-free detector)
Supports manual selection or auto-selection from models/selected_best_model.json.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional

from src.custom_cnn import CustomCNNDetector
from src.faster_rcnn_detector import FasterRCNNDetector
from src.yolo_detector import YOLODetector


BEST_MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "selected_best_model.json")

LABEL_COLORS = {
    "car": (255, 140, 0),        # Vibrant Deep Sky Blue / Orange
    "bus": (0, 165, 255),        # Orange
    "truck": (0, 0, 255),        # Red
    "bike": (50, 205, 50),       # Lime Green
    "auto": (255, 215, 0)        # Gold
}


class VehicleDetectorFactory:
    """
    Factory class that creates and coordinates multi-model vehicle detectors.
    """

    SUPPORTED_ENGINES = [
        "Auto (Best Selected)",
        "Custom CNN (Scratch)",
        "Faster R-CNN",
        "YOLOv8"
    ]

    def __init__(self, selected_engine: str = "Auto (Best Selected)", weights_dir: Optional[str] = None):
        self.selected_engine = selected_engine
        self.weights_dir = weights_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        # Instantiate detectors
        self._custom_cnn = None
        self._faster_rcnn = None
        self._yolo = None
        self.active_engine_name = ""

        self._resolve_engine()

    def _resolve_engine(self):
        """Resolves which model engine to activate."""
        engine = self.selected_engine

        if engine == "Auto (Best Selected)":
            if os.path.exists(BEST_MODEL_CONFIG_PATH):
                try:
                    with open(BEST_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        best_name = cfg.get("model_name", "YOLOv8")
                        if "YOLO" in best_name:
                            engine = "YOLOv8"
                        elif "Faster" in best_name:
                            engine = "Faster R-CNN"
                        else:
                            engine = "Custom CNN (Scratch)"
                except Exception:
                    engine = "YOLOv8"
            else:
                engine = "YOLOv8"

        self.active_engine_name = engine

    def get_detector(self, engine_name: Optional[str] = None):
        """Returns the requested or currently active detector instance."""
        target = engine_name or self.active_engine_name

        if "Custom" in target:
            if self._custom_cnn is None:
                weights = os.path.join(self.weights_dir, "custom_cnn_weights.pth")
                self._custom_cnn = CustomCNNDetector(weights_path=weights)
            return self._custom_cnn
        elif "Faster" in target:
            if self._faster_rcnn is None:
                self._faster_rcnn = FasterRCNNDetector()
            return self._faster_rcnn
        else:
            if self._yolo is None:
                self._yolo = YOLODetector()
            return self._yolo

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.50, engine_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs detection with the configured engine.
        """
        self._resolve_engine()
        detector = self.get_detector(engine_override)
        res = detector.detect(image_bgr, conf_threshold=conf_threshold)
        res["active_engine"] = self.active_engine_name
        return res

    @staticmethod
    def annotate_frame(image_bgr: np.ndarray, detection_result: Dict[str, Any], title_prefix: str = "") -> np.ndarray:
        """
        Annotates bounding boxes, class labels, and HUD summary bar on the frame.
        """
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        boxes = detection_result.get("boxes", [])
        labels = detection_result.get("labels", [])
        scores = detection_result.get("scores", [])
        counts = detection_result.get("counts", {})
        model_name = detection_result.get("model_name", "CogniSignal")

        # Draw HUD Header Bar
        hud_h = 36
        cv2.rectangle(annotated, (0, 0), (w, hud_h), (20, 20, 20), -1)
        hud_text = (f"[{title_prefix or model_name}] Total: {counts.get('total', 0)} | "
                    f"Cars: {counts.get('car', 0)} | Buses: {counts.get('bus', 0)} | "
                    f"Trucks: {counts.get('truck', 0)} | Bikes: {counts.get('bike', 0)} | "
                    f"FPS: {detection_result.get('fps', 0)}")
        cv2.putText(annotated, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Bounding Boxes
        for box, lbl, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            color = LABEL_COLORS.get(lbl, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

            # Label banner
            tag = f"{lbl.upper()}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (xmin, max(0, ymin - 18)), (xmin + tw + 6, ymin), color, -1)
            cv2.putText(annotated, tag, (xmin + 3, max(12, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated
