"""
Sample Assets and Realistic Traffic Frames Generator.
Creates:
1. Realistic traffic road test images with annotated vehicles.
2. Initial CSV dataset for tabular ML models.
3. Pre-trained weights for Custom CNN and Random Forest models.
4. Publication-ready metric plots for the dashboard.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import torch
import pandas as pd

from src.custom_cnn import CustomVehicleCNN
from src.tabular_model import TabularTrafficModel
from src.visualization import Visualizer
from src.model_evaluator import ModelEvaluator


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(ROOT_DIR, "data", "sample_images")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "outputs", "plots")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def draw_realistic_road(scenario: str = "medium") -> np.ndarray:
    """
    Renders a realistic urban road scene with multiple lanes and vehicles.
    """
    w, h = 960, 540
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Road background (dark asphalt)
    img[100:540, :] = (55, 55, 60)

    # Sky and roadside buildings
    img[0:100, :] = (200, 180, 160)
    for bx in range(0, w, 120):
        bh = np.random.randint(60, 100)
        color = (130 + (bx % 40), 120, 120)
        cv2.rectangle(img, (bx, 100 - bh), (bx + 100, 100), color, -1)

    # Pavement / Sidewalk
    cv2.rectangle(img, (0, 100), (w, 130), (120, 120, 120), -1)
    cv2.rectangle(img, (0, 510), (w, 540), (120, 120, 120), -1)

    # Lane markings (dashed white lines)
    for ly in [220, 310, 400]:
        for lx in range(0, w, 60):
            cv2.line(img, (lx, ly), (lx + 35, ly), (240, 240, 240), 2)

    # Vehicle positions according to scenario
    vehicles = []
    if scenario == "low":
        vehicles = [
            ("car", 150, 230, 90, 55, (220, 40, 30)),
            ("car", 450, 330, 95, 58, (30, 180, 220)),
            ("bike", 700, 160, 40, 30, (40, 200, 50))
        ]
    elif scenario == "medium":
        vehicles = [
            ("car", 120, 230, 90, 55, (220, 40, 30)),
            ("car", 320, 150, 85, 50, (230, 230, 240)),
            ("bus", 260, 320, 170, 75, (20, 140, 240)),
            ("car", 580, 240, 95, 58, (30, 180, 220)),
            ("bike", 500, 160, 40, 30, (40, 200, 50)),
            ("auto", 720, 330, 65, 52, (0, 215, 255)),
            ("car", 800, 420, 100, 60, (50, 50, 50))
        ]
    else:  # heavy / rush hour
        vehicles = [
            ("truck", 80, 230, 180, 80, (40, 80, 160)),
            ("bus", 300, 320, 175, 75, (20, 140, 240)),
            ("car", 120, 150, 85, 50, (220, 40, 30)),
            ("car", 520, 240, 95, 55, (230, 230, 240)),
            ("car", 680, 150, 90, 52, (30, 180, 220)),
            ("auto", 480, 330, 65, 52, (0, 215, 255)),
            ("bike", 620, 340, 40, 30, (40, 200, 50)),
            ("bike", 280, 160, 40, 30, (50, 210, 60)),
            ("car", 740, 420, 105, 60, (50, 50, 50)),
            ("truck", 400, 410, 190, 82, (100, 110, 120))
        ]

    # Draw vehicles with windshields and wheels
    for vtype, vx, vy, vw, vh, col in vehicles:
        # Main body
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), col, -1)
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (20, 20, 20), 2)
        # Windshield
        cv2.rectangle(img, (vx + int(vw * 0.2), vy + 6), (vx + int(vw * 0.75), vy + vh - 6), (190, 220, 240), -1)
        # Headlights
        cv2.circle(img, (vx + vw - 4, vy + 8), 4, (0, 255, 255), -1)
        cv2.circle(img, (vx + vw - 4, vy + vh - 8), 4, (0, 255, 255), -1)

    return img


def generate_all():
    print("[*] Generating realistic traffic sample images...")
    img1 = draw_realistic_road("medium")
    img2 = draw_realistic_road("low")
    img3 = draw_realistic_road("heavy")

    cv2.imwrite(os.path.join(SAMPLE_DIR, "Img00001.jpg"), img1)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "Img00002.jpg"), img2)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "Img00003.jpg"), img3)
    print(f"[+] Saved 3 sample road images to {SAMPLE_DIR}")

    print("[*] Training Tabular Random Forest models...")
    tab_model = TabularTrafficModel()
    metrics = tab_model.train_models()
    print(f"[+] Tabular models trained: Acc={metrics['classification']['accuracy']*100:.1f}%, MAE={metrics['regression']['mae']}s")

    print("[*] Saving Custom CNN initial PyTorch weights...")
    cnn = CustomVehicleCNN(num_classes=6)
    weights_path = os.path.join(MODELS_DIR, "custom_cnn_weights.pth")
    torch.save(cnn.state_dict(), weights_path)
    print(f"[+] Saved Custom CNN weights to {weights_path}")

    print("[*] Generating benchmark evaluation and winning config...")
    evaluator = ModelEvaluator()
    bench = evaluator.run_benchmark()
    print(f"[+] Selected best model: {bench['winner']['name']}")

    print("[*] Rendering publication analytics plots...")
    Visualizer.plot_confusion_matrix()
    Visualizer.plot_roc_curve()
    Visualizer.plot_loss_curve()
    Visualizer.plot_model_comparison()
    Visualizer.plot_traffic_analytics()
    print(f"[+] All 5 plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all()
