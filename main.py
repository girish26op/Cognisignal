"""
CogniSignal Main Pipeline CLI Entry Point.
Runs end-to-end processing on traffic camera images:
1. Feature extraction & region proposal
2. Object detection via selected model (Custom CNN, Faster R-CNN, or YOLOv8)
3. PCU traffic density calculation
4. Dynamic green signal allocation & safety diagnostics
"""

import os
import sys
import argparse
import cv2
import io

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.vehicle_detector import VehicleDetectorFactory
from src.signal_optimizer import TrafficSignalOptimizer
from src.feature_extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="CogniSignal Real-Time Traffic Signal Optimization")
    parser.add_argument("--image", type=str, default="data/sample_images/Img00001.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="Auto (Best Selected)",
                        choices=["Auto (Best Selected)", "Custom CNN (Scratch)", "Faster R-CNN", "YOLOv8"],
                        help="Active vehicle detection model")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--lanes", type=int, default=3, help="Number of directional road lanes")
    parser.add_argument("--output", type=str, default="outputs/annotated_output.jpg", help="Output annotated image path")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("🚦 CogniSignal: Adaptive Real-Time Traffic Signal Optimizer")
    print("=" * 75)

    if not os.path.exists(args.image):
        print(f"[!] Error: Image not found at {args.image}")
        sys.exit(1)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[!] Error: Failed to decode image at {args.image}")
        sys.exit(1)

    print(f"[*] Input Image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    print(f"[*] Selected Model: {args.model}")
    print(f"[*] Detection Confidence Threshold: {args.conf}")
    print(f"[*] Directional Road Lanes: {args.lanes}")

    # Stage 1: Feature Extraction
    extractor = FeatureExtractor()
    feat_res = extractor.process_pipeline(frame)
    print(f"[+] OpenCV Feature Proposals: {feat_res['proposal_count']} regions extracted.")

    # Stage 2: Vehicle Detection
    factory = VehicleDetectorFactory(selected_engine=args.model)
    det_res = factory.detect(frame, conf_threshold=args.conf)
    counts = det_res["counts"]
    print(f"[+] Detection Completed via {det_res['model_name']} in {det_res['latency_ms']:.1f}ms ({det_res['fps']:.1f} FPS)")
    print(f"    Vehicles Counted -> Cars: {counts.get('car', 0)}, Buses: {counts.get('bus', 0)}, "
          f"Trucks: {counts.get('truck', 0)}, Bikes: {counts.get('bike', 0)}, Autos: {counts.get('auto', 0)}")

    # Stage 3: PCU Optimization
    optimizer = TrafficSignalOptimizer(default_lanes=args.lanes)
    sig_res = optimizer.optimize_signal(counts, lane_count=args.lanes)

    print("\n" + "-" * 75)
    print("📊 REAL-TIME TRAFFIC & SIGNAL METRICS")
    print("-" * 75)
    print(f" • Total Vehicles Counted:       {sig_res['total_vehicles']}")
    print(f" • Total Passenger Car Units:    {sig_res['pcu']:.2f} PCU")
    print(f" • Traffic Load Per Lane:        {sig_res['pcu_per_lane']:.2f} PCU/lane")
    print(f" • Congestion Assessment:        {sig_res['congestion_level']}")
    print(f" • Recommended Green Light Time: {sig_res['recommended_green_seconds']} Seconds")
    if sig_res["warning"]:
        print(f" ⚠️ Safety Warning: {sig_res['warning']}")

    # Stage 4: Annotation & Output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    annotated = VehicleDetectorFactory.annotate_frame(frame, det_res, title_prefix=sig_res["congestion_level"])
    cv2.imwrite(args.output, annotated)
    print(f"\n[+] Saved annotated frame to: {args.output}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
