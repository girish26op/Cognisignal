"""
Traffic Signal Optimization Engine based on Passenger Car Units (PCU)
and Traffic Density Analysis.
"""

from typing import Dict, Any, Optional


class TrafficSignalOptimizer:
    """
    Traffic Signal Optimization Engine based on Passenger Car Units (PCU)
    and Traffic Density Analysis.
    """

    PCU_WEIGHTS = {
        "car": 1.0,
        "bus": 2.5,
        "truck": 3.0,
        "bike": 0.5,
        "auto": 1.2
    }

    def __init__(self, min_green: int = 15, max_green: int = 90, default_lanes: int = 3):
        self.min_green = min_green
        self.max_green = max_green
        self.default_lanes = default_lanes

    def calculate_pcu(self, vehicle_counts: Dict[str, int]) -> float:
        """
        Calculates total Passenger Car Units (PCU) from vehicle count dict.
        """
        total_pcu = 0.0
        for vtype, weight in self.PCU_WEIGHTS.items():
            total_pcu += vehicle_counts.get(vtype, 0) * weight
        return round(total_pcu, 2)

    def optimize_signal(self, vehicle_counts: Dict[str, int], lane_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes traffic congestion level and recommended green signal duration.
        
        Args:
            vehicle_counts: Dictionary mapping vehicle types to counts.
            lane_count: Number of lanes for the direction (defaults to default_lanes).
            
        Returns:
            Dictionary containing metrics, congestion level, and recommended green signal duration.
        """
        lanes = lane_count if lane_count and lane_count > 0 else self.default_lanes
        pcu = self.calculate_pcu(vehicle_counts)
        pcu_per_lane = round(pcu / lanes, 2)

        # Signal Timing & Congestion Logic
        if pcu_per_lane < 3.0:
            congestion = "Low"
            recommended_green = max(self.min_green, int(15 + pcu_per_lane * 2))
        elif pcu_per_lane < 8.0:
            congestion = "Medium"
            recommended_green = int(25 + (pcu_per_lane - 3.0) * 3)
        elif pcu_per_lane < 15.0:
            congestion = "High"
            recommended_green = int(40 + (pcu_per_lane - 8.0) * 3.5)
        else:
            congestion = "Heavy"
            recommended_green = min(self.max_green, int(65 + (pcu_per_lane - 15.0) * 2.5))

        # Clamp between bounds
        recommended_green = max(self.min_green, min(self.max_green, recommended_green))

        total_vehicles = vehicle_counts.get("total", sum(vehicle_counts.get(k, 0) for k in ["car", "bus", "truck", "bike", "auto"]))
        heavy_count = vehicle_counts.get("bus", 0) + vehicle_counts.get("truck", 0)
        heavy_ratio = round(heavy_count / max(1, total_vehicles), 2)
        
        warning = None
        if heavy_ratio > 0.4 and total_vehicles > 5:
            warning = "High proportion of heavy vehicles detected. Extended clearance phase recommended."

        return {
            "total_vehicles": total_vehicles,
            "pcu": pcu,
            "pcu_per_lane": pcu_per_lane,
            "congestion_level": congestion,
            "recommended_green_seconds": recommended_green,
            "heavy_vehicle_ratio": heavy_ratio,
            "warning": warning
        }
