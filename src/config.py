from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ScenarioConfig:
    static_spawn_region: np.ndarray
    dynamic_spawn_region: np.ndarray
    obj_scale_range: np.ndarray
    min_drop_height: float
    inner_radius: int
    outer_radius: int
    theta_min: float
    theta_max: float
    min_num_dynamic_objects: int
    max_num_dynamic_objects: int
    plane_asset_dir: str
    min_tower_size: int
    max_tower_size: int
    num_trials: int = 1
    max_num_towers: int = 6


@dataclass
class DominoConfig:
    static_spawn_region: np.ndarray
    min_control_points: int = 3
    max_control_points: int = 6
    base_spacing: float = 0.15  # Base spacing between dominoes
    curvature_density_factor: float = 2.0  # How much to increase density in high curvature areas
    min_spacing: float = 0.08  # Minimum spacing between dominoes
    max_spacing: float = 0.25  # Maximum spacing between dominoes
    control_point_pattern: str = "random"  # Options: "random", "spiral", "s_curve", "circle", "figure_8", "zigzag", "heart", "star"
    min_domino_height: float = 0.6
    max_domino_height: float = 2.0
    

class Configuration:
  @staticmethod
  def get_config_for_scenario(scenario):
    config_methods = {
      "fall": Configuration._fall_config,
      "domino": Configuration._domino_config,  # Added mapping for domino scenario
    }
    return config_methods.get(scenario, lambda: "Invalid scenario")()
  
  @staticmethod
  def _fall_config():
    return ScenarioConfig(
        static_spawn_region=np.array([(-1, -1.8, 0), (2.5, 2.5, 1)],dtype=np.float64),
        dynamic_spawn_region=np.array([(-1, -1.8, 0), (2.5, 2.5, 1.5)],dtype=np.float64),
        obj_scale_range=np.array([0.15, 0.5],dtype=np.float64),
        min_drop_height=0.5,
        inner_radius=10,
        outer_radius=12,
        theta_min=10.0,
        theta_max=45.0,
        min_num_dynamic_objects=1,
        max_num_dynamic_objects=25,
        plane_asset_dir="./objs/plane_glbs",  # TODO: update path to your plane GLB assets
        min_tower_size=0, 
        max_tower_size=4,
        max_num_towers=6,
    )
  
  @staticmethod
  def _domino_config():
    return DominoConfig(
        static_spawn_region=np.array([(-2.0, -2.0, 0), (2.0, 2.5, 0)], dtype=np.float64),
        min_control_points=4,
        max_control_points=20,
        curvature_density_factor=1.0,  # Not used in new algorithm but kept for compatibility
        min_spacing=0.3,  # Increased minimum spacing based on domino dimensions
        max_spacing=1.0,  # Increased max spacing for flexibility
        control_point_pattern="random",  # Can be changed to other patterns
    )