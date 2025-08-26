import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict, Any
from functools import partial
import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np # Keep if needed for other non-JAX ops, but avoid for JAX arrays
import functools as ft
import jax
import jax.debug as jd

# You must import the base BehaviorAssociator class
from dgppo.env.behavior_associator import BehaviorAssociator

class BehaviorIntersection(BehaviorAssociator):
    def __init__(self, intersections: List, bridges: List = [], buildings: List = [], obstacles: List = [], all_region_names = []):
        if not intersections:
            raise ValueError("BehaviorIntersection requires at least one intersection.")
        all_region_names = ["open_space", "in_intersection"]
        passages = ["0", "1", "2", "3"]
        for p in passages:
            all_region_names.append(f"passage_{p}_enter")
            all_region_names.append(f"passage_{p}_exit")
        super().__init__(intersections, bridges, buildings, obstacles, all_region_names)

    def _get_rectangle_corners_np(self, center, length, width, angle_rad):
        half_l = length / 2.0
        half_w = width / 2.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        local_corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])

        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        final_corners = rotated_corners + center
        return final_corners

    def _define_behavior_regions(self):
        regions = {}
        regions["open_space"] = jnp.array([(0, 0), (50, 0), (50, 50), (0, 50)], dtype=jnp.float32)

        # 1. Randomly generate parameters for 4 obstacles
        self.key, subkey = jr.split(self.key)
        rand_params = jr.uniform(subkey, shape=(4, 5), minval=jnp.array([10, 10, 15, 9, 0]), maxval=jnp.array([20, 20, 25, 12, 360]))

        # Define obstacle centers relative to grid corners
        centers = jnp.array([
            [rand_params[0, 0], rand_params[0, 1]],  # top_left
            [50 - rand_params[1, 0], rand_params[1, 1]], # top_right
            [50 - rand_params[2, 0], 50 - rand_params[2, 1]], # bottom_right
            [rand_params[3, 0], 50 - rand_params[3, 1]], # bottom_left
        ])
        
        # Adjust angles to keep them pointing towards the center
        angles = jnp.array([
            rand_params[0, 4] % 90,
            (rand_params[1, 4] % 90) + 90,
            (rand_params[2, 4] % 90) + 180,
            (rand_params[3, 4] % 90) + 270,
        ])

        self.obstacles_def = {}
        all_intersection_corners = []
        corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        for i in range(4):
            center = centers[i]
            length = rand_params[i, 2]
            width = rand_params[i, 3]
            angle = angles[i]
            self.obstacles_def[corner_names[i]] = {"center": center, "size": (length, width), "angle": angle}

            rad = np.radians(angle)
            corners = self._get_rectangle_corners_np(center, length, width, rad)
            all_intersection_corners.append(jnp.asarray(corners, dtype=jnp.float32))

        # 2. Define the IN INTERSECTION (ORANGE) polygon
        inner_corners_points = jnp.array([
            all_intersection_corners[0][2], # Top-left obstacle's bottom-right corner
            all_intersection_corners[1][3], # Top-right obstacle's bottom-left corner
            all_intersection_corners[2][0], # Bottom-right obstacle's top-left corner
            all_intersection_corners[3][1]  # Bottom-left obstacle's top-right corner
        ])
        
        regions["in_intersection"] = inner_corners_points

        # 3. Define the ELLIPTICAL passage regions
        passages_def = {
            "0": (all_intersection_corners[0][2], all_intersection_corners[1][3]),
            "1": (all_intersection_corners[1][2], all_intersection_corners[2][1]),
            "2": (all_intersection_corners[2][2], all_intersection_corners[3][1]),
            "3": (all_intersection_corners[3][2], all_intersection_corners[0][1]),
        }
        
        for name, (p1, p2) in passages_def.items():
            regions[f"passage_{name}_enter"] = self._create_elliptical_region(p1, p2)
            regions[f"passage_{name}_exit"] = self._create_elliptical_region(p1, p2)
            
        return regions
    
    def _create_elliptical_region(self, p1: jnp.ndarray, p2: jnp.ndarray):
        center = (p1 + p2) / 2.0
        major_axis = jnp.linalg.norm(p1 - p2)
        minor_axis = major_axis * 0.5 
        angle_rad = jnp.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_deg = np.degrees(angle_rad)

        return ("ellipse", (center[0], center[1], major_axis, minor_axis, angle_deg))
    
    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            "in_intersection": {"label": "Intersection", "color": "darkorange", "alpha": 0.8},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }
        passages = ["0", "1", "2", "3"]
        for p in passages:
            properties[f"passage_{p}_enter"] = {"label": f"Passage {p} Enter", "color": "lightblue", "alpha": 0.5}
            properties[f"passage_{p}_exit"] = {"label": f"Passage {p} Exit", "color": "lightgreen", "alpha": 0.5}
        return properties

    # @partial(jax.jit, static_argnums=(0,))
    # def get_region_centroid(self, region_id: jnp.ndarray):
    #     centroid = self.all_region_centroids_jax_array[region_id]
    #     return centroid
    
    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        """
        Returns the centroid of a specified region, handling both polygons and ellipses.
        """
        # Get the name of the region from its ID
        name = self.region_id_to_name[region_id]
        
        # Retrieve the data for the specified region
        region_data = self.behavior_regions[name]
        
        # Check the type of the region using JAX-compatible logic
        is_polygon = isinstance(region_data, jnp.ndarray) and region_data.ndim == 2
        is_ellipse = isinstance(region_data, tuple) and region_data[0] == "ellipse"

        # Define the centroid calculation functions for each type
        def polygon_centroid():
            return jnp.mean(region_data, axis=0)

        def ellipse_centroid():
            # The ellipse data tuple is ("ellipse", (center_x, center_y, ...))
            return jnp.array([region_data[1][0], region_data[1][1]])

        # Use jax.lax.cond to select the correct function based on the region type
        centroid = jax.lax.cond(is_polygon, polygon_centroid, ellipse_centroid)
        return centroid