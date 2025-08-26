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

class BehaviorBridge(BehaviorAssociator):
    def __init__(self, bridges: List, buildings: List = [], obstacles: List = [], all_region_names: List[str] = []):
        if not bridges:
            raise ValueError("BehaviorBridge requires at least one bridge.")
        super().__init__(bridges, buildings, obstacles, all_region_names)

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
        shrink_factor = 1  # 5% shrink toward centroid

        # Open space
        regions["open_space"] = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)

        bridge_polygons = []

        for i, (cx, cy, l, w, a) in enumerate(self.bridges):
            rad = jnp.radians(a)
            half_l, half_w = l / 2.0, w / 2.0

            local_corners = jnp.array([
                [-half_l, -half_w],  # Bottom-left
                [ half_l, -half_w],  # Bottom-right
                [ half_l,  half_w],  # Top-right
                [-half_l,  half_w]   # Top-left
            ], dtype=jnp.float32)

            cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = jnp.einsum('ij,kj->ki', rotation_matrix, local_corners)
            corners = rotated + jnp.array([cx, cy])
            bridge_polygons.append(corners)

        if len(bridge_polygons) >= 2:
            c1, c2 = bridge_polygons[0], bridge_polygons[1]

            # On bridge polygon
            on_bridge = jnp.array([c1[3], c2[0], c2[1], c1[2]], dtype=jnp.float32)
            centroid = jnp.mean(on_bridge, axis=0)
            on_bridge_shrunk = centroid + shrink_factor * (on_bridge - centroid)
            regions["on_bridge_0"] = on_bridge_shrunk

            # Approach bridge polygon (semi-circle + entrance line)
            entrance_pts = jnp.array([c1[3], c2[0]], dtype=jnp.float32)
            center_approach = jnp.mean(entrance_pts, axis=0)
            width = jnp.linalg.norm(entrance_pts[0] - entrance_pts[1])
            radius = width / 2.0
            num_points = 30
            angle = jnp.radians(self.bridges[0][4])
            angles = jnp.linspace(angle + jnp.pi/2, angle + 3*jnp.pi/2, num_points)
            semi_arc = jnp.stack([
                center_approach[0] + radius * jnp.cos(angles),
                center_approach[1] + radius * jnp.sin(angles)
            ], axis=1)
            approach_poly = jnp.concatenate([entrance_pts[::-1], semi_arc[::-1]], axis=0)
            centroid_app = jnp.mean(approach_poly, axis=0)
            regions["approach_bridge_0"] = centroid_app + shrink_factor * (approach_poly - centroid_app)

            # Exit bridge polygon
            exit_pts = jnp.array([c1[2], c2[1]], dtype=jnp.float32)
            center_exit = jnp.mean(exit_pts, axis=0)
            angles_exit = jnp.linspace(angle - jnp.pi/2, angle + jnp.pi/2, num_points)
            semi_exit = jnp.stack([
                center_exit[0] + radius * jnp.cos(angles_exit),
                center_exit[1] + radius * jnp.sin(angles_exit)
            ], axis=1)
            exit_poly = jnp.concatenate([exit_pts[::-1], semi_exit[::-1]], axis=0)
            centroid_exit = jnp.mean(exit_poly, axis=0)
            regions["exit_bridge_0"] = centroid_exit + shrink_factor * (exit_poly - centroid_exit)

        return regions



    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping bridge region names to visualization properties.
        """
        return {
            "on_bridge_0": {"label": "On Bridge", "color": "skyblue"},
            "approach_bridge_0": {"label": "Approach Bridge", "color": "lightgreen"},
            "exit_bridge_0": {"label": "Exit Bridge", "color": "lightcoral"},
            # You can add more regions here if your `_define_behavior_regions` creates them
            # E.g., "around_bridge_side" or "under_bridge"
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }


    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid



# import jax
# import jax.numpy as jnp
# from functools import partial
# from dgppo.env.behavior_associator import BehaviorAssociator

# class BehaviorBridge(BehaviorAssociator):
#     def __init__(self, bridges, buildings=[], obstacles=[], all_region_names=[]):
#         if not bridges:
#             raise ValueError("BehaviorBridge requires at least one bridge.")
#         super().__init__(bridges, buildings, obstacles, all_region_names)

#     def _define_behavior_regions(self):
#         """
#         Returns dict of region_name -> polygon (N,2) JAX arrays
#         """
#         regions = {}

#         # Open space
#         regions["open_space"] = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)

#         # Bridges
#         individual_bridge_corners = []
#         for i, (cx, cy, l, w, a) in enumerate(self.bridges):
#             rad = jnp.radians(a)
#             half_l, half_w = l / 2.0, w / 2.0

#             local_corners = jnp.array([
#                 [-half_l, -half_w],
#                 [ half_l, -half_w],
#                 [ half_l,  half_w],
#                 [-half_l,  half_w]
#             ], dtype=jnp.float32)

#             cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
#             rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
#             rotated = jnp.einsum('ij,kj->ki', rotation_matrix, local_corners)
#             corners_obstacle = rotated + jnp.array([cx, cy])
#             individual_bridge_corners.append(corners_obstacle)

#         if len(self.bridges) >= 2:
#             c1, c2 = individual_bridge_corners[0], individual_bridge_corners[1]

#             # On bridge region
#             on_bridge = jnp.array([c1[3], c2[0], c2[1], c1[2]])
#             regions["on_bridge_0"] = on_bridge

#             # Approach bridge
#             channel_entrance = jnp.array([c1[3], c2[0]])
#             center_approach = jnp.mean(channel_entrance, axis=0)
#             width = jnp.linalg.norm(channel_entrance[0] - channel_entrance[1])
#             radius = width / 2.0
#             num_points = 30
#             angle = jnp.radians(self.bridges[0][4])
#             angles = jnp.linspace(angle + jnp.pi/2, angle + 3*jnp.pi/2, num_points)
#             semi_arc = jnp.stack([
#                 center_approach[0] + radius * jnp.cos(angles),
#                 center_approach[1] + radius * jnp.sin(angles)
#             ], axis=1)
#             regions["approach_bridge_0"] = jnp.concatenate([channel_entrance[::-1], semi_arc[::-1]], axis=0)

#             # Exit bridge
#             channel_exit = jnp.array([c1[2], c2[1]])
#             center_exit = jnp.mean(channel_exit, axis=0)
#             angles_exit = jnp.linspace(angle - jnp.pi/2, angle + jnp.pi/2, num_points)
#             semi_exit = jnp.stack([
#                 center_exit[0] + radius * jnp.cos(angles_exit),
#                 center_exit[1] + radius * jnp.sin(angles_exit)
#             ], axis=1)
#             regions["exit_bridge_0"] = jnp.concatenate([channel_exit[::-1], semi_exit[::-1]], axis=0)

    #     # Convert all lists to JAX arrays
    #     for key, value in regions.items():
    #         if isinstance(value, list):
    #             regions[key] = jnp.asarray(value, dtype=jnp.float32)

    #     return regions

    # def _get_region_visualization_properties(self):
    #     """
    #     Returns a dictionary mapping bridge region names to visualization properties
    #     """
    #     return {
    #         "on_bridge_0": {"label": "On Bridge", "color": "skyblue"},
    #         "approach_bridge_0": {"label": "Approach Bridge", "color": "lightgreen"},
    #         "exit_bridge_0": {"label": "Exit Bridge", "color": "lightcoral"},
    #         "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
    #     }
