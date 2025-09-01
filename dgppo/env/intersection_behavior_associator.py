import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict, Any
from functools import partial
import jax.random as jr
import jax
import jax.debug as jd
import jax.lax as lax
from jax.lax import cond

from dgppo.env.behavior_associator import BehaviorAssociator

class BehaviorIntersection(BehaviorAssociator):
    def __init__(self, intersections: List, all_region_names: List[str], buildings: List = [], bridges: List = [], obstacles: List = []):
        if not intersections:
            raise ValueError("BehaviorBuildings requires at least one building.")
        super().__init__(intersections, bridges, buildings, obstacles, all_region_names)

    def _define_behavior_regions(self) -> Dict[str, Any]:
        intersection_params = self.intersections[0]
        center = jnp.array(intersection_params[0])
        global_angle = intersection_params[3]
        passage_width = intersection_params[1]
        obs_len = intersection_params[2]
        is_four_way = intersection_params[4]
        
        half_gap = passage_width / 2.0

        def four_way_intersection(args):
            """Calculates regions for a four-way intersection."""
            center, half_gap, global_angle = args
            # Vertices of the inner intersection square, relative to a (0,0) center.
            inner_corners_local = jnp.array([
                [-half_gap, -half_gap],
                [half_gap, -half_gap],
                [half_gap, half_gap],
                [-half_gap, half_gap]
            ])
            
            # Rotation matrix for the global angle
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate and translate the central square
            rotated_corners = jnp.einsum('ij,kj->ki', rotation_matrix, inner_corners_local)
            in_intersection_region = rotated_corners + center
            
            # Define and transform the passage regions (trapezoids)
            passage_0_local = jnp.array([[-half_gap, half_gap], [half_gap, half_gap], [half_gap, 50], [-half_gap, 50]])
            passage_1_local = jnp.array([[half_gap, -half_gap], [half_gap, half_gap], [50, half_gap], [50, -half_gap]])
            passage_2_local = jnp.array([[-half_gap, -half_gap], [half_gap, -half_gap], [half_gap, -50], [-half_gap, -50]])
            passage_3_local = jnp.array([[-half_gap, -half_gap], [-half_gap, half_gap], [-50, half_gap], [-50, -half_gap]])

            rotated_passages = jnp.stack([
                jnp.einsum('ij,kj->ki', rotation_matrix, p) + center for p in [
                    passage_0_local, passage_1_local, passage_2_local, passage_3_local
                ]
            ])
            
            return in_intersection_region, rotated_passages

        def three_way_intersection(args):
            """Calculates regions for a three-way intersection and pads the output."""
            center, half_gap, global_angle, obs_len, passage_width = args
            # Define the vertices for the central trapezoid, relative to a (0,0) center.
            long_obs_len = obs_len + passage_width + obs_len
            inner_corners_local = jnp.array([
                [-long_obs_len / 2.0, obs_len / 2.0],
                [long_obs_len / 2.0, obs_len / 2.0],
                [half_gap, -obs_len / 2.0],
                [-half_gap, -obs_len / 2.0]
            ])

            # Rotation matrix for the global angle
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])

            rotated_corners = jnp.einsum('ij,kj->ki', rotation_matrix, inner_corners_local)
            in_intersection_region = rotated_corners + center

            # NOTE: The shapes of these arrays must all be the same (4 vertices per passage)
            passage_0_local = jnp.array([
                [-half_gap, -obs_len / 2.0], 
                [half_gap, -obs_len / 2.0], 
                [half_gap, -50], 
                [-half_gap, -50]
            ])
            
            # Corrected passage arrays to have 4 vertices
            passage_1_local = jnp.array([
                [-long_obs_len / 2.0, obs_len / 2.0], 
                [-50, 50], 
                [-50, obs_len / 2.0],
                [-long_obs_len / 2.0, obs_len / 2.0] # Duplicate vertex to make it a quad
            ])
            passage_2_local = jnp.array([
                [long_obs_len / 2.0, obs_len / 2.0], 
                [50, 50], 
                [50, obs_len / 2.0],
                [long_obs_len / 2.0, obs_len / 2.0] # Duplicate vertex to make it a quad
            ])
            
            # Pad the output with a "dummy" passage to match the four-way output shape
            # The dummy passage should also have 4 vertices
            dummy_passage = jnp.zeros_like(passage_0_local) 
            rotated_passages = jnp.stack([
                jnp.einsum('ij,kj->ki', rotation_matrix, p) + center for p in [
                    passage_0_local, passage_1_local, passage_2_local, dummy_passage
                ]
            ])
            
            return in_intersection_region, rotated_passages

        # Use jax.lax.cond as before
        in_intersection_region, rotated_passages = jax.lax.cond(
            is_four_way,
            partial(four_way_intersection, args=(center, half_gap, global_angle)),
            partial(three_way_intersection, args=(center, half_gap, global_angle, obs_len, passage_width))
        )

        regions = {}
        regions["open_space"] = jnp.array([(0, 0), (50, 0), (50, 50), (0, 50)], dtype=jnp.float32)
        regions["in_intersection"] = in_intersection_region
        
        for i, passage in enumerate(rotated_passages):
            regions[f"passage_{i}_enter"] = regions[f"passage_{i}_exit"] = passage

        return regions

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            "in_intersection": {"label": "Intersection", "color": "darkorange", "alpha": 0.8},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }
        is_four_way = self.intersections[0][4]
        num_passages = 4 if is_four_way else 3
        for p in range(num_passages):
            properties[f"passage_{p}_enter"] = {"label": f"Passage {p} Enter", "color": "lightblue", "alpha": 0.5}
            properties[f"passage_{p}_exit"] = {"label": f"Passage {p} Exit", "color": "lightgreen", "alpha": 0.5}
        return properties

    @partial(jax.jit, static_argnums=(0,))
    def get_current_behavior_direction(self, pos: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        """
        Determines the current region and categorizes enter/exit based on direction.
        Args:
            pos: Agent's position (2,)
            vel: Agent's velocity (2,)
        Returns:
            The integer ID of the resolved behavioral region.
        """
        current_region_id = self.get_current_behavior(pos)
        
        in_intersection_id = self.region_name_to_id["in_intersection"]
        centroid_in_intersection = self.get_region_centroid(in_intersection_id)
        
        vec_to_centroid = centroid_in_intersection - pos
        dot_product = jnp.dot(vel, vec_to_centroid)
        is_moving_towards = dot_product > 0.01 
        
        final_id = lax.cond(
            current_region_id == in_intersection_id,
            lambda: in_intersection_id,
            lambda: lax.cond(
                self.id_to_curriculum_prefix_map.get(current_region_id, "").startswith("passage_"),
                lambda: self._get_enter_exit_id(current_region_id, is_moving_towards),
                lambda: current_region_id
            )
        )
        return final_id

    def _get_enter_exit_id(self, passage_id, is_moving_towards):
        base_name = self.id_to_curriculum_prefix_map.get(passage_id, "")
        enter_id = self.region_name_to_id.get(base_name + "_enter", -1)
        exit_id = self.region_name_to_id.get(base_name + "_exit", -1)
        
        return lax.cond(is_moving_towards, lambda: enter_id, lambda: exit_id)