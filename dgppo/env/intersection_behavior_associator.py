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
    def __init__(self, key: jr.PRNGKey, is_four_way: bool, intersection_params: Dict, obstacles: List = [], all_region_names: List[str] = [], params=None, area_size=None):
        self.key = key
        self.params = params if params is not None else {}
        self.all_region_names = all_region_names
        self.area_size = area_size
        self.obstacles = obstacles
        self.is_four_way = is_four_way
        self.intersection_params = intersection_params
        
        # Conditionally determine `self.is_four_way` based on the input argument
        if is_four_way is None:
            key_is_four_way, self.key = jax.random.split(self.key)
            is_four_way_p = self.params.get('is_four_way_p', 0.5)
            self.is_four_way = bool(jax.random.uniform(key_is_four_way) < is_four_way_p)
        else:
            self.is_four_way = is_four_way

        # Conditionally determine `self.intersection_params` based on the input argument
        if intersection_params is None:
            if self.area_size is None:
                raise ValueError("area_size must be provided to randomly generate intersection parameters.")
                
            key_intersection, self.key = jax.random.split(self.key)
            key_center, key_size = jax.random.split(key_intersection)
            
            self.intersection_params = {
                'center': jax.random.uniform(key_center, shape=(2,), minval=-self.area_size / 2, maxval=self.area_size / 2),
                'size': jax.random.uniform(key_size, shape=(), minval=5.0, maxval=10.0),
                'passage_min': 0.25,  # Example default value, adjust as needed
                'passage_max': 0.5,  # Example default value, adjust as needed
                'obs_len_min': 0.4, # Example default value, adjust as needed
                'obs_len_max': 0.75, # Example default value, adjust as needed
            }
        else:
            self.intersection_params = intersection_params

        # Generate region names based on the determined `self.is_four_way`
        num_passages = 4 if self.is_four_way else 3
        self.all_region_names = ["open_space", "in_intersection"]
        
        for i in range(num_passages):
            self.all_region_names.append(f"passage_{i}_enter")
            self.all_region_names.append(f"passage_{i}_exit")

        super().__init__(
            key=self.key,
            all_region_names=self.all_region_names,
            area_size=self.area_size,
            **self.params
        )
    
    def _define_behavior_regions(self):
        """
        Defines the polygonal regions of the intersection using JAX.
        This function is designed to be pure and JIT-compatible.
        """
        key, subkey_center, subkey_angle, subkey_dims = jr.split(self.key, 4)

        regions = {}
        regions["open_space"] = jnp.array([(0, 0), (50, 0), (50, 50), (0, 50)], dtype=jnp.float32)

        # Randomly determine the center, rotation, and dimensions of the intersection
        center = jr.uniform(subkey_center, shape=(2,), minval=15.0, maxval=35.0)
        global_angle = jr.uniform(subkey_angle, shape=(), minval=0.0, maxval=2 * jnp.pi)
        
        passage_width = jr.uniform(subkey_dims, minval=self.intersection_params["passage_min"], maxval=self.intersection_params["passage_max"])
        obs_len = jr.uniform(subkey_dims, shape=(), minval=self.intersection_params["obs_len_min"], maxval=self.intersection_params["obs_len_max"])
        
        half_gap = passage_width / 2.0
        
        if self.is_four_way:
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
            regions["in_intersection"] = rotated_corners + center
            
            # Define and transform the passage regions (trapezoids)
            passage_0_local = jnp.array([[-half_gap, half_gap], [half_gap, half_gap], [half_gap, 50], [-half_gap, 50]])
            passage_1_local = jnp.array([[half_gap, -half_gap], [half_gap, half_gap], [50, half_gap], [50, -half_gap]])
            passage_2_local = jnp.array([[-half_gap, -half_gap], [half_gap, -half_gap], [half_gap, -50], [-half_gap, -50]])
            passage_3_local = jnp.array([[-half_gap, -half_gap], [-half_gap, half_gap], [-50, half_gap], [-50, -half_gap]])

            rotated_passages = [
                jnp.einsum('ij,kj->ki', rotation_matrix, p) + center for p in [
                    passage_0_local, passage_1_local, passage_2_local, passage_3_local
                ]
            ]
            regions["passage_0_enter"] = regions["passage_0_exit"] = rotated_passages[0]
            regions["passage_1_enter"] = regions["passage_1_exit"] = rotated_passages[1]
            regions["passage_2_enter"] = regions["passage_2_exit"] = rotated_passages[2]
            regions["passage_3_enter"] = regions["passage_3_exit"] = rotated_passages[3]

        else: # 3-way intersection
            # Define the vertices for the central trapezoid, relative to a (0,0) center.
            long_obs_len = obs_len + passage_width + obs_len
            inner_corners_local = jnp.array([
                [-long_obs_len / 2.0, obs_len / 2.0],  # Top-left vertex
                [long_obs_len / 2.0, obs_len / 2.0],   # Top-right vertex
                [half_gap, -obs_len / 2.0],              # Bottom-right vertex
                [-half_gap, -obs_len / 2.0]               # Bottom-left vertex
            ])

            # Rotation matrix for the global angle
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Rotate and translate the central trapezoid
            rotated_corners = jnp.einsum('ij,kj->ki', rotation_matrix, inner_corners_local)
            regions["in_intersection"] = rotated_corners + center

            # Define and transform the passage regions for the 3-way
            passage_0_local = jnp.array([
                [-half_gap, -obs_len / 2.0],
                [half_gap, -obs_len / 2.0],
                [half_gap, -50],
                [-half_gap, -50]
            ])
            passage_1_local = jnp.array([
                [-long_obs_len / 2.0, obs_len / 2.0],
                [-50, 50],
                [-50, obs_len / 2.0]
            ])
            passage_2_local = jnp.array([
                [long_obs_len / 2.0, obs_len / 2.0],
                [50, 50],
                [50, obs_len / 2.0]
            ])

            rotated_passages = [
                jnp.einsum('ij,kj->ki', rotation_matrix, p) + center for p in [
                    passage_0_local, passage_1_local, passage_2_local
                ]
            ]
            regions["passage_0_enter"] = regions["passage_0_exit"] = rotated_passages[0]
            regions["passage_1_enter"] = regions["passage_1_exit"] = rotated_passages[1]
            regions["passage_2_enter"] = regions["passage_2_exit"] = rotated_passages[2]

        return regions

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            "in_intersection": {"label": "Intersection", "color": "darkorange", "alpha": 0.8},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }
        num_passages = 4 if self.is_four_way else 3
        for p in range(num_passages):
            properties[f"passage_{p}_enter"] = {"label": f"Passage {p} Enter", "color": "lightblue", "alpha": 0.5}
            properties[f"passage_{p}_exit"] = {"label": f"Passage {p} Exit", "color": "lightgreen", "alpha": 0.5}
        return properties

    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        """Returns the centroid of a specified polygon region."""
        # This will work because we are only using polygons now
        centroid = jnp.mean(self.behavior_regions_jax_array[region_id], axis=0)
        return centroid
        
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