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
            """
            Calculates regions for a four-way intersection.
            The central and passage regions are defined by the inner corners of
            obstacles placed at the grid's edges and rotated.
            """
            area_size, obs_len, global_angle = args
            area_size = jnp.array(area_size, float)
            obs_len = jnp.array(obs_len, float)
            half_obs_len = obs_len / 2.0
            
            # Get the fixed positions of the obstacle centers
            obs_pos = jnp.array([
                [0.0, 0.0],
                [area_size, 0.0],
                [area_size, area_size],
                [0.0, area_size]
            ])

            # Get the rotated obstacle angles
            obs_theta = jnp.array([0.0, 0.0, jnp.pi / 2.0, jnp.pi / 2.0]) + global_angle

            # FIX: The local inner corners must be geometrically correct to form the passages.
            # This is the correct array for the inner corners.
            local_inner_corners = jnp.array([
                [half_obs_len, half_obs_len],    # Obstacle at (0, 0)
                [-half_obs_len, half_obs_len],   # Obstacle at (area_size, 0)
                [-half_obs_len, -half_obs_len],  # Obstacle at (area_size, area_size)
                [half_obs_len, -half_obs_len]    # Obstacle at (0, area_size)
            ])
            
            # Create a single rotation matrix for the global angle
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Use einsum to efficiently rotate and translate all four corners
            rotated_in_intersection = (jnp.einsum('ij,kj->ki', rotation_matrix, local_inner_corners) + obs_pos)

            # NEW: Define passage regions based on inner corners and an outward extension
            passage_len = 0.3  # Length of the passage polygon, as requested
            outward_vectors_local = jnp.array([
                [0.0, -1.0],  # Bottom passage direction
                [1.0, 0.0],  # Left passage direction
                [0.0, 1.0],   # Right passage direction
                [-1.0, 0.0]    # Top passage direction
            ])
            # The local passages are trapezoids defined by two inner corners and two new points
            # created by extending outwards.
            # Bottom Passage (connecting inner corners of obstacles 0 and 1)
            rotated_outward_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, outward_vectors_local)

            # Construct the passage polygons directly from the rotated_in_intersection vertices
            rotated_passages = jnp.stack([
                jnp.array([
                    rotated_in_intersection[0],
                    rotated_in_intersection[1],
                    rotated_in_intersection[1] + rotated_outward_vectors[0] * passage_len,
                    rotated_in_intersection[0] + rotated_outward_vectors[0] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[1],
                    rotated_in_intersection[2],
                    rotated_in_intersection[2] + rotated_outward_vectors[1] * passage_len,
                    rotated_in_intersection[1] + rotated_outward_vectors[1] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[2],
                    rotated_in_intersection[3],
                    rotated_in_intersection[3] + rotated_outward_vectors[2] * passage_len,
                    rotated_in_intersection[2] + rotated_outward_vectors[2] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[3],
                    rotated_in_intersection[0],
                    rotated_in_intersection[0] + rotated_outward_vectors[3] * passage_len,
                    rotated_in_intersection[3] + rotated_outward_vectors[3] * passage_len
                ])
            ])
            
            open_space_len = 0.45
            open_space_polygons = jnp.stack([
                jnp.array([
                    rotated_passages[0][2],
                    rotated_passages[0][3],
                    rotated_passages[0][3] + rotated_outward_vectors[0] * open_space_len,
                    rotated_passages[0][2] + rotated_outward_vectors[0] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[1][2],
                    rotated_passages[1][3],
                    rotated_passages[1][3] + rotated_outward_vectors[1] * open_space_len,
                    rotated_passages[1][2] + rotated_outward_vectors[1] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[2][2],
                    rotated_passages[2][3],
                    rotated_passages[2][3] + rotated_outward_vectors[2] * open_space_len,
                    rotated_passages[2][2] + rotated_outward_vectors[2] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[3][2],
                    rotated_passages[3][3],
                    rotated_passages[3][3] + rotated_outward_vectors[3] * open_space_len,
                    rotated_passages[3][2] + rotated_outward_vectors[3] * open_space_len
                ])
            ])

            return rotated_in_intersection, rotated_passages, open_space_polygons
        
        def three_way_intersection(args):
            """
            Calculates regions for a three-way intersection using the specified logic.
            """
            area_size, obs_len, global_angle = args
            half_obs_len = obs_len / 2.0

            grid_corners = jnp.array([
                [0.0, 0.0],
                [area_size, 0.0],
                [area_size, area_size],
                [0.0, area_size]
            ])

            # Determine the inner corners for the central region
            # This is a trapezoid connecting the inner corners of the three effective obstacles.

            # Inner corner of small obstacle at top-left (grid_corners[3])
            c3_inner = grid_corners[3] + jnp.array([half_obs_len, -half_obs_len])
            
            # Inner corner of small obstacle at top-right (grid_corners[2])
            c2_inner = grid_corners[2] + jnp.array([-half_obs_len, -half_obs_len])
            
            # Inner corner of the large obstacle (effectively from grid_corners[0] and grid_corners[1])
            # The bottom edge of the central region will be from (0 + half_obs_len, 0 + half_obs_len)
            # to (area_size - half_obs_len, 0 + half_obs_len)
            
            c0_inner_for_large_obs = grid_corners[0] + jnp.array([half_obs_len, half_obs_len])
            c1_inner_for_large_obs = grid_corners[1] + jnp.array([-half_obs_len, half_obs_len])
            
            in_intersection_region = jnp.array([
                c3_inner,  # Top-left vertex of trapezoid
                c2_inner,  # Top-right vertex of trapezoid
                c1_inner_for_large_obs, # Bottom-right vertex of trapezoid
                c0_inner_for_large_obs  # Bottom-left vertex of trapezoid
            ])

            # Define the three outward-facing passages using the central region's vertices
            passage_len = 0.25
            
            # Outward vectors for the three passages (Left, Right, Bottom)
            # Note: The order here corresponds to how we'll extract the sides from in_intersection_region
            outward_vectors_local = jnp.array([
                [-1.0, 0.0],   # Left passage direction
                [1.0, 0.0],    # Right passage direction
                [0.0, -1.0]    # Bottom passage direction
            ])
            
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_outward_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, outward_vectors_local)

            # Construct the passage polygons
            rotated_passages = jnp.stack([
                jnp.array([ # Left passage (connects c3_inner and c0_inner_for_large_obs)
                    in_intersection_region[0], # c3_inner
                    in_intersection_region[3], # c0_inner_for_large_obs
                    in_intersection_region[3] + rotated_outward_vectors[0] * passage_len,
                    in_intersection_region[0] + rotated_outward_vectors[0] * passage_len
                ]),
                jnp.array([ # Right passage (connects c2_inner and c1_inner_for_large_obs)
                    in_intersection_region[1], # c2_inner
                    in_intersection_region[2], # c1_inner_for_large_obs
                    in_intersection_region[2] + rotated_outward_vectors[1] * passage_len,
                    in_intersection_region[1] + rotated_outward_vectors[1] * passage_len
                ]),
                jnp.array([ # Bottom passage (connects c0_inner_for_large_obs and c1_inner_for_large_obs)
                    in_intersection_region[3], # c0_inner_for_large_obs
                    in_intersection_region[2], # c1_inner_for_large_obs
                    in_intersection_region[2] + rotated_outward_vectors[2] * passage_len,
                    in_intersection_region[3] + rotated_outward_vectors[2] * passage_len
                ]),
                jnp.zeros((4, 2)) # Dummy passage for consistency with 4-way output
            ])
            
            # Apply global rotation and translation
            rotated_in_intersection = jnp.einsum('ij,kj->ki', rotation_matrix, in_intersection_region - center) + center
            rotated_passages = jnp.einsum('ij,kj->ki', rotation_matrix, rotated_passages.reshape(-1, 2) - center) + center
            rotated_passages = rotated_passages.reshape(4, 4, 2)

            return rotated_in_intersection, rotated_passages, rotated_passages
        
        # Use jax.lax.cond as before
        in_intersection, passage_polygons, open_space_polygons = jax.lax.cond(
            is_four_way,
            four_way_intersection,
            three_way_intersection,
            (1.5, obs_len, global_angle)
        )

        regions = {}
        regions["in_intersection"] = in_intersection

        for i, passage in enumerate(passage_polygons):
            # Assign specific passage polygons to both enter and exit
            regions[f"passage_{i}_enter"] = passage_polygons[i]
            regions[f"passage_{i}_exit"] = passage_polygons[i]
            
            # Assign the corresponding open_space polygon
            regions[f"open_space_{i}"] = open_space_polygons[i]
            
        return regions

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            "in_intersection": {"label": "Intersection", "color": "darkorange", "alpha": 0.8},
        }
        is_four_way = self.intersections[0][4]
        num_passages = 4 if is_four_way else 3
        for p in range(num_passages):
            properties[f"passage_{p}_enter"] = {"label": f"Passage {p} Enter", "color": "lightblue", "alpha": 0.5}
            properties[f"passage_{p}_exit"] = {"label": f"Passage {p} Exit", "color": "lightgreen", "alpha": 0.5}
            properties[f"open_space_{p}"] = {"label": f"Open Space {p}", "color": "yellow", "alpha": 0.1}

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
        
        # Determine if the current region is a passage using JAX-compatible logic
        is_passage = jnp.logical_and(current_region_id >= self.passage_start_id, current_region_id < self.passage_end_id)
        
        final_id = lax.cond(
            current_region_id == in_intersection_id,
            lambda: in_intersection_id,
            lambda: lax.cond(
                is_passage,
                lambda: self._get_enter_exit_id(current_region_id, is_moving_towards),
                lambda: current_region_id
            )
        )
        return final_id

    def _get_enter_exit_id(self, passage_id, is_moving_towards):
        base_passage_idx = (passage_id - self.passage_start_id) // 2

        # Calculate the IDs for both enter and exit
        enter_id = self.passage_start_id + base_passage_idx * 2
        exit_id = self.passage_start_id + base_passage_idx * 2 + 1

        # Use jnp.where to choose the correct ID based on the condition
        return jnp.where(is_moving_towards, enter_id, exit_id)


def visualize_intersection_terrain(
    ax,
    center: np.ndarray,    # (2,) center of intersection, in 0-to-side_length coords
    passage_width: float,
    global_angle: float,   # radians
    side_length: float,
    terrain_config: int = 1,
    resolution: int = 200,
) -> list:
    """
    Draw intersection terrain zones as a raster background.
    Mirrors get_intersection_terrain_id in base.py exactly.

    Terrain IDs: 0=road (gray), 1=sidewalk (tan), 2=grass (green)
    Grass = expanded per-obstacle box (obs_len + 2*SLIVER), peeking 0.01 past each obstacle edge.
    Config 1: all sidewalk in passages
    Config 2: road centred in each arm, sidewalk borders (SIDEWALK_BORDER wide)
    Config 3: sidewalk throughout, small grass disk at intersection centre
    """
    from matplotlib.patches import Patch

    SLIVER = 0.01
    SIDEWALK_BORDER = 0.12
    GRASS_PATCH_RADIUS = 0.10

    obs_len = side_length - passage_width   # area_size - (area_size - obs_len)
    obs_half = obs_len / 2.0 + SLIVER
    S = side_length

    xs = np.linspace(0, side_length, resolution)
    ys = np.linspace(0, side_length, resolution)
    XX, YY = np.meshgrid(xs, ys)

    # Per-obstacle grass: expanded box for each of the 4 corner obstacles
    # Obstacle thetas: [0,0],[S,0] → global_angle;  [0,S],[S,S] → pi/2+global_angle
    def _in_obs(cx, cy, theta):
        rx = XX - cx;  ry = YY - cy
        lx = np.cos(-theta) * rx - np.sin(-theta) * ry
        ly = np.sin(-theta) * rx + np.cos(-theta) * ry
        return (np.abs(lx) <= obs_half) & (np.abs(ly) <= obs_half)

    in_grass = (
        _in_obs(0., 0., global_angle) |
        _in_obs(S,  0., global_angle) |
        _in_obs(0., S,  np.pi / 2.0 + global_angle) |
        _in_obs(S,  S,  np.pi / 2.0 + global_angle)
    )

    # Passage terrain in intersection-local frame
    cos_t = np.cos(-global_angle);  sin_t = np.sin(-global_angle)
    rel_x = XX - center[0];  rel_y = YY - center[1]
    local_x = cos_t * rel_x - sin_t * rel_y
    local_y = sin_t * rel_x + cos_t * rel_y

    half_pass = passage_width / 2.0
    in_h = np.abs(local_y) < half_pass
    in_v = np.abs(local_x) < half_pass
    in_center_sq = in_h & in_v
    in_h_arm = in_h & ~in_center_sq
    in_v_arm = in_v & ~in_center_sq

    if terrain_config == 1:
        terrain_grid = np.where(in_grass, 2, 1)

    elif terrain_config == 2:
        h_road = in_h_arm & (np.abs(local_y) < half_pass - SIDEWALK_BORDER)
        v_road = in_v_arm & (np.abs(local_x) < half_pass - SIDEWALK_BORDER)
        is_road = in_center_sq | h_road | v_road
        terrain_grid = np.where(in_grass, 2, np.where(is_road, 0, 1))

    else:  # config 3
        dist_from_center = np.sqrt(local_x**2 + local_y**2)
        in_central_grass = in_center_sq & (dist_from_center <= GRASS_PATCH_RADIUS)
        terrain_grid = np.where(in_grass | in_central_grass, 2, 1)

    road_rgba  = (0.55, 0.55, 0.55, 0.45)
    side_rgba  = (0.88, 0.78, 0.50, 0.50)
    grass_rgba = (0.45, 0.72, 0.35, 0.35)

    rgba = np.zeros((resolution, resolution, 4))
    rgba[terrain_grid == 0] = road_rgba
    rgba[terrain_grid == 1] = side_rgba
    rgba[terrain_grid == 2] = grass_rgba

    ax.imshow(rgba, extent=[0, side_length, 0, side_length], origin='lower', zorder=0)

    return [
        Patch(facecolor=road_rgba[:3],  alpha=road_rgba[3],  label="Road"),
        Patch(facecolor=side_rgba[:3],  alpha=side_rgba[3],  label="Sidewalk"),
        Patch(facecolor=grass_rgba[:3], alpha=grass_rgba[3], label="Grass"),
    ]
