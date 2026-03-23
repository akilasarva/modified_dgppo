import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict, Any

# Assuming the base BehaviorAssociator class is in the same file
from dgppo.env.behavior_associator import BehaviorAssociator

class BehaviorBuildings(BehaviorAssociator):
    def __init__(self, buildings: List, all_region_names: List[str], bridges: List = [], obstacles: List = []):
        if not buildings:
            raise ValueError("BehaviorBuildings requires at least one building.")
        super().__init__(bridges, buildings, obstacles, all_region_names)


    def _define_behavior_regions(self) -> Dict[str, Any]:
        regions = {}
        regions["open_space"] = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)

        if len(self.buildings) >= 1:
            building_params = self.buildings[0]
            building_center = jnp.array(building_params[0])
            building_width = building_params[1]
            building_height = building_params[2]
            building_theta = building_params[3]

            cos_theta, sin_theta = jnp.cos(building_theta), jnp.sin(building_theta)
            rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            half_width = building_width / 2.0
            half_height = building_height / 2.0

            wall_thickness = half_width
            corner_size = half_width*2/3

            regions_local = {}
            
            along_wall_length_offset = half_width / 4

            # Along Wall Regions (shortened to prevent overlap)
            regions_local["along_wall_0"] = jnp.array([
                [-half_width + along_wall_length_offset, half_height],
                [half_width - along_wall_length_offset, half_height],
                [half_width - along_wall_length_offset, half_height + wall_thickness],
                [-half_width + along_wall_length_offset, half_height + wall_thickness],
            ])
            regions_local["along_wall_1"] = jnp.array([
                [half_width, -half_height + along_wall_length_offset],
                [half_width + wall_thickness, -half_height + along_wall_length_offset],
                [half_width + wall_thickness, half_height - along_wall_length_offset],
                [half_width, half_height - along_wall_length_offset],
            ])
            regions_local["along_wall_2"] = jnp.array([
                [-half_width + along_wall_length_offset, -half_height],
                [half_width - along_wall_length_offset, -half_height],
                [half_width - along_wall_length_offset, -half_height - wall_thickness],
                [-half_width + along_wall_length_offset, -half_height - wall_thickness],
            ])
            regions_local["along_wall_3"] = jnp.array([
                [-half_width, -half_height + along_wall_length_offset],
                [-half_width - wall_thickness, -half_height + along_wall_length_offset],
                [-half_width - wall_thickness, half_height - along_wall_length_offset],
                [-half_width, half_height - along_wall_length_offset],
            ])

                    
            # Past Building Regions (the blue boxes that fit in the corners)
            regions_local["past_building_0"] = jnp.array([
                [-half_width - wall_thickness, half_height - along_wall_length_offset],
                [-half_width + along_wall_length_offset, half_height - along_wall_length_offset],
                [-half_width + along_wall_length_offset, half_height + wall_thickness],
                [-half_width - wall_thickness, half_height + wall_thickness],
            ])
            regions_local["past_building_1"] = jnp.array([
                [half_width - along_wall_length_offset, half_height - along_wall_length_offset],
                [half_width + wall_thickness, half_height - along_wall_length_offset],
                [half_width + wall_thickness, half_height + wall_thickness],
                [half_width - along_wall_length_offset, half_height + wall_thickness],
            ])
            regions_local["past_building_2"] = jnp.array([
                [half_width - along_wall_length_offset, -half_height + along_wall_length_offset],
                [half_width + wall_thickness, -half_height + along_wall_length_offset],
                [half_width + wall_thickness, -half_height - wall_thickness],
                [half_width - along_wall_length_offset, -half_height - wall_thickness],
            ])
            regions_local["past_building_3"] = jnp.array([
                [-half_width - wall_thickness, -half_height + along_wall_length_offset],
                [-half_width + along_wall_length_offset, -half_height + along_wall_length_offset],
                [-half_width + along_wall_length_offset, -half_height - wall_thickness],
                [-half_width - wall_thickness, -half_height - wall_thickness],
            ])
            
            # Around Corner Arc Regions
            arc_radius = half_width/4
            num_points = 30

            local_corners = {
                "around_corner_0": jnp.array([-half_width, half_height]),
                "around_corner_1": jnp.array([half_width, half_height]),
                "around_corner_2": jnp.array([half_width, -half_height]),
                "around_corner_3": jnp.array([-half_width, -half_height]),
            }

            corner_angles = {
                "around_corner_0": (0, 3*jnp.pi/2),
                "around_corner_3": (jnp.pi / 2, 2*jnp.pi),
                "around_corner_2": (jnp.pi, 5*jnp.pi/2),
                "around_corner_1": (3*jnp.pi / 2, 3*jnp.pi),
            }
        

            # The loop now uses the correct names
            for corner_name, corner_pos in local_corners.items():
                start_angle, end_angle = corner_angles[corner_name]
                angles = jnp.linspace(start_angle, end_angle, num_points)
                
                arc_points_local = jnp.array([
                    corner_pos[0] + arc_radius * jnp.cos(angles),
                    corner_pos[1] + arc_radius * jnp.sin(angles)
                ]).T
                
                closed_polygon_local = jnp.concatenate([
                    arc_points_local,
                    jnp.array([corner_pos])
                ], axis=0)
                
                # The name is now correctly a key that the rest of the code expects
                regions_local[corner_name] = closed_polygon_local

            # Final loop to apply rotation and translation to ALL local polygons
            for name, poly in regions_local.items():
                regions[name] = jnp.einsum('ij,kj->ki', rotation_matrix, poly) + building_center
            
        return regions

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            #"in_building": {"label": "In Building", "color": "darkred", "alpha": 0.8},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }
        # Assign properties for 'past_building' regions
        for i in range(4):
            properties[f"past_building_{i}"] = {"label": "Past Building", "color": "blue", "alpha": 0.3}

        # Assign properties for 'along_wall' regions
        for i in range(4):
            properties[f"along_wall_{i}"] = {"label": "Along Wall", "color": "orange", "alpha": 1}
        
        # Assign properties for 'around_corner_arc' regions
        for i in range(4):
            properties[f"around_corner_{i}"] = {"label": "Around Corner Arc", "color": "lime", "alpha": 0.5}

        return properties


def visualize_building_terrain(
    ax,
    building_center: np.ndarray,  # (2,) numpy array
    building_width: float,
    building_height: float,
    building_theta: float,         # radians
    side_length: float,
    terrain_config: int = 1,
    resolution: int = 200,
) -> list:
    """
    Draw terrain zones as a raster background behind the scene.

    Terrain IDs (match TERRAIN_NAMES in base.py): 0=road, 1=sidewalk, 2=grass

    Config 1: perimeter sidewalk (0–0.15 from surface), rest road, inside grass
    Config 2: perimeter sidewalk (0–0.15), middle road, outer border sidewalk (<0.2 from boundary)
    Config 3: grass halo (0–0.10), sidewalk ring (0.10–0.25), rest road, inside grass

    Returns a list of matplotlib Patch objects for use in a legend.
    """
    from matplotlib.patches import Patch

    xs = np.linspace(0, side_length, resolution)
    ys = np.linspace(0, side_length, resolution)
    XX, YY = np.meshgrid(xs, ys)

    # Rotate grid into building-local frame
    cos_t = np.cos(-building_theta)
    sin_t = np.sin(-building_theta)
    rel_x = XX - building_center[0]
    rel_y = YY - building_center[1]
    local_x = cos_t * rel_x - sin_t * rel_y
    local_y = sin_t * rel_x + cos_t * rel_y

    half_w = building_width / 2.0
    half_h = building_height / 2.0

    # Distance from building surface (0 at surface, positive outside)
    dx = np.maximum(np.abs(local_x) - half_w, 0.0)
    dy = np.maximum(np.abs(local_y) - half_h, 0.0)
    dist_to_surface = np.sqrt(dx**2 + dy**2)

    inside_building = (np.abs(local_x) <= half_w) & (np.abs(local_y) <= half_h)

    # Distance from env boundary
    dist_to_boundary = np.minimum(
        np.minimum(XX, side_length - XX),
        np.minimum(YY, side_length - YY)
    )
    near_boundary = dist_to_boundary < 0.2

    if terrain_config == 1:
        terrain_grid = np.where(inside_building, 2,
                       np.where(dist_to_surface <= 0.15, 1, 0))
    elif terrain_config == 2:
        terrain_grid = np.where(inside_building, 2,
                       np.where(dist_to_surface <= 0.15, 1,
                       np.where(near_boundary, 1, 0)))
    else:  # config 3
        terrain_grid = np.where(inside_building, 2,
                       np.where(dist_to_surface <= 0.10, 2,
                       np.where(dist_to_surface <= 0.25, 1, 0)))

    # RGBA colours: Road=gray, Sidewalk=sandy tan, Grass=muted green
    color_map = {
        0: np.array([0.55, 0.55, 0.55, 0.45]),  # Road
        1: np.array([0.88, 0.78, 0.50, 0.50]),  # Sidewalk
        2: np.array([0.45, 0.72, 0.35, 0.35]),  # Grass
    }
    rgba = np.zeros((*terrain_grid.shape, 4), dtype=np.float32)
    for tid, col in color_map.items():
        rgba[terrain_grid == tid] = col

    ax.imshow(rgba, extent=[0, side_length, 0, side_length], origin='lower', zorder=0)

    return [
        Patch(facecolor=color_map[0][:3], label="Road"),
        Patch(facecolor=color_map[1][:3], label="Sidewalk"),
        Patch(facecolor=color_map[2][:3], label="Grass"),
    ]