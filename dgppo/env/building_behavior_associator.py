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