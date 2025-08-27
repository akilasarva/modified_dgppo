import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib
import functools as ft
import jax.tree_util as jtu
from functools import partial
import jax.debug as jd

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union, Tuple, Dict, Any

class BehaviorAssociator:
    """Base class for associating an agent's position with a behavior region."""

    def __init__(self, bridges: List, buildings: List, obstacles: List, all_region_names: List[str]):
        self.bridges = bridges
        self.buildings = buildings
        self.obstacles = obstacles
        self.all_region_names = all_region_names  # Store the master list

        self.behavior_regions = self._define_behavior_regions()

        self._initialize_full_region_maps()
        
        self._prepare_jax_data()

    def _initialize_region_maps(self):
        self.region_name_to_id = {name: i for i, name in enumerate(self.behavior_regions.keys())}
        self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
        self.sorted_region_names = sorted(self.region_name_to_id, key=self.region_name_to_id.get)

    def _initialize_full_region_maps(self):
        self.region_name_to_id = {name: i for i, name in enumerate(self.all_region_names)}
        self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
        self.sorted_region_names = self.all_region_names # The list is already sorted by index

    def _define_behavior_regions(self):
        """This method must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _define_behavior_regions()")

    def _prepare_jax_data(self):
        """Prepares padded JAX arrays for rectangles and circles for get_current_behavior."""
        temp_boxes = []
        temp_centroids = []
        temp_labels = []
        temp_colors = []

        for name in self.sorted_region_names:
            coords = self.behavior_regions.get(name)
            if coords is None:
                continue

            # Rectangle: store bounding box [min_x, min_y, max_x, max_y]
            if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
                min_xy = jnp.min(coords, axis=0)
                max_xy = jnp.max(coords, axis=0)
                flat_box = jnp.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=jnp.float32)
                temp_boxes.append(flat_box)
                temp_centroids.append(jnp.mean(coords, axis=0))
                temp_labels.append(name)
                temp_colors.append("blue")

            # Circle
            elif isinstance(coords, tuple) and coords[0] == "circle":
                center_x, center_y, radius = coords[1]
                flat_circle = jnp.array([center_x, center_y, radius], dtype=jnp.float32)
                temp_boxes.append(flat_circle)  # use same array for scan
                temp_centroids.append(jnp.array([center_x, center_y], dtype=jnp.float32))
                temp_labels.append(name)
                temp_colors.append("red")

        self.all_region_boxes = jnp.stack(temp_boxes, axis=0)
        self.all_region_centroids_jax_array = jnp.stack(temp_centroids, axis=0)
        self.region_labels = temp_labels
        self.region_colors = temp_colors

    @partial(jax.jit, static_argnums=(0,))
    def get_current_behavior(self, robot_position, debug=False):
        robot_position_jnp = jnp.asarray(robot_position, dtype=jnp.float32)
        x, y = robot_position_jnp

        region_ids_list = []
        region_types_list = []
        region_data_list = []

        for name in self.sorted_region_names:
            region_id = self.region_name_to_id[name]
            region_data = self.behavior_regions[name]

            # Rectangle
            if isinstance(region_data, jnp.ndarray) and region_data.ndim == 2:
                min_xy = jnp.min(region_data, axis=0)
                max_xy = jnp.max(region_data, axis=0)
                flat_data = jnp.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=jnp.float32)
                region_ids_list.append(region_id)
                region_types_list.append(0)
                region_data_list.append(flat_data)

            # Semicircle (approach / exit)
            elif isinstance(region_data, jnp.ndarray) and region_data.ndim == 1 and region_data.shape[0] == 4:
                # flat_data = [cx, cy, r, orientation_angle]
                region_ids_list.append(region_id)
                region_types_list.append(1)
                region_data_list.append(region_data)

        all_region_ids = jnp.array(region_ids_list, dtype=jnp.int32)
        all_region_types = jnp.array(region_types_list, dtype=jnp.int32)
        all_region_data = jnp.stack(region_data_list, axis=0)

        # --- JAX scan to find the current region ---
        def scan_fn(current_behavior_id, region_info):
            region_id, region_type, flat_data = region_info
            is_rect = region_type == 0
            is_semicircle = region_type == 1  # update region_types_list accordingly
            print()

            # Rectangle check
            def rect_check(_):
                min_x, min_y, max_x, max_y = flat_data
                return jnp.all(jnp.array([x >= min_x, x <= max_x, y >= min_y, y <= max_y]))

            # Semicircle check
            def semicircle_check(_):
                cx, cy, r, angle = flat_data
                vec = robot_position_jnp - jnp.array([cx, cy])
                dist_ok = jnp.linalg.norm(vec) <= r
                orientation_vec = jnp.array([jnp.cos(angle), jnp.sin(angle)])
                in_front = jnp.dot(vec, orientation_vec) >= 0
                return dist_ok & in_front

            in_rect = jax.lax.cond(is_rect, rect_check, lambda _: False, operand=None)
            in_semicircle = jax.lax.cond(is_semicircle, semicircle_check, lambda _: False, operand=None)


            new_behavior_id = jnp.where(jnp.logical_or(in_rect, in_semicircle),
                                        region_id, current_behavior_id)
            # jd.print("Checking region ID: {}, In rect: {}, In semi: {}", region_id, in_rect, in_semicircle)

            return new_behavior_id, None

        final_behavior_id, _ = jax.lax.scan(
            scan_fn,
            -1,  # default: not in any region
            (all_region_ids, all_region_types, all_region_data)
        )

        return final_behavior_id


    @partial(jax.jit, static_argnums=(0,))
    def _is_point_in_polygon(self, point, polygon_padded, num_actual_vertices):
        x, y = point
        is_degenerate = num_actual_vertices < 3
        if_degenerate_result = False

        vertex_mask = jnp.arange(self.MAX_POLYGON_VERTICES) < num_actual_vertices

        first_vertex = polygon_padded[0]
        closed_polygon = jnp.concatenate([polygon_padded, first_vertex[jnp.newaxis, :]], axis=0)
        

        p1 = closed_polygon[:-1]
        p2 = closed_polygon[1:]
        edges = jnp.stack([p1, p2], axis=1)  # shape (MAX_POLYGON_VERTICES, 2, 2)

        def edge_fn(carry, i):
            inside = carry
            def compute_fn(_):
                p1x, p1y = edges[i, 0]
                p2x, p2y = edges[i, 1]

                cond1 = y > jnp.min(jnp.array([p1y, p2y]))
                cond2 = y <= jnp.max(jnp.array([p1y, p2y]))
                cond3 = x <= jnp.max(jnp.array([p1x, p2x]))

                denominator = p2y - p1y
                xinters = jnp.where(denominator != 0.0,
                                    (y - p1y) * (p2x - p1x) / denominator + p1x,
                                    jnp.inf * jnp.sign(x))

                toggle = jnp.logical_and(cond1, jnp.logical_and(cond2, cond3 & (x <= xinters)))
                return jnp.logical_xor(inside, toggle)

            inside = jax.lax.cond(vertex_mask[i], compute_fn, lambda _: inside, operand=None)
            return inside, None

        initial_inside = False
        final_inside, _ = jax.lax.scan(edge_fn, initial_inside, jnp.arange(self.MAX_POLYGON_VERTICES))
        return jnp.where(is_degenerate, if_degenerate_result, final_inside)


    def visualize_behavior_regions(self, ax: Axes):
        added_labels = set()
        region_properties = self._get_region_visualization_properties()

        for i, name in enumerate(self.sorted_region_names):
            region = self.behavior_regions[name]
            properties = region_properties.get(name, {})
            color = properties.get('color', 'blue')
            label_for_legend = properties.get('label', name)
            alpha = properties.get('alpha', 0.5)

            if isinstance(region, jnp.ndarray) and region.ndim == 2:
                region_np = np.asarray(region)
                
                if name == "open_space":
                    min_x = np.min(region_np[:, 0])
                    max_x = np.max(region_np[:, 0])
                    min_y = np.min(region_np[:, 1])
                    max_y = np.max(region_np[:, 1])
                    patch = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                            alpha=alpha, color=color, fill=True)
                else:
                    patch = plt.Polygon(region_np, closed=True, alpha=alpha, color=color, fill=True)
                
                if label_for_legend not in added_labels:
                    patch.set_label(label_for_legend)
                    added_labels.add(label_for_legend)
                ax.add_patch(patch)

            elif isinstance(region, tuple) and region[0] == "circle":
                center_x, center_y, radius = region[1]
                circle_patch = plt.Circle((center_x, center_y), radius,
                                            alpha=alpha, color=color, fill=False, linestyle='--')
                if label_for_legend not in added_labels:
                    circle_patch.set_label(label_for_legend)
                    added_labels.add(label_for_legend)
                ax.add_patch(circle_patch)
            
            if name != "open_space":
                 centroid = np.asarray(self.all_region_centroids_jax_array[i])
                 ax.text(centroid[0], centroid[1], 'o',
                         fontsize=8, ha='center', va='center', color='black')

        ax.legend(loc='upper right', fontsize='small')

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping region names to visualization properties.
        This method MUST be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_region_visualization_properties()")


    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid

    
    

# # import jax
# # import jax.numpy as jnp
# # from functools import partial

# # class BehaviorAssociator:
# #     def __init__(self, bridges=[], buildings=[], obstacles=[], all_region_names=[]):
# #         """
# #         Base class for behavior regions. Stores polygons and centroids in a JAX-friendly way.
# #         bridges, buildings, obstacles: list of polygon definitions
# #         all_region_names: list of names of all regions in canonical order
# #         """
# #         self.bridges = bridges
# #         self.buildings = buildings
# #         self.obstacles = obstacles
# #         self.all_region_names = all_region_names

# #         # --- define polygons ---
# #         self.all_region_polygons_padded = None  # (num_regions, max_vertices, 2)
# #         self.num_actual_vertices = None         # (num_regions,)
# #         self.all_region_ids = None              # priority-ordered indices
# #         self.all_region_centroids_jax_array = None

# #         self._initialize_regions()

# #     def _initialize_regions(self):
# #         """
# #         Convert polygons into padded arrays and compute centroids as JAX arrays.
# #         Also applies priority ordering (highest priority first).
# #         """
# #         regions = self._define_behavior_regions()  # dict: name -> (N,2) jnp array
# #         num_regions = len(regions)
# #         max_vertices = max([v.shape[0] for v in regions.values()])
# #         self.max_vertices = max_vertices

# #         padded_polygons = []
# #         num_vertices_list = []
# #         centroids = []
# #         region_names_in_priority = list(reversed(self.all_region_names))  # highest priority first

# #         for name in region_names_in_priority:
# #             poly = regions[name]
# #             n = poly.shape[0]
# #             padded = jnp.pad(poly, ((0, max_vertices - n), (0, 0)), constant_values=0.0)
# #             padded_polygons.append(padded)
# #             num_vertices_list.append(n)
# #             centroids.append(jnp.mean(poly, axis=0))

# #         # Stack into JAX arrays
# #         self.all_region_polygons_padded = jnp.stack(padded_polygons)       # (num_regions, max_vertices, 2)
# #         self.num_actual_vertices = jnp.array(num_vertices_list)            # (num_regions,)
# #         self.all_region_centroids_jax_array = jnp.stack(centroids)         # (num_regions, 2)
# #         self.all_region_ids = jnp.arange(num_regions)      

# #     def _define_behavior_regions(self):
# #         """
# #         Subclasses must implement this. Returns dict of region_name -> (N,2) JAX arrays
# #         """
# #         raise NotImplementedError

# #     def get_polygon_mask(self, region_id: int) -> jnp.ndarray:
# #         """
# #         Returns a boolean mask of shape (max_vertices,) 
# #         indicating which vertices are real (True) vs padded (False)
# #         """
# #         num_vertices = self.num_actual_vertices[region_id]
# #         mask = jnp.arange(self.max_vertices) < num_vertices
# #         return mask

# #     def _is_point_in_polygon(self, point: jnp.ndarray, region_id: int) -> jnp.ndarray:
# #         """
# #         Checks if a 2D point is inside a polygon, using only the real vertices.
# #         Uses dynamic_slice instead of boolean mask for JIT compatibility.
# #         """
# #         polygon_padded = self.all_region_polygons_padded[region_id]  # (max_vertices, 2)
# #         n = self.num_actual_vertices[region_id]                       # scalar (dynamic)
        
# #         # Make a mask of shape (max_vertices,)
# #         mask = jnp.arange(polygon_padded.shape[0]) < n               # True for real vertices
        
# #         polygon_x = polygon_padded[:, 0] * mask                       # padded vertices become 0
# #         polygon_y = polygon_padded[:, 1] * mask
# #         # Ray-casting algorithm for point-in-polygon
# #         x, y = point

# #         def ray_cross(i, val):
# #             j = (i + 1) % polygon_padded.shape[0]
# #             cond = ((polygon_y[i] > y) != (polygon_y[j] > y)) & \
# #                 (x < (polygon_x[j] - polygon_x[i]) * (y - polygon_y[i]) / (polygon_y[j] - polygon_y[i] + 1e-12) + polygon_x[i])
# #             return val ^ cond

# #         inside = 0
# #         inside = jax.lax.fori_loop(0, polygon_padded.shape[0], ray_cross, inside)
# #         return inside


# #     @partial(jax.jit, static_argnums=(0,))
# #     def get_current_behavior(self, robot_position):
# #         """
# #         Returns the region index for robot_position.
# #         Vectorized over all regions to be JIT-compatible.
# #         """
# #         def check_region(region_id):
# #             return self._is_point_in_polygon(robot_position, region_id)

# #         is_inside = jax.vmap(check_region)(self.all_region_ids)  # shape: (num_regions,)

# #         # Pick first True (highest-priority)
# #         def first_true_index(mask):
# #             # Use jnp.where, returns array of indices
# #             indices = jnp.where(mask, size=1, fill_value=-1)[0]
# #             return indices

# #         first_idx = first_true_index(is_inside)
# #         return first_idx


# #     @partial(jax.jit, static_argnums=(0,))
# #     def get_region_centroid(self, region_id):
# #         """
# #         Returns centroid of a region by index
# #         """
# #         return self.all_region_centroids_jax_array[region_id]
    
# #     def visualize_behavior_regions(self, ax: Axes):
# #         added_labels = set()
        
# #         # We need to get the color and label for each region from the subclass.
# #         # This is the key change. We'll add a new method to handle this.
# #         region_properties = self._get_region_visualization_properties()

# #         for i, name in enumerate(self.sorted_region_names):
# #             region = self.behavior_regions[name]
# #             properties = region_properties.get(name, {})
# #             color = properties.get('color', 'blue')
# #             label_for_legend = properties.get('label', name)
# #             alpha = properties.get('alpha', 0.5)

# #             if isinstance(region, jnp.ndarray) and region.ndim == 2:
# #                 region_np = np.asarray(region)
                
# #                 if name == "open_space":
# #                     min_x = np.min(region_np[:, 0])
# #                     max_x = np.max(region_np[:, 0])
# #                     min_y = np.min(region_np[:, 1])
# #                     max_y = np.max(region_np[:, 1])
# #                     patch = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
# #                                             alpha=alpha, color=color, fill=True)
# #                 else:
# #                     patch = plt.Polygon(region_np, closed=True, alpha=alpha, color=color, fill=True)
                
# #                 if label_for_legend not in added_labels:
# #                     patch.set_label(label_for_legend)
# #                     added_labels.add(label_for_legend)
# #                 ax.add_patch(patch)

# #             elif isinstance(region, tuple) and region[0] == "circle":
# #                 center_x, center_y, radius = region[1]
# #                 circle_patch = plt.Circle((center_x, center_y), radius,
# #                                             alpha=alpha, color=color, fill=False, linestyle='--')
# #                 if label_for_legend not in added_labels:
# #                     circle_patch.set_label(label_for_legend)
# #                     added_labels.add(label_for_legend)
# #                 ax.add_patch(circle_patch)
            
# #             if name != "open_space":
# #                  centroid = np.asarray(self.all_region_centroids_jax_array[i])
# #                  ax.text(centroid[0], centroid[1], 'o',
# #                          fontsize=8, ha='center', va='center', color='black')

# #         ax.legend(loc='upper right', fontsize='small')

# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import numpy as np
# import jax
# import pathlib
# import functools as ft
# import jax.tree_util as jtu
# from functools import partial
# import jax.debug as jd

# from colour import hsl2hex
# from matplotlib.animation import FuncAnimation
# from matplotlib.collections import LineCollection, PatchCollection
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.pyplot import Axes
# from matplotlib.patches import Polygon as MplPolygon
# from matplotlib.patches import Rectangle as MplRectangle
# from mpl_toolkits.mplot3d import proj3d, Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from typing import List, Optional, Union, Tuple, Dict, Any

# class BehaviorAssociator:
#     """Base class for associating an agent's position with a behavior region."""

#     def __init__(self, bridges: List, buildings: List, obstacles: List, all_region_names: List[str]):
#         self.bridges = bridges
#         self.buildings = buildings
#         self.obstacles = obstacles
#         self.all_region_names = all_region_names  # Store the master list

#         # This method MUST be overridden by subclasses
#         self.behavior_regions = self._define_behavior_regions()
        
#         # Now, initialize the IDs and names based on the defined regions
#         # Call a new method to build maps from the complete list
#         self._initialize_full_region_maps()
        
#         # Prepare JAX arrays for faster computation
#         self._prepare_jax_data()

#     def _initialize_region_maps(self):
#         # A bit more robust initialization
#         self.region_name_to_id = {name: i for i, name in enumerate(self.behavior_regions.keys())}
#         self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
#         self.sorted_region_names = sorted(self.region_name_to_id, key=self.region_name_to_id.get)

#     def _initialize_full_region_maps(self):
#         # A bit more robust initialization
#         self.region_name_to_id = {name: i for i, name in enumerate(self.all_region_names)}
#         self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
#         self.sorted_region_names = self.all_region_names # The list is already sorted by index

#     def _define_behavior_regions(self):
#         """This method must be overridden by subclasses."""
#         raise NotImplementedError("Subclasses must implement _define_behavior_regions()")

#     def _prepare_jax_data(self):
#         """Prepares the JAX arrays for `get_current_behavior`."""
#         temp_polygons = []
#         temp_centroids = []
#         temp_labels = []
#         temp_colors = []

#         self.MAX_POLYGON_VERTICES = 4
#         for name, coords in self.behavior_regions.items():
#             if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
#                 self.MAX_POLYGON_VERTICES = max(self.MAX_POLYGON_VERTICES, coords.shape[0])

#         #priority_order = ["open_space", "along_wall", "past_building", "around_corner"]
#         priority_order = ["open_space", "approach_bridge", "on_bridge", "exit_bridge"]
    
#         reordered_names = []
#         for prio_name in priority_order:
#             for full_name in self.all_region_names:
#                 if full_name.startswith(prio_name):
#                     reordered_names.append(full_name)
        
#         self.sorted_region_names = reordered_names

#         for name in self.sorted_region_names:            
#             coords = self.behavior_regions.get(name)

#             if coords is not None:
#                 if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
#                     temp_polygons.append(coords)
#                     temp_centroids.append(jnp.mean(coords, axis=0))
#                     temp_labels.append(name)
#                     # NOTE: Colors will be determined in the `visualize` method
#                     temp_colors.append('blue')
#                 elif isinstance(coords, tuple) and coords[0] == "circle":
#                     center_x, center_y, radius = coords[1]
#                     temp_polygons.append(jnp.array([[center_x, center_y]], dtype=jnp.float32))
#                     temp_centroids.append(jnp.array([center_x, center_y], dtype=jnp.float32))
#                     temp_labels.append(name)
#                     temp_colors.append('red')

#         padded_polygons_for_stack = []
#         for p in temp_polygons:
#             num_actual_vertices = p.shape[0]
#             padded_p = jnp.pad(p, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
#             padded_polygons_for_stack.append(padded_p)

#         self.all_region_polygons = jnp.stack(padded_polygons_for_stack, axis=0)
#         self.all_region_centroids_jax_array = jnp.stack(temp_centroids, axis=0)
#         self.region_labels = temp_labels
#         self.region_colors = temp_colors
        

#     @partial(jax.jit, static_argnums=(0,))
#     def get_current_behavior(self, robot_position):

#         robot_position_jnp = jnp.asarray(robot_position, dtype=jnp.float32)

#         MAX_DATA_DIM = max(self.MAX_POLYGON_VERTICES * 2 + 1, 3 + 1)

#         region_ids_list = []
#         region_types_list = []
#         region_data_list = []

#         region_ids_list.append(-1)
#         region_types_list.append(-1)
#         region_data_list.append(jnp.zeros(MAX_DATA_DIM, dtype=jnp.float32))

#         for name in self.sorted_region_names:            
#             region_id = self.region_name_to_id.get(name)
#             region_data = self.behavior_regions[name]

#             if isinstance(region_data, jnp.ndarray) and region_data.ndim == 2:
#                 num_actual_vertices = region_data.shape[0]
#                 padded_polygon = jnp.pad(region_data, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
#                 flat_data = jnp.concatenate((padded_polygon.flatten(), jnp.array([num_actual_vertices], dtype=jnp.float32)))
#                 region_ids_list.append(region_id)
#                 region_types_list.append(0)
#                 region_data_list.append(flat_data)
#             elif isinstance(region_data, tuple) and region_data[0] == "circle":
#                 center_x, center_y, radius = region_data[1]
#                 dummy_num_vertices = 0.0
#                 padded_circle_data_and_num = jnp.concatenate((circle_data_flat, jnp.array([dummy_num_vertices], dtype=jnp.float32)))
#                 padded_circle_data_and_num = jnp.pad(padded_circle_data_and_num, (0, MAX_DATA_DIM - padded_circle_data_and_num.shape[0]), mode='constant', constant_values=0.0)

#                 region_ids_list.append(region_id)
#                 region_types_list.append(1)
#                 region_data_list.append(padded_circle_data_and_num)

#         all_region_ids = jnp.array(region_ids_list, dtype=jnp.int32)
#         all_region_types = jnp.array(region_types_list, dtype=jnp.int32)
#         all_region_padded_data = jnp.stack(region_data_list, axis=0)

#         @jax.jit
#         def _get_behavior_jit(position, region_ids, region_types, region_padded_data):
#             initial_carry = (-1, False)

#             def scan_fn(carry, region_info):
#                 current_behavior_id, is_found_flag = carry
#                 region_id, region_type, flat_padded_data_and_num_vertices = region_info

#                 is_polygon_type = (region_type == 0)
#                 is_circle_type = (region_type == 1)

#                 num_actual_vertices = jnp.floor(flat_padded_data_and_num_vertices[-1]).astype(jnp.int32)
#                 polygon_coords_flat = flat_padded_data_and_num_vertices[:-1]
#                 polygon_coords = jnp.reshape(polygon_coords_flat, (self.MAX_POLYGON_VERTICES, 2))

#                 is_in_poly = self._is_point_in_polygon(position, polygon_coords, num_actual_vertices)
#                 is_current_in_polygon = jnp.logical_and(is_polygon_type, is_in_poly)

#                 center_x, center_y, radius = flat_padded_data_and_num_vertices[0], flat_padded_data_and_num_vertices[1], flat_padded_data_and_num_vertices[2]
#                 circle_distance = jnp.linalg.norm(position - jnp.array([center_x, center_y]))
#                 is_in_circle = circle_distance <= radius
#                 is_current_in_circle = jnp.logical_and(is_circle_type, is_in_circle)

#                 new_behavior_id = jnp.where(is_current_in_polygon, region_id, current_behavior_id)
#                 new_behavior_id = jnp.where(is_current_in_circle, region_id, new_behavior_id)
                
#                 new_is_found_flag = jnp.logical_or(is_found_flag, jnp.logical_or(is_current_in_polygon, is_current_in_circle))

#                 return (new_behavior_id, new_is_found_flag), None

#             final_carry, _ = jax.lax.scan(scan_fn, initial_carry, (all_region_ids, all_region_types, all_region_padded_data))
#             final_behavior_id, _ = final_carry

#             return final_behavior_id

#         return _get_behavior_jit(robot_position_jnp, all_region_ids, all_region_types, all_region_padded_data)

#     @partial(jax.jit, static_argnums=(0,))
#     def _is_point_in_polygon(self, point, polygon_padded, num_actual_vertices):
#         x, y = point

#         is_degenerate_polygon = (num_actual_vertices < 3)
#         if_degenerate_result = jnp.array(False)

#         first_vertex_index = jnp.array([0])
#         first_actual_vertex = jax.lax.dynamic_slice(polygon_padded, (0, 0), (1, 2)).squeeze()

#         closed_polygon_padded = jnp.concatenate((polygon_padded, first_actual_vertex[jnp.newaxis, :]), axis=0)

#         p1_coords = closed_polygon_padded[:-1]
#         p2_coords = closed_polygon_padded[1:]
#         edges = jnp.stack((p1_coords, p2_coords), axis=1)

#         is_vertex = jnp.any(jnp.where(jnp.arange(self.MAX_POLYGON_VERTICES)[:, None] < num_actual_vertices,
#                                         jnp.all(jnp.isclose(point, polygon_padded), axis=1),
#                                         False))

#         initial_inside = jnp.where(is_vertex, jnp.array(True), jnp.array(False))

#         def scan_body(current_inside_state, i):
#             is_actual_edge = (i < num_actual_vertices)

#             p1x, p1y = edges[i, 0]
#             p2x, p2y = edges[i, 1]

#             def true_fn():
#                 cond1 = y > jnp.min(jnp.array([p1y, p2y]))
#                 cond2 = y <= jnp.max(jnp.array([p1y, p2y]))
#                 cond3 = x <= jnp.max(jnp.array([p1x, p2x]))

#                 denominator = p2y - p1y
#                 xinters = jnp.where(
#                     denominator != 0.0,
#                     (y - p1y) * (p2x - p1x) / denominator + p1x,
#                     jnp.inf * jnp.sign(x)
#                 )

#                 should_toggle_base = jnp.logical_and(
#                     cond1,
#                     jnp.logical_and(
#                         cond2,
#                         jnp.logical_or(
#                             x <= xinters,
#                             jnp.isclose(x, xinters)
#                         )
#                     )
#                 )

#                 line_vec = edges[i, 1] - edges[i, 0]
#                 point_vec = point - edges[i, 0]
#                 cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
#                 is_collinear = jnp.isclose(cross_product, 0.0)
#                 dot_product = jnp.dot(point_vec, line_vec)
#                 squared_length = jnp.dot(line_vec, line_vec)
#                 is_within_segment = jnp.where(squared_length > 1e-6, jnp.logical_and(dot_product >= 0, dot_product <= squared_length), False)
#                 is_on_segment = jnp.logical_and(is_collinear, is_within_segment)

#                 is_horizontal = jnp.isclose(p1y, p2y)
#                 should_toggle = jnp.where(is_horizontal, jnp.array(False), should_toggle_base)

#                 new_inside_state = jnp.where(should_toggle, jnp.logical_not(current_inside_state), current_inside_state)
#                 final_inside_state = jnp.where(is_on_segment, jnp.array(True), new_inside_state)
#                 return final_inside_state

#             def false_fn():
#                 return current_inside_state

#             updated_inside = jax.lax.cond(is_actual_edge, true_fn, false_fn)

#             return updated_inside, None

#         final_inside_if_valid, _ = jax.lax.scan(scan_body, initial_inside, jnp.arange(self.MAX_POLYGON_VERTICES + 1))


#         return jnp.where(is_degenerate_polygon, if_degenerate_result, final_inside_if_valid)

#     def visualize_behavior_regions(self, ax: Axes):
#         added_labels = set()
        
#         # We need to get the color and label for each region from the subclass.
#         # This is the key change. We'll add a new method to handle this.
#         region_properties = self._get_region_visualization_properties()

#         for i, name in enumerate(self.sorted_region_names):
#             region = self.behavior_regions[name]
#             properties = region_properties.get(name, {})
#             color = properties.get('color', 'blue')
#             label_for_legend = properties.get('label', name)
#             alpha = properties.get('alpha', 0.5)

#             if isinstance(region, jnp.ndarray) and region.ndim == 2:
#                 region_np = np.asarray(region)
                
#                 if name == "open_space":
#                     min_x = np.min(region_np[:, 0])
#                     max_x = np.max(region_np[:, 0])
#                     min_y = np.min(region_np[:, 1])
#                     max_y = np.max(region_np[:, 1])
#                     patch = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
#                                             alpha=alpha, color=color, fill=True)
#                 else:
#                     patch = plt.Polygon(region_np, closed=True, alpha=alpha, color=color, fill=True)
                
#                 if label_for_legend not in added_labels:
#                     patch.set_label(label_for_legend)
#                     added_labels.add(label_for_legend)
#                 ax.add_patch(patch)

#             elif isinstance(region, tuple) and region[0] == "circle":
#                 center_x, center_y, radius = region[1]
#                 circle_patch = plt.Circle((center_x, center_y), radius,
#                                             alpha=alpha, color=color, fill=False, linestyle='--')
#                 if label_for_legend not in added_labels:
#                     circle_patch.set_label(label_for_legend)
#                     added_labels.add(label_for_legend)
#                 ax.add_patch(circle_patch)
            
#             if name != "open_space":
#                  centroid = np.asarray(self.all_region_centroids_jax_array[i])
#                  ax.text(centroid[0], centroid[1], 'o',
#                          fontsize=8, ha='center', va='center', color='black')

#         ax.legend(loc='upper right', fontsize='small')

#     def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Returns a dictionary mapping region names to visualization properties.
#         This method MUST be overridden by subclasses.
#         """
#         raise NotImplementedError("Subclasses must implement _get_region_visualization_properties()")


#     @partial(jax.jit, static_argnums=(0,))
#     def get_region_centroid(self, region_id: jnp.ndarray):
#         centroid = self.all_region_centroids_jax_array[region_id]
#         return centroid
