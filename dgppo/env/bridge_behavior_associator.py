import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon
from typing import List, Dict, Any
from functools import partial
import numpy as np 
import jax

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

        regions["open_space"] = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)
        individual_bridge_corners = []

        for i, (cx, cy, l, w, a) in enumerate(self.bridges):
            rad = jnp.radians(a)

            half_l = l / 2.0
            half_w = w / 2.0

            # local_corners order:
            #   [0] = (-half_l, -half_w)  entry end, -perp side
            #   [1] = (+half_l, -half_w)  exit  end, -perp side
            #   [2] = (+half_l, +half_w)  exit  end, +perp side
            #   [3] = (-half_l, +half_w)  entry end, +perp side
            local_corners = jnp.array([
                [-half_l, -half_w],
                [ half_l, -half_w],
                [ half_l,  half_w],
                [-half_l,  half_w],
            ], dtype=jnp.float32)

            cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
            rotation_matrix = jnp.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            rotated_corners_local = jnp.einsum('ij,kj->ki', rotation_matrix, local_corners)
            corners_obstacle = rotated_corners_local + jnp.array([cx, cy])
            individual_bridge_corners.append(corners_obstacle)

        num_points = 30

        if len(self.bridges) == 4:
            # ── Bent bridge: 4 walls (seg1-left, seg1-right, seg2-left, seg2-right) ──
            #
            # Wall layout produced by create_bent_bridge:
            #   bridges[0] = seg1 left  wall (displaced in +perp direction from seg1 center)
            #   bridges[1] = seg1 right wall (displaced in -perp direction from seg1 center)
            #   bridges[2] = seg2 left  wall
            #   bridges[3] = seg2 right wall
            #
            # For a wall at +perp offset (left wall):
            #   corners[3] = outer entry (+half_w in local = away from gap)
            #   corners[2] = outer exit
            #   corners[0] = inner entry (-half_w in local = toward gap)
            #   corners[1] = inner exit
            # For a wall at -perp offset (right wall):
            #   corners[0] = outer entry (-half_w in local = away from gap)
            #   corners[1] = outer exit
            #   corners[3] = inner entry (+half_w in local = toward gap)
            #   corners[2] = inner exit

            c_s1l = individual_bridge_corners[0]  # seg1 left
            c_s1r = individual_bridge_corners[1]  # seg1 right
            c_s2l = individual_bridge_corners[2]  # seg2 left
            c_s2r = individual_bridge_corners[3]  # seg2 right

            # Outer corners used to define the on_bridge_0 polygon perimeter
            entry_left          = c_s1l[3]   # outer entry of seg1 left wall
            entry_right         = c_s1r[0]   # outer entry of seg1 right wall
            junction_right_seg1 = c_s1r[1]   # outer exit  of seg1 right (at junction)
            junction_right_seg2 = c_s2r[0]   # outer entry of seg2 right (at junction)
            exit_right          = c_s2r[1]   # outer exit  of seg2 right wall
            exit_left           = c_s2l[2]   # outer exit  of seg2 left  wall
            junction_left_seg2  = c_s2l[3]   # outer entry of seg2 left  (at junction)
            junction_left_seg1  = c_s1l[2]   # outer exit  of seg1 left  (at junction)

            # on_bridge_0: octagonal perimeter encompassing both segments
            regions["on_bridge_0"] = jnp.array([
                entry_left, entry_right,
                junction_right_seg1, junction_right_seg2,
                exit_right, exit_left,
                junction_left_seg2, junction_left_seg1,
            ], dtype=jnp.float32)

            # approach_bridge_0: semicircle at the entry face of segment 1
            approach_center = (entry_left + entry_right) / 2.0
            approach_width  = jnp.linalg.norm(entry_left - entry_right)
            approach_radius = approach_width / 2.0
            rad_seg1 = jnp.radians(self.bridges[0][4])   # seg1 angle
            angles_approach = jnp.linspace(rad_seg1 + jnp.pi / 2, rad_seg1 + 3 * jnp.pi / 2, num_points)
            approach_arc = jnp.stack([
                approach_center[0] + approach_radius * jnp.cos(angles_approach),
                approach_center[1] + approach_radius * jnp.sin(angles_approach),
            ], axis=-1)
            regions["approach_bridge_0"] = jnp.concatenate([
                jnp.array([entry_right, entry_left]),
                approach_arc[::-1],
            ], axis=0)

            # exit_bridge_0: semicircle at the exit face of segment 2
            exit_center = (exit_left + exit_right) / 2.0
            exit_width  = jnp.linalg.norm(exit_left - exit_right)
            exit_radius = exit_width / 2.0
            rad_seg2 = jnp.radians(self.bridges[2][4])   # seg2 angle
            angles_exit = jnp.linspace(rad_seg2 - jnp.pi / 2, rad_seg2 + jnp.pi / 2, num_points)
            exit_arc = jnp.stack([
                exit_center[0] + exit_radius * jnp.cos(angles_exit),
                exit_center[1] + exit_radius * jnp.sin(angles_exit),
            ], axis=-1)
            regions["exit_bridge_0"] = jnp.concatenate([
                jnp.array([exit_right, exit_left]),
                exit_arc[::-1],
            ], axis=0)

        elif len(self.bridges) >= 2:
            # ── Straight bridge: original 2-wall logic ────────────────────────────
            corners_obs1 = individual_bridge_corners[0]
            corners_obs2 = individual_bridge_corners[1]

            on_bridge_p1 = corners_obs1[3]
            on_bridge_p2 = corners_obs1[2]
            on_bridge_p3 = corners_obs2[1]
            on_bridge_p4 = corners_obs2[0]

            regions["on_bridge_0"] = jnp.array([
                on_bridge_p1, on_bridge_p4, on_bridge_p3, on_bridge_p2
            ], dtype=jnp.float32)

            channel_entrance_p1 = on_bridge_p1
            channel_entrance_p2 = on_bridge_p4
            approach_center = (channel_entrance_p1 + channel_entrance_p2) / 2.0
            channel_width   = jnp.linalg.norm(on_bridge_p1 - on_bridge_p4)
            semi_r          = channel_width / 2.0
            rad_for_arcs    = jnp.radians(self.bridges[0][4])

            angles_approach = jnp.linspace(rad_for_arcs + jnp.pi / 2, rad_for_arcs + 3 * jnp.pi / 2, num_points)
            approach_arc    = jnp.array([
                approach_center[0] + semi_r * jnp.cos(angles_approach),
                approach_center[1] + semi_r * jnp.sin(angles_approach),
            ]).T
            regions["approach_bridge_0"] = jnp.concatenate([
                jnp.array([channel_entrance_p2, channel_entrance_p1]),
                approach_arc[::-1],
            ], axis=0)

            channel_exit_p1      = on_bridge_p2
            channel_exit_p2      = on_bridge_p3
            exit_center          = (channel_exit_p1 + channel_exit_p2) / 2.0
            angles_exit          = jnp.linspace(rad_for_arcs - jnp.pi / 2, rad_for_arcs + jnp.pi / 2, num_points)
            exit_arc             = jnp.array([
                exit_center[0] + semi_r * jnp.cos(angles_exit),
                exit_center[1] + semi_r * jnp.sin(angles_exit),
            ]).T
            regions["exit_bridge_0"] = jnp.concatenate([
                jnp.array([channel_exit_p2, channel_exit_p1]),
                exit_arc[::-1],
            ], axis=0)

        for key, value in regions.items():
            if isinstance(value, list):
                regions[key] = jnp.asarray(value, dtype=jnp.float32)

        return regions


    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping bridge region names to visualization properties.
        """
        return {
            "on_bridge_0": {"label": "On Bridge", "color": "skyblue"},
            "approach_bridge_0": {"label": "Approach Bridge", "color": "lightgreen"},
            "exit_bridge_0": {"label": "Exit Bridge", "color": "lightcoral"},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }


    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid


def visualize_terrain(
    ax,
    bridge_center: np.ndarray,    # (2,) numpy — bend/junction point
    bridge_gap_width: float,
    bridge_wall_thickness: float,
    bridge_theta: float,           # segment-1 angle, radians
    side_length: float,
    terrain_config: int = 2,       # 1 or 2
    resolution: int = 300,
    bridge_length: float = 1.0,   # total bridge length
    bridge_bend_angle: float = 0.0,  # segment-2 offset (0 = straight)
) -> list:
    """
    Draw terrain zones as a background raster image behind the scene.

    For a bent bridge, each pixel is assigned to whichever segment it is
    longitudinally closest to (with 10 % overlap at the junction), then the
    perpendicular distance to that segment's centerline determines the terrain.

    Terrain IDs:
      0 = Road     (gray)   — centre of gap            [config 2 only]
      1 = Grass    (green)  — open field away from bridge
      2 = Sidewalk (tan)    — beside the walls / within the full band [config 1]

    Returns a list of matplotlib Patch objects for a legend.
    """
    from matplotlib.patches import Patch

    xs = np.linspace(0, side_length, resolution)
    ys = np.linspace(0, side_length, resolution)
    XX, YY = np.meshgrid(xs, ys)

    half_gap        = bridge_gap_width / 2.0
    full_half       = half_gap + bridge_wall_thickness
    sidewalk_border = bridge_gap_width * 0.2
    road_half       = half_gap - sidewalk_border

    def _tid_from_perp(perp_arr):
        if terrain_config == 1:
            return np.where(perp_arr <= full_half, 2, 1)
        else:
            return np.where(perp_arr <= road_half, 0,
                   np.where(perp_arr <= half_gap,  2, 1))

    if abs(bridge_bend_angle) < 1e-6:
        # ── Straight bridge: infinite perpendicular strip ─────────────────
        dx   = XX - bridge_center[0]
        dy   = YY - bridge_center[1]
        cos_t, sin_t = np.cos(bridge_theta), np.sin(bridge_theta)
        perp = np.abs(-sin_t * dx + cos_t * dy)
        terrain_grid = _tid_from_perp(perp)
    else:
        # ── Bent bridge: two longitudinally-clamped segments ─────────────
        seg_quarter = bridge_length / 4.0

        # Segment 1
        cos1, sin1 = np.cos(bridge_theta), np.sin(bridge_theta)
        seg1_cx = bridge_center[0] - seg_quarter * cos1
        seg1_cy = bridge_center[1] - seg_quarter * sin1
        dxs1 = XX - seg1_cx;  dys1 = YY - seg1_cy
        perp1 = np.abs(-sin1 * dxs1 + cos1 * dys1)

        # Segment 2
        theta2 = bridge_theta + bridge_bend_angle
        cos2, sin2 = np.cos(theta2), np.sin(theta2)
        seg2_cx = bridge_center[0] + seg_quarter * cos2
        seg2_cy = bridge_center[1] + seg_quarter * sin2
        dxs2 = XX - seg2_cx;  dys2 = YY - seg2_cy
        perp2 = np.abs(-sin2 * dxs2 + cos2 * dys2)

        tid1 = _tid_from_perp(perp1)
        tid2 = _tid_from_perp(perp2)

        # Split at the kink (bridge_center) using the bisector of the two segment directions.
        # Entry side → seg1 terrain; exit side → seg2 terrain.
        bisector_x = cos1 + cos2
        bisector_y = sin1 + sin2
        dp_kink_x = XX - bridge_center[0]
        dp_kink_y = YY - bridge_center[1]
        on_seg2_side = (dp_kink_x * bisector_x + dp_kink_y * bisector_y) >= 0
        terrain_grid = np.where(on_seg2_side, tid2, tid1)

    # RGBA colour per terrain ID
    color_map = {
        0: np.array([0.55, 0.55, 0.55, 0.45]),
        1: np.array([0.45, 0.72, 0.35, 0.35]),
        2: np.array([0.88, 0.78, 0.50, 0.50]),
    }
    rgba = np.zeros((*terrain_grid.shape, 4), dtype=np.float32)
    for tid, col in color_map.items():
        rgba[terrain_grid == tid] = col

    ax.imshow(
        rgba,
        extent=[0, side_length, 0, side_length],
        origin='lower',
        zorder=0,
        aspect='auto',
        interpolation='nearest',
    )

    terrain_patches = [
        Patch(facecolor=color_map[1][:3], alpha=float(color_map[1][3]), label='Terrain: Grass'),
        Patch(facecolor=color_map[2][:3], alpha=float(color_map[2][3]), label='Terrain: Sidewalk'),
    ]
    if terrain_config == 2:
        terrain_patches.insert(
            0, Patch(facecolor=color_map[0][:3], alpha=float(color_map[0][3]), label='Terrain: Road')
        )

    print(f"[TERRAIN VIZ DEBUG] config={terrain_config}  "
          f"half_gap={half_gap:.3f}  full_half={full_half:.3f}  "
          f"bridge_theta_deg={np.degrees(bridge_theta):.1f}  "
          f"bend_deg={np.degrees(bridge_bend_angle):.1f}")
    unique, counts = np.unique(terrain_grid, return_counts=True)
    names = {0: 'Road', 1: 'Grass', 2: 'Sidewalk'}
    for u, c in zip(unique, counts):
        print(f"  {names.get(int(u), '?'):10s} id={u}  pixels={c}")

    return terrain_patches
