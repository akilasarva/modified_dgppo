import jax.numpy as jnp
import jax.debug as jdebug

from typing import Optional, Dict, Tuple
import functools as ft

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple, TARGET_TERRAIN_ID
from dgppo.utils.utils import jax_vmap

ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "approach_bridge_0",
        "on_bridge_0",
        "exit_bridge_0"
    ]

def _calculate_bearing_reward(agent_vel: jnp.ndarray, target_bearing: float) -> float:
    target_vector = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])

    norm_agent_vel = jnp.linalg.norm(agent_vel)
    safe_norm_agent_vel = jnp.where(norm_agent_vel > 1e-6, norm_agent_vel, 1e-6)

    cosine_sim = jnp.dot(agent_vel, target_vector) / safe_norm_agent_vel

    return cosine_sim

def _calculate_cluster_reward_per_agent(
    current_cluster_oh_episode_i: Array,
    start_cluster_oh_episode_i: Array,
    next_cluster_oh_episode_i: Array,
    has_been_awarded: jnp.ndarray,
    # Reward coefficients
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float

) -> float:

    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    start_cluster_id_episode_i = jnp.argmax(start_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)

    current_agent_reward = 0.0

    has_moved_into_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i) & \
                                 (current_cluster_id_episode_i != start_cluster_id_episode_i)

    # `give_bonus` is true only on the first step the agent moves into the next cluster
    give_bonus = has_moved_into_next_cluster & (~has_been_awarded)

    # Award the one-time bonus correctly
    bonus_reward = jnp.where(give_bonus, next_cluster_bonus, 0.0)

    # Update the flag to prevent future bonuses
    updated_has_been_awarded = jnp.logical_or(has_been_awarded, give_bonus)

    current_agent_reward += bonus_reward

    # This is the line that was missing. It awards a continuous bonus for staying in the target cluster.
    is_in_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i)
    current_agent_reward += jnp.where(is_in_next_cluster, stay_in_cluster_bonus, 0.0)

    is_incorrect_path_taken = (current_cluster_id_episode_i != start_cluster_id_episode_i) & \
                             (current_cluster_id_episode_i != next_cluster_id_episode_i)

    current_agent_reward += jnp.where(
        is_incorrect_path_taken,
        incorrect_cluster_penalty,
        0.0
    )

    return current_agent_reward, updated_has_been_awarded


def _terrain_reward_per_agent(
    boundary_hit_positions: jnp.ndarray,   # (n_rays, 2) — boundary hit positions
    boundary_terrain_ids: jnp.ndarray,     # (n_rays,)   — terrain on OTHER SIDE of each boundary
    agent_pos: jnp.ndarray,                # (2,)
    current_terrain_id: jnp.ndarray,       # scalar
    sense_range: float,
) -> float:
    """
    Mid-range terrain navigation reward using per-ray delta terrain value.

    For every boundary hit ray j:
        contribution_j = proximity_j × (V_other_j − V_current)

    where V = [Road: −1.0, Grass: −0.3, Sidewalk: +1.0] and
          proximity_j = max(0, 1 − dist_j / sense_range)   (0 when ray hits sensor boundary)

    This naturally rewards:
      − approaching sidewalk from road/grass      (+1.3 / +1.7 per ray)
      − staying far from road boundary on sidewalk  (proximity → 0 when far from edge)
      − approaching road from sidewalk or grass    (−2.0 / −0.7 per ray, penalised)
    Rays with no visible boundary (alpha=2.0 → dist > sense_range) contribute 0.
    """
    TERRAIN_VALUES = jnp.array([-1.0, -0.3, 1.0])  # Road=0, Grass=1, Sidewalk=2

    v_current = TERRAIN_VALUES[current_terrain_id]          # scalar
    v_other   = TERRAIN_VALUES[boundary_terrain_ids]        # (n_rays,)
    delta_v   = v_other - v_current                         # (n_rays,)

    dists     = jnp.linalg.norm(boundary_hit_positions - agent_pos[None, :], axis=-1)  # (n_rays,)
    proximity = jnp.where(dists < sense_range, 1.0 - dists / sense_range, 0.0)         # (n_rays,)

    on_target = (current_terrain_id == TARGET_TERRAIN_ID)
    return jnp.where(on_target, 0.0, (delta_v * proximity).mean())


def _calculate_preference_vector_reward(
    boundary_hit_positions: jnp.ndarray,  # (n_rays, 2) — boundary hit positions
    boundary_terrain_ids: jnp.ndarray,    # (n_rays,)   — other-side terrain IDs
    agent_pos: jnp.ndarray,               # (2,)
    agent_vel: jnp.ndarray,               # (2,)
    current_terrain_id: jnp.ndarray,      # scalar
    target_bearing: jnp.ndarray,          # scalar — global waypoint bearing (fallback)
    sense_range: float,
) -> float:
    """
    Preference vector reward for one agent.

    - Not in sidewalk + entry visible:  cos(vel, direction to nearest sidewalk boundary)
    - Not in sidewalk + no entry:       cos(vel, global bearing)   ← fallback
    - In sidewalk + both edges visible: cos(vel, direction to lateral centerline)
    - In sidewalk + edges not visible:  cos(vel, global bearing)   ← fallback
    """
    dists = jnp.linalg.norm(boundary_hit_positions - agent_pos[None, :], axis=-1)  # (n_rays,)
    in_target = (current_terrain_id == TARGET_TERRAIN_ID)

    vel_norm = jnp.linalg.norm(agent_vel)
    safe_vel_norm = jnp.where(vel_norm > 1e-6, vel_norm, 1e-6)

    # Global bearing fallback (always well-defined)
    bearing_vec = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])
    cosine_bearing = jnp.dot(agent_vel, bearing_vec) / safe_vel_norm

    # ── Not in sidewalk: point toward nearest entry ───────────────────────
    is_entry = (boundary_terrain_ids == TARGET_TERRAIN_ID)
    entry_masked = jnp.where(is_entry, dists, sense_range * 2.0)
    nearest_entry_idx = jnp.argmin(entry_masked)
    any_entry_visible = (entry_masked[nearest_entry_idx] < sense_range)

    entry_dir = boundary_hit_positions[nearest_entry_idx] - agent_pos
    entry_dir_norm = entry_dir / (jnp.linalg.norm(entry_dir) + 1e-6)
    cosine_entry = jnp.where(
        any_entry_visible,
        jnp.dot(agent_vel, entry_dir_norm) / safe_vel_norm,
        cosine_bearing,  # no sidewalk visible: follow global bearing
    )

    # ── In sidewalk: point toward lateral centerline ──────────────────────
    is_road_side  = (boundary_terrain_ids == 0)
    is_grass_side = (boundary_terrain_ids == 1)

    road_masked  = jnp.where(is_road_side,  dists, sense_range * 2.0)
    grass_masked = jnp.where(is_grass_side, dists, sense_range * 2.0)

    nearest_road_hit  = boundary_hit_positions[jnp.argmin(road_masked)]
    nearest_grass_hit = boundary_hit_positions[jnp.argmin(grass_masked)]
    any_center_visible = (road_masked.min() < sense_range) & (grass_masked.min() < sense_range)

    center_pos = (nearest_road_hit + nearest_grass_hit) / 2.0
    center_dir = center_pos - agent_pos
    center_dir_norm = center_dir / (jnp.linalg.norm(center_dir) + 1e-6)
    cosine_center = jnp.where(
        any_center_visible,
        jnp.dot(agent_vel, center_dir_norm) / safe_vel_norm,
        cosine_bearing,  # edges not visible: follow global bearing
    )

    return jnp.where(in_target, cosine_center, cosine_entry)


class LidarTarget(LidarEnv):

    COSINE_SIM_REWARD_COEFF = 0.1        # global waypoint bearing alignment
    NEXT_CLUSTER_BONUS = 4.0
    INCORRECT_CLUSTER_PENALTY = -2.0
    STAY_IN_CLUSTER_BONUS = 0.2
    VELOCITY_PENALTY_IN_CLUSTER = -2
    TERRAIN_REWARD_COEFF = 0.3           # mid-range: per-ray delta terrain value
    PREF_VECTOR_REWARD_COEFF = 0.1       # long-range: preference vector alignment
    TERRAIN_PENALTY_COEFF = 0.2          # immediate: road/grass occupancy penalty
    WRONG_TERRAIN_SPEED_COEFF = 0.3      # immediate: speed penalty when off sidewalk

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 2,
        "default_area_size": 1.5,
        "dist2goal": 0.01, # This value is now largely unused for reward unless repurposed
        "top_k_rays": 8,

        # Ensure bridge-related params are also in LidarEnv's PARAMS for reset() to use
        "num_bridges": 1,
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1]
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarTarget, self).__init__(num_agents, area_size, max_step, dt, params)

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Tuple[Reward, jnp.ndarray]:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        agent_vel = agent_states[:, 2:4]

        # --- Extract necessary info from env_states ---
        env_states = graph.env_states
        current_cluster_oh_episode = env_states.current_cluster_oh # Per-agent current high-level cluster OH
        start_cluster_oh_episode = env_states.start_cluster_oh
        next_cluster_oh_episode = env_states.next_cluster_oh     # Per-agent next high-level cluster OH
        actual_cluster_id = jnp.argmax(current_cluster_oh_episode, axis=-1)
        bonus_awarded_state = env_states.next_cluster_bonus_awarded

        bridge_center = env_states.bridge_center
        bridge_length = env_states.bridge_length
        bridge_gap_width = env_states.bridge_gap_width
        bridge_wall_thickness = env_states.bridge_wall_thickness
        bridge_theta = env_states.bridge_theta # In radians

        bearing = env_states.bearing
        current_terrain_oh = env_states.current_terrain_oh  # (n_agents, 3)
        current_terrain_id = jnp.argmax(current_terrain_oh, axis=-1)  # (n_agents,)

        # Check if the agent is in the next cluster
        next_cluster_id = jnp.argmax(next_cluster_oh_episode, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        # ── Global waypoint bearing reward (long-range, cluster navigation) ──
        dense_reward = jnp.where(
            ~is_in_next_cluster,  # Condition: If NOT in the next cluster
            jax_vmap(_calculate_bearing_reward, in_axes=(0, 0))(agent_vel, bearing).mean(),
            0.0 # If in the next cluster, the reward is 0.0
        ).mean()
        reward = dense_reward * self.COSINE_SIM_REWARD_COEFF

        is_bridge_env = bridge_length > 0.0

        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY,
        ), in_axes=(0, 0, 0, 0))

        per_agent_reward, per_agent_bonus_awarded_updated = vmap_cluster_reward_fn(
            current_cluster_oh_episode,
            start_cluster_oh_episode,
            next_cluster_oh_episode,
            bonus_awarded_state
        )

        # Use jnp.where separately for each output
        reward_cluster = jnp.where(is_bridge_env, per_agent_reward, 0.0)
        next_cluster_bonus_awarded_updated = jnp.where(is_bridge_env, per_agent_bonus_awarded_updated, bonus_awarded_state)

        reward += jnp.mean(reward_cluster)

        velocity_magnitude_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        velocity_penalty = jnp.where(is_in_next_cluster, velocity_magnitude_sq, 0.0).mean()
        reward += velocity_penalty * self.VELOCITY_PENALTY_IN_CLUSTER

        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.001

        # ── Immediate terrain penalties (one-hot "haptic" feedback) ──────────
        # Road is strongly penalised; Grass is mildly penalised; Sidewalk = 0.
        road_penalty  = jnp.where(current_terrain_id == 0, -1.0, 0.0)
        grass_penalty = jnp.where(current_terrain_id == 1, -0.3, 0.0)
        immediate_terrain_penalty = (road_penalty + grass_penalty).mean()
        reward += jnp.where(is_bridge_env, immediate_terrain_penalty * self.TERRAIN_PENALTY_COEFF, 0.0)

        # ── Speed penalty when on non-target terrain ─────────────────────────
        # Discourage fast movement on Road or Grass (rough/dangerous terrain).
        not_on_sidewalk = (current_terrain_id != TARGET_TERRAIN_ID)
        speed_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        wrong_terrain_speed = jnp.where(not_on_sidewalk, speed_sq, 0.0).mean()
        reward -= jnp.where(is_bridge_env, wrong_terrain_speed * self.WRONG_TERRAIN_SPEED_COEFF, 0.0)

        # ── Semantic lidar rewards (mid-range + long-range from lidar data) ──
        # is_bridge_env is a JAX array — no Python if; gate with jnp.where per term.
        n_rays = self._params["n_rays"]
        sense_range = self._params["comm_radius"]

        # Use full lidar data from env_states (all n_rays per agent, not top-k graph subset)
        all_hit_pos = jnp.reshape(
            env_states.lidar_hit_positions[:2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays, 2),
        )
        boundary_hit_pos = all_hit_pos[:, n_rays:, :]  # (n_agents, n_rays, 2)

        all_terrain_ids = jnp.reshape(
            env_states.lidar_hit_terrain_ids[:2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays),
        )
        boundary_terrain_ids = all_terrain_ids[:, n_rays:]  # (n_agents, n_rays)

        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # Mid-range: per-ray delta terrain value reward
        per_agent_terrain_reward = jax_vmap(
            ft.partial(_terrain_reward_per_agent, sense_range=sense_range)
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, current_terrain_id)
        reward += jnp.where(is_bridge_env, jnp.mean(per_agent_terrain_reward) * self.TERRAIN_REWARD_COEFF, 0.0)

        # Long-range: preference vector (sidewalk entry / centerline tracking)
        # Falls back to global bearing when terrain boundaries are not visible.
        per_agent_pref_reward = jax_vmap(
            ft.partial(_calculate_preference_vector_reward, sense_range=sense_range),
            in_axes=(0, 0, 0, 0, 0, 0),
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, agent_vel, current_terrain_id, bearing)
        reward += jnp.where(is_bridge_env, per_agent_pref_reward.mean() * self.PREF_VECTOR_REWARD_COEFF, 0.0)

        return reward*0.1, next_cluster_bonus_awarded_updated

    def get_reward_components(self, graph: LidarEnvGraphsTuple) -> dict:
        """Returns individual reward components as a flat dict for wandb logging.
        Call outside JIT (trainer eval loop) on a single graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        env_states = graph.env_states
        agent_vel = agent_states[:, 2:4]
        bearing = env_states.bearing
        current_terrain_oh = env_states.current_terrain_oh
        current_terrain_id = jnp.argmax(current_terrain_oh, axis=-1)
        is_bridge_env = env_states.bridge_length > 0.0

        actual_cluster_id = jnp.argmax(env_states.current_cluster_oh, axis=-1)
        next_cluster_id = jnp.argmax(env_states.next_cluster_oh, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        # Bearing reward
        bearing_r = jnp.where(
            ~is_in_next_cluster,
            jax_vmap(_calculate_bearing_reward, in_axes=(0, 0))(agent_vel, bearing).mean(),
            0.0,
        ).mean() * self.COSINE_SIM_REWARD_COEFF

        # Immediate terrain penalties
        road_penalty  = jnp.where(current_terrain_id == 0, -1.0, 0.0)
        grass_penalty = jnp.where(current_terrain_id == 1, -0.3, 0.0)
        immediate_r = float(jnp.where(
            is_bridge_env,
            (road_penalty + grass_penalty).mean() * self.TERRAIN_PENALTY_COEFF,
            0.0,
        ))

        # Speed penalty on wrong terrain
        not_on_sidewalk = (current_terrain_id != TARGET_TERRAIN_ID)
        speed_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        speed_r = -float(jnp.where(
            is_bridge_env,
            jnp.where(not_on_sidewalk, speed_sq, 0.0).mean() * self.WRONG_TERRAIN_SPEED_COEFF,
            0.0,
        ))

        # Semantic lidar components
        n_rays = self._params["n_rays"]
        sense_range = self._params["comm_radius"]
        all_hit_pos = jnp.reshape(
            env_states.lidar_hit_positions[: 2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays, 2),
        )
        boundary_hit_pos = all_hit_pos[:, n_rays:, :]
        all_terrain_ids = jnp.reshape(
            env_states.lidar_hit_terrain_ids[: 2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays),
        )
        boundary_terrain_ids = all_terrain_ids[:, n_rays:]
        agent_pos = agent_states[:, :2]

        per_agent_terrain = jax_vmap(
            ft.partial(_terrain_reward_per_agent, sense_range=sense_range)
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, current_terrain_id)
        terrain_r = float(jnp.where(
            is_bridge_env, jnp.mean(per_agent_terrain) * self.TERRAIN_REWARD_COEFF, 0.0
        ))

        per_agent_pref = jax_vmap(
            ft.partial(_calculate_preference_vector_reward, sense_range=sense_range),
            in_axes=(0, 0, 0, 0, 0, 0),
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, agent_vel, current_terrain_id, bearing)
        pref_r = float(jnp.where(
            is_bridge_env, per_agent_pref.mean() * self.PREF_VECTOR_REWARD_COEFF, 0.0
        ))

        # Current terrain distribution
        terrain_names = {0: "road", 1: "grass", 2: "sidewalk"}
        terrain_fracs = {
            f"eval/terrain_frac_{name}": float((current_terrain_id == tid).mean())
            for tid, name in terrain_names.items()
        }

        # Cost components: agent-agent vs obstacle
        cost = self.get_cost(graph)  # (n_agents, 2): col 0=agent-agent, col 1=obstacle
        cost_agent_agent = float(jnp.maximum(cost[:, 0], 0.0).mean())
        cost_obstacle    = float(jnp.maximum(cost[:, 1], 0.0).mean())

        return {
            "eval/reward_bearing":           float(bearing_r),
            "eval/reward_terrain_immediate": immediate_r,
            "eval/reward_speed_penalty":     speed_r,
            "eval/reward_terrain_midrange":  terrain_r,
            "eval/reward_pref_vector":       pref_r,
            "eval/cost_agent_agent":         cost_agent_agent,
            "eval/cost_obstacle":            cost_obstacle,
            **terrain_fracs,
        }

    def state2feat(self, state: State) -> Array:
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # # agent - goal connection
        agent_goal_edges = []

        # agent - obs connection
        # Each agent has 2*top_k hit nodes in the graph: first top_k = obstacle hits, last top_k = boundary hits
        agent_obs_edges = []
        hits_per_agent = 2 * self._params["top_k_rays"]
        n_hits = hits_per_agent * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * hits_per_agent, (i + 1) * hits_per_agent)
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, hits_per_agent))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges
