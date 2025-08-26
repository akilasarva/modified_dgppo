import jax.numpy as jnp
import jax.debug as jdebug

from typing import Optional, Dict
import functools as ft

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.utils.utils import jax_vmap

import functools as ft
from typing import Optional, Dict

ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "along_wall_0", "along_wall_1", "along_wall_2", "along_wall_3",
        "past_building_0", "past_building_1", "past_building_2", "past_building_3",
        "around_corner_0", "around_corner_1", "around_corner_2", "around_corner_3"
    ]

def _calculate_bearing_reward(agent_vel: jnp.ndarray, target_bearing: float) -> float:
    # Get the x and y components of the target bearing vector
    target_vector = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])
    
    # Normalize the agent's velocity vector
    norm_agent_vel = jnp.linalg.norm(agent_vel)
    safe_norm_agent_vel = jnp.where(norm_agent_vel > 1e-6, norm_agent_vel, 1e-6)
    
    # Calculate the cosine similarity between the velocity and the target vector
    cosine_sim = jnp.dot(agent_vel, target_vector) / safe_norm_agent_vel

    # Return a scaled cosine similarity as a dense reward
    return cosine_sim * 0.05  # Scale the reward for better impact

# Your existing helper function, which will now be correctly vectorized
def _calculate_cluster_reward_per_agent(
    agent_pos_i: jnp.ndarray,
    current_cluster_oh_episode_i: jnp.ndarray,
    next_cluster_oh_episode_i: jnp.ndarray,
    building_center: jnp.ndarray,
    building_width: float,
    building_height: float,
    building_theta: float,
    environment_area_size: float,
    cluster_map: Dict[str, int],
    id_to_curriculum_prefix_map: Dict[int, str],
    allowed_id_transitions_array: jnp.ndarray,
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float
) -> float:
    # This code remains the same as your original
    from dgppo.env.building_behavior_associator import BehaviorBuildings
    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)
    building_params = [(building_center, building_width, building_height, building_theta)]
    associator = BehaviorBuildings(buildings=building_params, all_region_names=ALL_POSSIBLE_REGION_NAMES)
    actual_cluster_id = associator.get_current_behavior(agent_pos_i)
    current_agent_reward = 0.0
    has_moved_into_next_cluster = (actual_cluster_id == next_cluster_id_episode_i) & \
                                 (actual_cluster_id != current_cluster_id_episode_i)
    current_agent_reward += jnp.where(has_moved_into_next_cluster, next_cluster_bonus, 0.0)
    is_in_next_cluster = (actual_cluster_id == next_cluster_id_episode_i)
    current_agent_reward += jnp.where(is_in_next_cluster, stay_in_cluster_bonus, 0.0)
    is_in_correct_cluster = (actual_cluster_id == current_cluster_id_episode_i)
    is_in_next_cluster_bonus = (actual_cluster_id == next_cluster_id_episode_i)
    is_in_target_region = jnp.logical_or(is_in_correct_cluster, is_in_next_cluster_bonus)
    current_agent_reward += jnp.where(jnp.logical_not(is_in_target_region), incorrect_cluster_penalty, 0.0)
    return current_agent_reward

class LidarTarget(LidarEnv):

    COSINE_SIM_REWARD_COEFF = 0.005
    NEXT_CLUSTER_BONUS = 40.0
    INCORRECT_CLUSTER_PENALTY = -0.5
    STAY_IN_CLUSTER_BONUS = 0.05
    VELOCITY_PENALTY_IN_CLUSTER = -0.5

    # NEW: Update PARAMS for the biuilding environment
    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        
        # New building-related params
        "num_buildings": 1,
        "num_bridges": 0, # Ensure this is set to 0
        "building_width_range": [0.3, 0.6],
        "building_height_range": [0.3, 0.6],
        "building_theta_range": [0, 2 * jnp.pi],
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
        
    
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :2]
        agent_vel = agent_states[:, 2:4]
        
        env_states = graph.env_states
        current_cluster_oh_episode = env_states.current_cluster_oh
        next_cluster_oh_episode = env_states.next_cluster_oh
        bearing = env_states.bearing

        # --- 1. Dense Reward: Cosine similarity with the target bearing ---
        # The coefficient of 0.5 is already a bit high, consider lowering it
        dense_reward = jax_vmap(ft.partial(_calculate_bearing_reward), 
                                 in_axes=(0, 0))(agent_vel, bearing).mean()
        reward = dense_reward

        # --- 2. Cluster Bonus/Penalty ---
        is_building_env = env_states.building_width > 0.0
        
        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            building_center=env_states.building_center,
            building_width=env_states.building_width,
            building_height=env_states.building_height,
            building_theta=env_states.building_theta,
            environment_area_size=self.area_size,
            cluster_map=self.CLUSTER_MAP,
            id_to_curriculum_prefix_map=self._id_to_curriculum_prefix_map,
            allowed_id_transitions_array=self.CURRICULUM_TRANSITIONS_INT_IDS,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY
        ), in_axes=(0, 0, 0))

        reward_cluster = jnp.where(is_building_env,
            jnp.mean(vmap_cluster_reward_fn(
                agent_states[:, :2],
                current_cluster_oh_episode,
                next_cluster_oh_episode
            )),
            0.0
        )
        reward += reward_cluster

        # --- 3. New: Velocity Penalty in Target Cluster ---
        # This directly addresses the "zooming past" problem
        # First, determine which cluster the agent is in
        from dgppo.env.building_behavior_associator import BehaviorBuildings
        building_params = [(env_states.building_center, env_states.building_width, env_states.building_height, env_states.building_theta)]
        associator = BehaviorBuildings(buildings=building_params, all_region_names=ALL_POSSIBLE_REGION_NAMES)
        actual_cluster_id = jax_vmap(associator.get_current_behavior)(agent_pos)
        
        # Check if the agent is in the next cluster
        next_cluster_id = jnp.argmax(next_cluster_oh_episode, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        # Calculate velocity magnitude squared for penalty
        velocity_magnitude_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        
        # Apply a penalty only when the agent is in the next cluster
        # This forces the agent to slow down to accumulate positive reward
        velocity_penalty = jnp.where(is_in_next_cluster, velocity_magnitude_sq, 0.0).mean()
        reward += velocity_penalty * self.VELOCITY_PENALTY_IN_CLUSTER
        
        # --- 4. Other Penalties (keep as is, but their impact is now relatively smaller) ---
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        dist2goal = jnp.linalg.norm(goals[:, :2] - agent_pos, axis=-1)
        reward_dist = -dist2goal.mean() * 0.0001
        reward += reward_dist

        reward_not_reaching_goal = jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * -0.001
        reward += reward_not_reaching_goal
        
        reward_action_penalty = (jnp.linalg.norm(action, axis=1) ** 2).mean() * -0.0001
        reward += reward_action_penalty
        
        return reward*0.1
    
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
        # for i_agent in range(self.num_agents):
        #     agent_state_i = state.agent[i_agent]
        #     goal_state_i = state.goal[i_agent]
        #     agent_goal_feats_i = self.state2feat(agent_state_i) - self.state2feat(goal_state_i)
        #     agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
        #                                       jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # agent - obs connection
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

