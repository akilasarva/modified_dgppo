import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np # Keep if needed for other non-JAX ops, but avoid for JAX arrays
import functools as ft
import jax
import jax.debug as jd
from jax.random import PRNGKey, uniform, split

from typing import NamedTuple, Tuple, Optional, List, Dict 
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState, PRNGKey
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.plot import render_lidar # Ensure BehaviorAssociator is imported from plot
from dgppo.env.utils import get_lidar, get_node_goal_rng
from dgppo.env.intersection_behavior_associator import BehaviorIntersection

from jax import jit, lax

class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
    
    bearing: Float[Array, "n_agent"]
    current_cluster_oh: Float[Array, "n_agent n_cluster"]
    start_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_bonus_awarded: jnp.ndarray
    
    is_four_way: bool
    center: jnp.ndarray
    passage_width: float
    obs_len: float
    global_angle: float
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]
    
    @property
    def n_cluster(self) -> int:
        return 4

LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]

def create_fourway_intersection(
    center: jnp.ndarray,
    passage_width: float,
    obs_len: float,
    global_angle: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    # center = uniform(key_center, shape=(2,), minval=-area_size / 2, maxval=area_size / 2)
    
    # passage_width = uniform(
    #     key_passage, 
    #     (), 
    #     minval=intersection_params.get("passage_min", 0.25), 
    #     maxval=intersection_params.get("passage_max", 0.5)
    # )
    # obs_len = uniform(
    #     key_obs_len, 
    #     (), 
    #     minval=intersection_params.get("obs_len_min", 0.4), 
    #     maxval=intersection_params.get("obs_len_max", 0.75)
    # )

    # global_angle = uniform(key_angle, (), minval=0.0, maxval=2 * jnp.pi)
    
    half_passage = passage_width / 2.0
    offset = obs_len / 2.0 + half_passage

    obs_pos_local = jnp.array([
        [-offset, 0.0], [offset, 0.0], [0.0, offset], [0.0, -offset]
    ])

    obs_len_x = jnp.array([obs_len, obs_len, obs_len, obs_len])
    obs_len_y = jnp.array([obs_len, obs_len, obs_len, obs_len])
    obs_theta = jnp.array([0.0, 0.0, jnp.pi / 2.0, jnp.pi / 2.0])

    cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
    rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_obs_pos = jnp.einsum('ij,kj->ki', rotation_matrix, obs_pos_local)
    rotated_obs_pos += center
    rotated_obs_theta = obs_theta + global_angle

    return rotated_obs_pos, obs_len_x, obs_len_y, rotated_obs_theta

def create_threeway_intersection(
    center: jnp.ndarray,
    passage_width: float,
    obs_len: float,
    global_angle: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    half_passage = passage_width / 2.0
    diag_len = jnp.sqrt((obs_len + half_passage)**2 + (half_passage)**2)
    
    pos1_local = jnp.array([0, -obs_len/2 - half_passage])
    
    angle2 = jnp.pi * 5 / 6
    pos2_local = jnp.array([diag_len * jnp.cos(angle2), diag_len * jnp.sin(angle2)])
    
    angle3 = jnp.pi * 1 / 6
    pos3_local = jnp.array([diag_len * jnp.cos(angle3), diag_len * jnp.sin(angle3)])
    
    obs_pos_local = jnp.stack([pos1_local, pos2_local, pos3_local])
    
    obs_len_x = jnp.array([obs_len, diag_len * 2, diag_len * 2])
    obs_len_y = jnp.array([obs_len, obs_len, obs_len])
    
    theta1 = jnp.pi / 2
    theta2 = angle2 + jnp.pi / 2
    theta3 = angle3 + jnp.pi / 2
    obs_theta_local = jnp.stack([theta1, theta2, theta3])
    
    cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
    rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_obs_pos = jnp.einsum('ij,kj->ki', rotation_matrix, obs_pos_local)
    rotated_obs_pos += center
    rotated_obs_theta = obs_theta_local + global_angle

    return rotated_obs_pos, obs_len_x, obs_len_y, rotated_obs_theta

class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "n_obs": 1,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        
        "is_four_way_p": 0.5, # Probability of generating a 4-way intersection (0.5 for a 50/50 chance)
        "intersection_size_range": [0.5, 1], # Overall size of the intersection region
        "passage_width_range": [0.25, 0.5], # Min/max for the width of the road passages
        "obs_wall_range": [0.4, 0.75],
    }
    
    # In LidarEnv class
    ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "in_intersection",
        "passage_0_enter", "passage_0_exit",
        "passage_1_enter", "passage_1_exit",
        "passage_2_enter", "passage_2_exit",
        "passage_3_enter", "passage_3_exit",
    ]

    CLUSTER_MAP: Dict[str, int] = {
        "open_space": 0,
        "in_intersection": 1,
        "passage_enter": 2,
        "passage_exit": 3,
    }
    
    _SPECIFIC_TO_GENERAL_MAP = {
        "in_intersection": "in_intersection",
        "passage_0_enter": "passage_enter",
        "passage_0_exit": "passage_exit",
        "passage_1_enter": "passage_enter",
        "passage_1_exit": "passage_exit",
        "passage_2_enter": "passage_enter",
        "passage_2_exit": "passage_exit",
        "passage_3_enter": "passage_enter",
        "passage_3_exit": "passage_exit",
        "open_space": "open_space",
        "passage_enter": "passage_enter",
        "passage_exit": "passage_exit",
    }
    
    # 3. Override _CURRICULUM_TRANSITIONS_STR for building tasks
    _CURRICULUM_TRANSITIONS_STR = [
        ("open_space", "passage_enter"),
        ("passage_enter", "in_intersection"),
        ("in_intersection", "passage_exit"),
        ("passage_exit", "open_space")
    ]
    
    # 3. Override _CURRICULUM_TRANSITIONS_STR for building tasks
    VALID_GOAL_MAP = {
        "open_space": ["passage_0_enter", "passage_1_enter", "passage_2_enter", "passage_3_enter"],
        "passage_0_enter": ["in_intersection"],
        "passage_1_enter": ["in_intersection"],
        "passage_2_enter": ["in_intersection"],
        "passage_3_enter": ["in_intersection"],
        "in_intersection": [
            "passage_0_exit", "passage_1_exit", "passage_2_exit", "passage_3_exit"
        ],
        "passage_0_exit": ["open_space"],
        "passage_1_exit": ["open_space"],
        "passage_2_exit": ["open_space"],
        "passage_3_exit": ["open_space"]
    }

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 128,
        dt: float = 0.03,
        params: dict = None,
        n_cluster: int = 4
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents
        self.n_cluster = n_cluster
        
        # # --- All behavior-related data should be defined here, NOT in the reset function ---
        # if self.n_cluster < len(set(self.CLUSTER_MAP.values())):
        #     print(f"Warning: n_cluster ({self.n_cluster}) is less than the number of unique cluster IDs in CLUSTER_MAP ({len(set(self.CLUSTER_MAP.values()))}).")

        self._id_to_curriculum_prefix_map: Dict[int, str] = {v: k for k, v in self.CLUSTER_MAP.items()}
        
        allowed_id_transitions_list: List[Tuple[int, int]] = []
        for start_specific, end_specific in self._CURRICULUM_TRANSITIONS_STR:
            start_general = self._SPECIFIC_TO_GENERAL_MAP.get(start_specific, None)
            end_general = self._SPECIFIC_TO_GENERAL_MAP.get(end_specific, None)

            if start_general is not None and end_general is not None:
                start_id = self.CLUSTER_MAP.get(start_general, -1)
                end_id = self.CLUSTER_MAP.get(end_general, -1)

                if start_id != -1 and end_id != -1:
                    allowed_id_transitions_list.append((start_id, end_id))
                else:
                    print(f"Warning: Cluster map missing IDs for {start_general} or {end_general}.")
            else:
                print(f"Warning: Specific to general map missing entry for {start_specific} or {end_specific}.")

        if allowed_id_transitions_list:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.array(allowed_id_transitions_list, dtype=jnp.int32)
        else:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.empty((0, 2), dtype=jnp.int32)

        # IMPORTANT: These must be set to None so they can be generated in the reset function.
        # DO NOT GENERATE THEM HERE.
        self.is_four_way = None
        self.intersection_params = None

        # The associator itself is part of the dynamic state, so it will be created in reset
        self.associator = None
        self._name_to_id = None
        self._id_to_name = None
        self.all_region_centroids_jax_array = None
        
        # The transition centroids can't be prepared here because the associator doesn't exist yet
        self.VALID_TRANSITIONS_INT_IDS = None
        
    def _prepare_transition_centroids(self):
        # This method is called from __init__
        valid_transitions_list = []
        
        for start_name, goal_names in self.VALID_GOAL_MAP.items():
            if start_name in self._name_to_id:
                start_id = self._name_to_id[start_name]
                for goal_name in goal_names:
                    if goal_name in self._name_to_id:
                        goal_id = self._name_to_id[goal_name]
                        
                        # Look up the centroids using the consistent ID-to-centroid mapping
                        start_centroid = self.associator.all_region_centroids_jax_array[start_id]
                        goal_centroid = self.associator.all_region_centroids_jax_array[goal_id]

                        # Store the centroids as a pair
                        valid_transitions_list.append((start_centroid, goal_centroid))

        # Convert the list to a JAX array
        self.all_region_centroid_pairs_jax_array = jnp.array(valid_transitions_list)

    def _get_valid_transitions_as_ids(self):
        # This function now correctly accesses the pre-initialized associator.
        transitions = []
        name_to_id = self._name_to_id
        
        if not hasattr(self, 'VALID_GOAL_MAP'):
            return transitions

        for start_specific, goals_specific in self.VALID_GOAL_MAP.items():
            if start_specific in name_to_id:
                start_id = name_to_id[start_specific]
                for goal_specific in goals_specific:
                    if goal_specific in name_to_id:
                        goal_id = name_to_id[goal_specific]
                        transitions.append((start_id, goal_id))
        return transitions
        
    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy
    
    @property
    def bearing_dim(self) -> int:
        return 1 # Bearing is a single float
    
    @property
    def cluster_oh_dim(self) -> int:
        return self.n_cluster # One-hot encoding dimension

    @property
    def node_dim(self) -> int:
        return self.state_dim + self.bearing_dim + 3 * self.cluster_oh_dim + 3
    
    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(
        self,
        key: jr.PRNGKey,
        current_clusters: Optional[Array] = None,
        start_clusters: Optional[Array] = None,
        next_clusters: Optional[Array] = None,
        custom_obstacles: Optional[Tuple[Array, Array, Array, Array]] = None,
        transition_index: Optional[int] = None
    ) -> GraphsTuple:
        
        # Initialize lists and default values
        all_obs_pos_list: List[jnp.ndarray] = []
        all_obs_len_x_list: List[jnp.ndarray] = []
        all_obs_len_y_list: List[jnp.ndarray] = []
        all_obs_theta_list: List[jnp.ndarray] = []
        
        # This is where we'll store the intersection parameters for the env state
        intersection_params_env_state: Dict = {}
        is_four_way_env_state: bool = False
        
        # --- Generate Obstacles (including the intersection) ---
        if custom_obstacles is not None:
            obs_pos, obs_len_x, obs_len_y, obs_theta = custom_obstacles
            all_obs_pos_list.append(obs_pos)
            all_obs_len_x_list.append(obs_len_x)
            all_obs_len_y_list.append(obs_len_y)
            all_obs_theta_list.append(obs_theta)
        else:
            # Standard random obstacles (same as your bridge code)
            n_rng_obs = self._params.get("n_obs", 0)
            if n_rng_obs > 0:
                obstacle_key, key = jr.split(key, 2)
                obs_pos_orig = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
                length_key, theta_key, key = jr.split(key, 3)
                obs_len_orig = jr.uniform(
                    length_key,
                    (n_rng_obs, 2),
                    minval=self._params["obs_len_range"][0],
                    maxval=self._params["obs_len_range"][1],
                )
                obs_theta_orig = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)
                all_obs_pos_list.append(obs_pos_orig)
                all_obs_len_x_list.append(obs_len_orig[:, 0])
                all_obs_len_y_list.append(obs_len_orig[:, 1])
                all_obs_theta_list.append(obs_theta_orig)
                
            # --- Intersection-specific obstacle generation logic ---
            # Same random parameter generation pattern as your bridge code
            intersection_key, key = jr.split(key)
            is_four_way_key, center_key, angle_key, passage_key, obs_len_key = jr.split(intersection_key, 5)

            is_four_way_env_state = self.is_four_way # Use class attribute if set, otherwise...
            # Conditionally override with a random choice
            is_four_way_p = self.params.get('is_four_way_p', 0.5)
            is_four_way_env_state = True
            # lax.cond(
            #     self.is_four_way is None,
            #     lambda: bool(jr.bernoulli(is_four_way_key, p=is_four_way_p)),
            #     lambda: self.is_four_way
            # )
            
            # Randomly generate intersection parameters (if not provided)
            # intersection_params_env_state = lax.cond(
            #     self.intersection_params is None,
            #     lambda: {
            #         'center': jr.uniform(center_key, shape=(2,), minval=-self.area_size / 2, maxval=self.area_size / 2),
            #         'size': jr.uniform(key, shape=(), minval=5.0, maxval=10.0),
            #         'passage_min': self.params.get('passage_min', 0.25),
            #         'passage_max': self.params.get('passage_max', 0.5),
            #         'obs_len_min': self.params.get('obs_len_min', 0.4),
            #         'obs_len_max': self.params.get('obs_len_max', 0.75),
            #     },
            #     lambda: self.intersection_params
            # )
            
            # Use our new obstacle-building function
            intersection_obs_pos, intersection_obs_len_x, intersection_obs_len_y, intersection_obs_theta, intersection_params_env_state = \
                create_intersection_obstacles(
                    key=intersection_key,
                    is_four_way=is_four_way_env_state,
                    intersection_params={},
                    area_size=self.area_size
                )

            all_obs_pos_list.append(intersection_obs_pos)
            all_obs_len_x_list.append(intersection_obs_len_x)
            all_obs_len_y_list.append(intersection_obs_len_y)
            all_obs_theta_list.append(intersection_obs_theta)

        # Combine all obstacles and create the final obstacle object
        if all_obs_pos_list:
            combined_obs_pos = jnp.concatenate(all_obs_pos_list, axis=0)
            combined_obs_len_x = jnp.concatenate(all_obs_len_x_list, axis=0)
            combined_obs_len_y = jnp.concatenate(all_obs_len_y_list, axis=0)
            combined_obs_theta = jnp.concatenate(all_obs_theta_list, axis=0)
            obstacles = self.create_obstacles(
                combined_obs_pos,
                combined_obs_len_x,
                combined_obs_len_y,
                combined_obs_theta
            )
        else:
            obstacles = None
        
        # --- Agent and Goal Generation ---
        # Create the BehaviorIntersection object using the generated parameters
        associator = BehaviorIntersection(
            key=key,
            is_four_way=is_four_way_env_state,
            intersection_params=intersection_params_env_state,
            all_region_names=self.ALL_POSSIBLE_REGION_NAMES,
            area_size=self.area_size,
            obstacles=obstacles # Pass theobstacles to the associator
        )
        
        # Use the associator to select a valid curriculum transition
        valid_transitions_list = []
        for start_name, goal_names in self.VALID_GOAL_MAP.items():
            if start_name in associator.region_name_to_id:
                start_id = associator.region_name_to_id[start_name]
                for goal_name in goal_names:
                    if goal_name in associator.region_name_to_id:
                        goal_id = associator.region_name_to_id[goal_name]
                        valid_transitions_list.append((start_id, goal_id))
        
        VALID_TRANSITIONS_INT_IDS_local = jnp.array(valid_transitions_list)
        
        key_select, key = jr.split(key)
        transition_idx = jr.choice(key_select, jnp.arange(VALID_TRANSITIONS_INT_IDS_local.shape[0]))
        start_specific_id, goal_specific_id = VALID_TRANSITIONS_INT_IDS_local[transition_idx]
        
        initial_pos = associator.get_region_centroid(start_specific_id)
        goal_pos = associator.get_region_centroid(goal_specific_id)
        
        current_cluster_id = jnp.squeeze(self.associator.get_current_behavior(initial_pos))
        
        start_cluster_id = jnp.squeeze(self._specific_id_to_general_id[start_specific_id])
        next_cluster_id = jnp.squeeze(self._specific_id_to_general_id[goal_specific_id])
        
        key = key_loop
                
        def create_agent_state(key_single_agent: jax.Array):
            key_agent_pos, key_goal_pos = jr.split(key_single_agent, 2)

            initial_pos_agent = initial_pos + jr.normal(key_agent_pos, (2,)) * 0.025
            goal_pos_agent = goal_pos + jr.normal(key_goal_pos, (2,)) * 0.025
            
            initial_pos_agent = jnp.clip(initial_pos_agent, 0, self.area_size)
            goal_pos_agent = jnp.clip(goal_pos_agent, 0, self.area_size)

            bearing = jnp.arctan2(goal_pos_agent[1] - initial_pos_agent[1], goal_pos_agent[0] - initial_pos_agent[0])

            agent_state = jnp.array([initial_pos_agent[0], initial_pos_agent[1], 0.0, 0.0])
            goal_state = jnp.array([goal_pos_agent[0], goal_pos_agent[1], 0.0, 0.0])
            
            return agent_state, goal_state, bearing
            
        key_agents = jr.split(key, self.num_agents)
        states, goals, bearing = jax.vmap(create_agent_state)(key_agents)

        current_clusters = jnp.broadcast_to(
            jax.nn.one_hot(current_cluster_id, self.n_cluster), 
            (self.num_agents, self.n_cluster)
        )
        start_clusters = jnp.broadcast_to(
            jax.nn.one_hot(start_cluster_id, self.n_cluster), 
            (self.num_agents, self.n_cluster)
        )
        next_clusters = jnp.broadcast_to(
            jax.nn.one_hot(next_cluster_id, self.n_cluster), 
            (self.num_agents, self.n_cluster)
        )
        
        env_states = LidarEnvState(
            agent=states, 
            goal=goals, 
            obstacle=obstacles,
            bearing=bearing, 
            current_cluster_oh=current_clusters,
            start_cluster_oh=start_clusters,
            next_cluster_oh=next_clusters,
            next_cluster_bonus_awarded=jnp.zeros(self.num_agents, dtype=jnp.bool_),
            is_four_way=is_four_way_env_state,
            intersection_params=intersection_params_env_state
        )

        lidar_data = self.get_lidar_data(states, obstacles)
        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)


    def step(
        self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        agent_base_states = graph.env_states.agent
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle
        
        is_four_way = graph.env_states.is_four_way
        intersection_params = graph.env_states.intersection_params
        bearing = graph.env_states.bearing

        current_id = jax_vmap(self.associator.get_current_behavior)(agent_base_states[:, :2])
        current_cluster_oh = jax.nn.one_hot(current_id, self.n_cluster)
        start_cluster_oh = graph.env_states.start_cluster_oh        
        next_cluster_oh = graph.env_states.next_cluster_oh

        next_agent_base_states = self.agent_step_euler(agent_base_states, action)
        bonus_awarded_state = graph.env_states.next_cluster_bonus_awarded
        reward, bonus_awarded_updated = self.get_reward(graph, action) #, bonus_awarded_state)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()
        
        next_env_state = LidarEnvState(
            next_agent_base_states, 
            goals, 
            obstacles,
            bearing, 
            current_cluster_oh,
            start_cluster_oh,
            next_cluster_oh,
            bonus_awarded_updated,
            is_four_way=is_four_way,
            intersection_params=intersection_params
        )
        
        lidar_data_next = self.get_lidar_data(next_agent_base_states, obstacles)
        info = {}
        done = jnp.array(False)

        return self.get_graph(next_env_state, lidar_data_next), reward, cost, done, info
    
    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward: #, bonus_awarded_state: jnp.ndarray
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost

    def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        **kwargs
    ) -> None:
        
        first_env_state = rollout.graph.env_states
    
        intersection_params_for_render = {
            "is_four_way": first_env_state.is_four_way,
            "intersection_params": first_env_state.intersection_params,
        }
        
        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2, # Assuming 2D for buildings environment visualization
            n_agent=self.num_agents,
            n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 or self.params["num_buildings"] > 0 else 0,
            r=self.params["car_radius"],
            obs_r=0.0,
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **intersection_params_for_render, # Pass the extracted building parameters explicitly
            **kwargs
        )

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = 8  
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)
        elif n_hits > 0:
            lidar_data = jnp.zeros((n_hits, 2), dtype=jnp.float32)

        # Node features: (x, y, vx, vy, bearing, current_cluster_oh, start_cluster_oh, next_cluster_oh, is_obs, is_goal, is_agent)
        node_feats = jnp.zeros((n_nodes, self.node_dim), dtype=jnp.float32)

        agent_start_idx = 0
        # Base state
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, :self.state_dim].set(state.agent)
        # Bearing
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim].set(state.bearing)
        # Current cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim:self.state_dim+self.bearing_dim+self.n_cluster].set(state.current_cluster_oh)
        # Start cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+self.n_cluster:self.state_dim+self.bearing_dim+2*self.n_cluster].set(state.start_cluster_oh)
        # Next cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+2*self.n_cluster:self.state_dim+self.bearing_dim+3*self.n_cluster].set(state.next_cluster_oh)
        # Is agent indicator (last of 3)
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+3*self.n_cluster+2].set(1.0)

        # Goal features
        goal_start_idx = self.num_agents
        node_feats = node_feats.at[goal_start_idx:goal_start_idx+self.num_goals, :self.state_dim].set(state.goal)
        # Is goal indicator (middle)
        node_feats = node_feats.at[goal_start_idx:goal_start_idx+self.num_goals, self.state_dim+self.bearing_dim+3*self.n_cluster+1].set(1.0)

        # Obstacle (lidar hits)
        if n_hits > 0 and lidar_data is not None:
            obs_start_idx = self.num_agents + self.num_goals
            node_feats = node_feats.at[obs_start_idx:obs_start_idx+n_hits, :2].set(lidar_data)
            # Is obs indicator (first)
            node_feats = node_feats.at[obs_start_idx:obs_start_idx+n_hits, self.state_dim+self.bearing_dim+3*self.n_cluster].set(1.0)

        # Node types
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents:self.num_agents+self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[self.num_agents+self.num_goals:].set(LidarEnv.OBS)

        edge_blocks = self.edge_blocks(state, lidar_data)

        # Raw states
        raw_states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states_padded = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - 2), dtype=lidar_data.dtype)],
                axis=1
            )
            raw_states = jnp.concatenate([raw_states, lidar_states_padded], axis=0)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=raw_states
        ).to_padded()
    
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim