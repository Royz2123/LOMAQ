import numpy as np
import random

from envs.multi_particle.multiagent_particle_env.multiagent.core import World, Agent, Landmark
from envs.multi_particle.multiagent_particle_env.multiagent.scenario import BaseScenario


def is_number(s):
    try:
        float(s)
        return True
    except Exception as e:
        return False


class LocalSpreadScenario(BaseScenario):
    def __init__(self, params):
        self.params = params

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 0
        world.collaborative = True
        world.reward_thresh = self.params["rules"]["reward"]["landmark_radius"]

        # add agents
        world.agents = [Agent() for _ in range(self.params["num_agents"])]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.size = self.params["rules"]["reward"]["agent_radius"]

        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.params["num_landmarks"])]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.params["rules"]["reward"]["landmark_radius"]

            if is_number(self.params["rules"]["reward"]["landmark_occupant_coeff"]):
                landmark.reward = self.params["rules"]["reward"]["landmark_occupant_coeff"]
            elif type(self.params["rules"]["reward"]["landmark_occupant_coeff"]) == list:
                landmark.reward = self.params["rules"]["reward"]["landmark_occupant_coeff"][i]
            else:
                raise Exception("Unsupported landmark reward type")

            # Create different shapes for different reward types
            landmark.res = 30 if landmark.reward <= 1.0 else 2 + int(landmark.reward)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1.0, 0.0, 0.0])

        # set random initial states
        for agent_idx, agent in enumerate(world.agents):
            agent.state.p_pos = self.generate_location(world, agent_idx, is_landmark=False)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            # bounded stuff
            agent.is_bound = self.params["rules"]["bounding"]["is_bound"]
            agent.bound_type = self.params["rules"]["bounding"]["bound_type"]
            agent.bound_dist = self.params["rules"]["bounding"]["bound_dist"]

        for landmark_idx, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.generate_location(world, landmark_idx, is_landmark=True)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Re-eorder landmarks if necessary
        if self.params["rules"]["obs"]["obs_landmarks_order"] == "random":
            random.shuffle(world.landmarks)

    def compute_occupant_rewards(self, world):
        rewards = np.zeros(self.params["num_agents"])

        # set all landmark rewards
        for landmark_idx, landmark in enumerate(world.landmarks):
            coeff = landmark.reward

            # If reward is shared between agents, disregard occupants (irrelevant)
            if self.params["rules"]["reward"]["landmark_occupant_reward"] == "shared":
                # compute all of the agents that are close enough
                viable_agents = [agent for agent in world.agents if LocalSpreadScenario.is_collision(agent, landmark)]

                # Are there any agents on the landmark. If not, continue
                if len(viable_agents) == 0:
                    continue

                for agent in viable_agents:
                    rewards[agent.id] += coeff / len(viable_agents)

            # Or if reward is based on distance, also disregard occupants (irrelevant)
            elif self.params["rules"]["reward"]["landmark_occupant_reward"] == "closest":
                agent_dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                closest_agent = world.agents[np.argmin(agent_dists)]
                closest_dist = min(agent_dists)
                rewards[closest_agent.id] += coeff * self.distance_to_reward(closest_dist)

            else:
                raise Exception("Unrecognized Landmark Occupants Reward Type, exiting...")
        return rewards

    # We reveal here a possible variant for local rewards that should be learnt by the decomposition
    def compute_bonus_rewards(self, world):
        rewards = np.zeros(self.params["num_agents"])

        for landmark in world.landmarks:
            if self.params["rules"]["reward"]["landmark_bonus_reward"] == "binary":
                bonus_reward = np.array([
                    LocalSpreadScenario.is_collision(agent, landmark)
                    for agent in world.agents
                ])
            elif self.params["rules"]["reward"]["landmark_bonus_reward"] == "continuous":
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                bonus_reward = np.array([self.distance_to_reward(dist) for dist in dists])
            else:
                raise Exception("Unrecognized Landmark Bonus Reward Type, exiting...")
            rewards += self.params["rules"]["reward"]["landmark_bonus_coeff"] * bonus_reward
        return rewards

    def compute_collision_rewards(self, world):
        rewards = np.zeros(self.params["num_agents"])

        if self.params["rules"]["reward"]["collisions_reward"] != 0:
            collide_reward = self.params["rules"]["reward"]["collisions_reward"]
            for i in range(self.params["num_agents"]):
                for j in range(i + 1, self.params["num_agents"]):
                    # punish them both equally
                    if LocalSpreadScenario.is_collision(world.agents[i], world.agents[j]):
                        rewards[i] += collide_reward / 2
                        rewards[j] += collide_reward / 2
        return rewards

    def compute_all_rewards(self, world):
        local_rewards = self.compute_occupant_rewards(world)
        local_rewards += self.compute_bonus_rewards(world)
        local_rewards += self.compute_collision_rewards(world)
        return local_rewards

    def generate_location(self, world, entity_idx, is_landmark=False):
        agent_bound_pos = np.zeros(shape=(world.dim_p,))

        # First set the entities base position
        if self.params["rules"]["grid"]["use_grid"]:
            x_idx = entity_idx // self.params["rules"]["grid"]["num_y_agents"]
            y_idx = entity_idx % self.params["rules"]["grid"]["num_y_agents"]
            grid_dist_x = self.params["rules"]["grid"]["grid_dist_x"]
            grid_dist_y = self.params["rules"]["grid"]["grid_dist_y"]

            # add offset to the grid if necessary
            entity_base_pos = np.array([x_idx * grid_dist_x, y_idx * grid_dist_y])
            entity_base_pos[1] += self.params["rules"]["grid"]["grid_offset"] * (x_idx % 2 == 1)
            agent_bound_pos = entity_base_pos.copy()

            if is_landmark:
                entity_base_pos += np.array([
                    self.params["rules"]["grid"]["landmark_spawn_offset_x"],
                    self.params["rules"]["grid"]["landmark_spawn_offset_y"],
                ])
            else:
                entity_base_pos += np.array([
                    self.params["rules"]["grid"]["agent_spawn_offset_x"],
                    self.params["rules"]["grid"]["agent_spawn_offset_y"],
                ])

        elif (
                self.params["rules"]["manual"]["use_manual"]
                and is_landmark and "landmarks" in self.params["rules"]["manual"]
        ):
            entity_base_pos = np.array(self.params["rules"]["manual"]["landmarks"][entity_idx])
        elif (
                self.params["rules"]["manual"]["use_manual"]
                and not is_landmark and "agents" in self.params["rules"]["manual"]
        ):
            entity_base_pos = np.array(self.params["rules"]["manual"]["agents"][entity_idx])
        else:
            entity_base_pos = np.zeros(shape=(world.dim_p,))

        # Next, set the spawn radius
        if is_landmark:
            spawn_rad = self.params["rules"]["landmark_spawn_radius"]
        else:
            spawn_rad = self.params["rules"]["agent_spawn_radius"]

        # set initial pos for agents only
        if not is_landmark:
            world.agents[entity_idx].initial_pos = agent_bound_pos

        return entity_base_pos + np.random.uniform(-spawn_rad, +spawn_rad, world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if LocalSpreadScenario.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    # normalizes reward to be distance between 1 and 0
    def distance_to_reward(self, dist):
        normalizing_const = self.params["rules"]["reward"]["landmark_radius"]
        return normalizing_const / max(normalizing_const, dist)

    @staticmethod
    def is_collision(agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sum(np.square(delta_pos))
        dist_min = (agent1.size + agent2.size) ** 2
        return True if dist < dist_min else False

    @staticmethod
    def get_all_relative_positions(entities, entity2, include_self=False):
        return [entity.state.p_pos - entity2.state.p_pos for entity in entities if (entity != entity2 or include_self)]

    @staticmethod
    def get_closest_relative_position(entities, entity2, include_self=False):
        entities = [entity for entity in entities if (entity != entity2 or include_self)]
        closest_entity = min(entities, key=lambda x: np.linalg.norm(x.state.p_pos - entity2.state.p_pos))
        return [closest_entity.state.p_pos - entity2.state.p_pos]

    @staticmethod
    def get_range_overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    @staticmethod
    def in_range(val, range_var):
        return val >= range_var[0] and val <= range_var[1]

    @staticmethod
    def get_agent_ranges(agent):
        # if agent isnt bounded then it can live on all the range
        if not getattr(agent, "is_bound", False):
            return [-np.inf, np.inf], [-np.inf, np.inf]

        bottom_range = agent.initial_pos - agent.bound_dist
        top_range = agent.initial_pos + agent.bound_dist
        return [bottom_range[0], top_range[0]], [bottom_range[1], top_range[1]]

    def observation(self, curr_agent, world, graph_obj=None):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        # display firstly landmarks and then agents
        entity_mapping = [
            ("landmarks", world.landmarks),
            ("agents", world.agents),
        ]
        for entity_type, all_entities in entity_mapping:
            obs_type = self.params["rules"]["obs"][entity_type]

            obs_list = []
            if obs_type == "none":
                obs_list = []
            elif obs_type == "closest":
                if (
                        entity_type == "agents" and self.params["num_landmarks"] > 1
                        or entity_type == "landmarks" and self.params["num_landmarks"] > 0
                ):
                    obs_list = LocalSpreadScenario.get_closest_relative_position(all_entities, curr_agent)
            elif obs_type == "local":
                # local is the most difficult out of these, since not all agents are exactly homogenous, and this
                # creates a difference in the local observations. Not only is non-parameter sharing not implemented
                # here yet, 0-padding could be very bad. For example, in the agents case, this means that there is
                # constantly an agent that is going to collide with this agents

                # For this reason - in the agents case, we will first observe all of the immediate neighbors. Then,
                # we shall pad with the next closest agents locations, padded to max_deg of the graph.

                # In the landmarks case, we will give the k best landmarks. By default, k will be the max_deg of the
                # graph. We will start off with the landmarks that are reachable and then pad with the closest ones.

                if graph_obj is None:
                    raise Exception("Can't give local observation without dependency graph")

                if entity_type == "agents":
                    # First, add the agents that are neighbors. Should be from cache so no problem about order
                    # Note - nbrs_idx contains also agent.id - shouldnt be accounted for in #nbrs
                    nbrs_idx = graph_obj.get_nbrhood(agent_index=curr_agent.id)
                    nbrs_agents = [agent for agent in all_entities if agent.id in nbrs_idx]
                    nbrs_pos = LocalSpreadScenario.get_all_relative_positions(nbrs_agents, curr_agent)

                    # Pad the rest of the observation with the next closest agents
                    num_padding_agents = graph_obj.max_deg - len(nbrs_pos)
                    rest_agents = [agent for agent in all_entities if agent.id not in nbrs_idx]
                    rest_pos = LocalSpreadScenario.get_all_relative_positions(rest_agents, curr_agent)
                    rest_pos = sorted(rest_pos, key=lambda x: np.linalg.norm(x))
                    rest_pos = rest_pos[:num_padding_agents]

                    obs_list = nbrs_pos + rest_pos

                else:
                    # number of landmarks we should see
                    num_obs_landmarks = self.params["rules"]["obs"]["landmark_k"]
                    if num_obs_landmarks is None:
                        num_obs_landmarks = min(self.params["num_landmarks"], graph_obj.max_deg + 1)

                    # First, find the landmarks that are reachable for this agent
                    agent_ranges = LocalSpreadScenario.get_agent_ranges(curr_agent)
                    reachable_landmarks = [
                        lndmrk for lndmrk in all_entities
                        if (
                                LocalSpreadScenario.in_range(lndmrk.state.p_pos[0], agent_ranges[0])
                                and LocalSpreadScenario.in_range(lndmrk.state.p_pos[1], agent_ranges[1])
                        )
                    ]
                    reachable_pos = LocalSpreadScenario.get_all_relative_positions(reachable_landmarks, curr_agent)
                    reachable_pos = reachable_pos[:num_obs_landmarks]

                    # Pad the rest of the observation with the next closest agents
                    num_padding_landmarks = max(0, num_obs_landmarks - len(reachable_pos))
                    rest_landmarks = [landmark for landmark in all_entities if landmark not in reachable_landmarks]
                    rest_pos = LocalSpreadScenario.get_all_relative_positions(rest_landmarks, curr_agent)
                    rest_pos = sorted(rest_pos, key=lambda x: np.linalg.norm(x))
                    rest_pos = rest_pos[:num_padding_landmarks]

                    obs_list = reachable_pos + rest_pos

                    # trying to add the relative distance of the landmark
                    obs_list += [np.array([np.sqrt(np.sum(np.square(pos))) for pos in obs_list])]

            elif obs_type == "all":
                obs_list = LocalSpreadScenario.get_all_relative_positions(all_entities, curr_agent)
            else:
                raise Exception(f"Unsupported observation type: {obs_type}")

            # Re-eorder landmarks if necessary
            if entity_type == "landmarks" and self.params["rules"]["obs"]["obs_landmarks_order"] == "sorted":
                obs_list = sorted(obs_list, key=lambda x: np.linalg.norm(x))

            entity_pos += obs_list

        # Add a count of how many agents currently occupy the landmark
        if self.params["rules"]["obs"]["show_num_agents_on_landmark"]:
            # Compute the num of landmarks current agent occupies, and the total number of agent (including this one
            total_occupants = 0
            for landmark_idx, landmark in enumerate(world.landmarks):
                # compute all of the agents that are close enough
                agents = [agent for agent in world.agents if LocalSpreadScenario.is_collision(agent, landmark)]
                if curr_agent in agents:
                    if total_occupants == 0:
                        total_occupants = 1
                    total_occupants += len(agents) - 1
            entity_pos += [np.array([total_occupants])]

        return np.concatenate([curr_agent.state.p_vel] + [curr_agent.state.p_pos] + entity_pos)

    # returns the global state - landmarks and agents
    def state(self, world):
        state = []

        for agent in world.agents:
            state += [agent.state.p_vel] + [agent.state.p_pos]

        for landmark in world.landmarks:
            state += [landmark.state.p_pos]

        return np.concatenate(state)
