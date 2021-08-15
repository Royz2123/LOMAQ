import numpy as np
from envs.multi_particle.multiagent_particle_envs.multiagent.core import World, Agent, Landmark
from envs.multi_particle.multiagent_particle_envs.multiagent.scenario import BaseScenario


class LocalSpreadScenario(BaseScenario):
    def __init__(self, params):
        self.params = params

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 0
        world.collaborative = True
        world.reward_thresh = self.params["rules"]["landmark_radius"]

        # add agents
        world.agents = [Agent() for _ in range(self.params["num_agents"])]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.params["num_landmarks"])]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.params["rules"]["landmark_radius"]

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

    def generate_location(self, world, entity_idx, is_landmark=False):
        if self.params["rules"]["grid"]["use_grid"]:
            x_idx = entity_idx // self.params["rules"]["grid"]["num_y_agents"]
            y_idx = entity_idx % self.params["rules"]["grid"]["num_y_agents"]
            grid_dist_x = self.params["rules"]["grid"]["grid_dist_x"]
            grid_dist_y = self.params["rules"]["grid"]["grid_dist_y"]

            # add offset to the grid if necessary
            base_pos = np.array([x_idx * grid_dist_x, y_idx * grid_dist_y])
            base_pos[1] += self.params["rules"]["grid"]["grid_offset"] * (x_idx % 2 == 1)

            if is_landmark:
                base_pos += np.array([
                    self.params["rules"]["grid"]["landmark_spawn_offset_x"],
                    self.params["rules"]["grid"]["landmark_spawn_offset_y"],
                ])
                spawn_rad = self.params["rules"]["grid"]["landmark_spawn_radius"]
            else:
                spawn_rad = self.params["rules"]["grid"]["agent_spawn_radius"]

        else:
            base_pos = np.zeros(shape=(world.dim_p,))
            spawn_rad = 1

        # set initial pos for agents only
        if not is_landmark:
            world.agents[entity_idx].initial_pos = base_pos

        return base_pos + np.random.uniform(-spawn_rad, +spawn_rad, world.dim_p)

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
        normalizing_const = self.params["rules"]["landmark_radius"]
        return normalizing_const / (normalizing_const + dist)

    # compute the global reward
    # We reveal here a possible variant for local ewards that should be learnt by the decomposition
    def compute_all_rewards(self, world):
        rewards = np.zeros(self.params["num_agents"])

        # set all landmark rewards
        for landmark in world.landmarks:
            # compute all the distances
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]

            closest_agent = np.argmin(dists)
            closest_dist = min(dists)

            # add the main reward
            if (
                    self.params["rules"]["binary_reward"]
                    and LocalSpreadScenario.is_collision(world.agents[closest_agent], landmark)
            ):
                rewards[closest_agent] += 1.0
            else:
                rewards[closest_agent] += self.distance_to_reward(closest_dist)

            # add the bonus reward for every agent
            rewards += self.params["rules"]["bonus_coeff"] * self.distance_to_reward(np.array(dists))

        # set collision rewards
        if self.params["rules"]["collisions_reward"]:
            for i in range(self.params["num_agents"]):
                for j in range(i + 1, self.params["num_agents"]):
                    # punish them both equally
                    if LocalSpreadScenario.is_collision(world.agents[i], world.agents[j]):
                        rewards[i] -= 0.5
                        rewards[j] -= 0.5
        return rewards

    @staticmethod
    def is_collision(agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
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

    def observation(self, agent, world, graph_obj=None):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        # display landmarks and agents
        entity_mapping = {
            "agents": world.agents,
            "landmarks": world.landmarks
        }
        for entity_type, all_entities in entity_mapping.items():
            obs_type = self.params["rules"]["obs"][entity_type]

            obs_list = []
            if obs_type == "none":
                obs_list = []
            elif obs_type == "closest":
                if (
                    entity_type == "agents" and self.params["num_landmarks"] > 1
                    or entity_type == "landmarks" and self.params["num_landmarks"] > 0
                ):
                    obs_list = LocalSpreadScenario.get_closest_relative_position(all_entities, agent)
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
                    nbrs_idx = graph_obj.get_nbrhood(agent_index=agent.id)
                    nbrs_agents = [agent for agent in all_entities if agent.id in nbrs_idx]
                    nbrs_pos = LocalSpreadScenario.get_all_relative_positions(nbrs_agents, agent)

                    # Pad the rest of the observation with the next closest agents
                    num_padding_agents = graph_obj.max_deg - len(nbrs_pos)
                    rest_agents = [agent for agent in all_entities if agent.id not in nbrs_idx]
                    rest_pos = LocalSpreadScenario.get_all_relative_positions(rest_agents, agent)
                    rest_pos = sorted(rest_pos, key=lambda x: np.linalg.norm(x))
                    rest_pos = rest_pos[:num_padding_agents]

                    obs_list = nbrs_pos + rest_pos

                else:
                    # number of landmarks we should see
                    num_obs_landmarks = self.params["rules"]["obs"]["landmark_k"]
                    if num_obs_landmarks is None:
                        num_obs_landmarks = graph_obj.max_deg

                    # First, find the landmarks that are reachable for this agent
                    agent_ranges = LocalSpreadScenario.get_agent_ranges(agent)
                    reachable_landmarks = [
                        lndmrk for lndmrk in all_entities
                        if (
                            LocalSpreadScenario.in_range(lndmrk.state.p_pos[0], agent_ranges[0])
                            and LocalSpreadScenario.in_range(lndmrk.state.p_pos[1], agent_ranges[1])
                        )
                    ]
                    reachable_pos = LocalSpreadScenario.get_all_relative_positions(reachable_landmarks, agent)
                    reachable_pos = reachable_pos[:num_obs_landmarks]

                    # Pad the rest of the observation with the next closest agents
                    num_padding_landmarks = max(0, num_obs_landmarks - len(reachable_pos))
                    rest_landmarks = [landmark for landmark in all_entities if landmark not in reachable_landmarks]
                    rest_pos = LocalSpreadScenario.get_all_relative_positions(rest_landmarks, agent)
                    rest_pos = sorted(rest_pos, key=lambda x: np.linalg.norm(x))
                    rest_pos = rest_pos[:num_padding_landmarks]

                    obs_list = reachable_pos + rest_pos

            elif obs_type == "all":
                obs_list = LocalSpreadScenario.get_all_relative_positions(all_entities, agent)
            else:
                raise Exception(f"Unsupported observation type: {obs_type}")

            entity_pos += obs_list

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

    # returns the global state - landmarks and agents
    def state(self, world):
        state = []

        for agent in world.agents:
            state += [agent.state.p_vel] + [agent.state.p_pos]

        for landmark in world.landmarks:
            state += [landmark.state.p_pos]

        return np.concatenate(state)
