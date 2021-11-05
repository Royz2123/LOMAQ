import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
import numpy as np

from modules.mixers.lomaq import LocalQMixer
from modules.mixers.gcn.gcn_lomaq import GraphQMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.graphmix import GraphMixer

from reward_decomposition.decomposer import RewardDecomposer
import reward_decomposition.decompose as decompose


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        # Observes rewards locally?
        self.args.local_observer = getattr(self.args, "local_observer", False)

        if hasattr(self.args, "l_params"):
            self.depth_ls = [{"depth_l": self.args.l_params["start_depth_l"], "weight": 1}]
        else:
            self.depth_ls = [{"depth_l": 0, "weight": 1}]

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "lomaq":
                if getattr(args, "use_gcn", False):
                    self.mixer = GraphQMixer(args=args)
                else:
                    self.mixer = LocalQMixer(args=args)
            elif args.mixer == "graphmix":
                self.mixer = GraphMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

            self.logger.log_model(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # find the #parameters of the learner
        num_params = 0
        for param_group in self.params:
            model_parameters = filter(lambda p: p.requires_grad, param_group)
            num_params += sum([np.prod(p.size()) for p in model_parameters])

        # log #parameters
        self.args.exp_logger.runtime_data["total parameters"][args.name] = num_params.tolist()
        self.args.exp_logger.log_runtime_data()

    # Normal L2 loss, take mean over actual data, global observation!
    def compute_global_loss(self, rewards, terminated, mask, target_max_qvals, chosen_action_qvals):
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        return (masked_td_error ** 2).sum() / mask.sum()

    # My L2 loss, local observation!
    def compute_local_loss(self, rewards, terminated, mask, target_max_qvals, chosen_action_qvals):
        # This is the main contribution of my work - localized losses for scalable problems
        # An exact explanation of the loss function is available under documentation, Status Update - Week 5

        # Our goal is to be able to compute LNilT (L_N^l_i(theta))

        # Now we assume that target_max_qvals and chosen_action_qvals is the individual Q function Q_i,
        # where the index of each Q is the agent that that reward belongs to. In that case, we need to sum
        # Q_i's and rewards by looking at that neighborhood in the graph

        # for every depth LNIT
        # first, get relevant reward neighborhoods
        reward_nbrhoods = self.args.graph_obj.get_nbrhoods(depth=getattr(self.args, "reward_depth_k", 0))
        weights = th.FloatTensor([1 / len(nbrhood) for nbrhood in reward_nbrhoods]).to(device=self.args.device)
        weights = weights.expand_as(rewards)
        glocal_rewards = rewards * weights

        # Now get relevant l's, can be with partition also but currently not hooked up to l, NOT USING THIS FEATURE!
        loss_nbrhoods = self.args.graph_obj.get_nbrhoods(depth=0)

        total_loss = 0
        for agent_index in range(self.args.n_agents):
            local_rewards = glocal_rewards[:, :, reward_nbrhoods[agent_index]]
            local_rewards = th.sum(local_rewards, dim=-1, keepdims=True)
            local_terminated = terminated.expand_as(local_rewards)

            chosen_action_local_qvals = chosen_action_qvals[:, :, loss_nbrhoods[agent_index]]
            target_max_local_qvals = target_max_qvals[:, :, loss_nbrhoods[agent_index]]
            chosen_action_local_qvals = th.sum(chosen_action_local_qvals, dim=-1, keepdims=True)
            target_max_local_qvals = th.sum(target_max_local_qvals, dim=-1, keepdims=True)

            # Calculate 1-step Q-Learning targets
            targets = local_rewards + self.args.gamma * (1 - local_terminated) * target_max_local_qvals

            # Td-error
            td_error = (chosen_action_local_qvals - targets.detach())

            # 0-out the targets that came from padded data
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            total_loss += (masked_td_error ** 2).sum() / mask.sum()
        return total_loss

    def update_l_params(self, t_env):
        # Maybe l requires no change becoause attribute doesnt exist
        if not hasattr(self.args, "l_params"):
            return

        params = self.args.l_params

        # we want to update the weights. First find in what interval update we are and where we are in it
        interval_index = t_env // params["update_interval_t"]
        interval_step = t_env % params["update_interval_t"]

        # compute current l
        if params["growth_type"] == "constant":
            current_l = params["start_depth_l"]
            next_l = current_l
        elif params["growth_type"] == "linear":
            current_l = params["start_depth_l"] + params["growth_jump"] * interval_index
            next_l = current_l + params["growth_jump"]
        elif params["growth_type"] == "exponent":
            current_l = params["start_depth_l"] * (params["growth_jump"] ** interval_index)
            next_l = current_l * params["growth_jump"]
        else:
            raise Exception("Error when updating l - Growth type not found")

        # cap both current l's with the total depth
        current_l = min(current_l, self.args.n_agents)
        next_l = min(next_l, self.args.n_agents)

        # update the l data based on the update type
        if params["update_type"] == "hard":
            self.depth_ls = [{"depth_l": current_l, "weight": 1}]
        elif params["update_type"] == "soft":
            curr_weight = 1 - interval_step / params["update_interval_t"]
            self.depth_ls = [
                {"depth_l": current_l, "weight": curr_weight},
                {"depth_l": next_l, "weight": 1 - curr_weight},
            ]
        else:
            raise Exception("Error when updating l - Update type not found")

    def build_rewards(self, batch):
        # Lets assume that the env gives us the rewards in local form
        # Sum up local rewards in last dimension (shouldn't affect other envs but worry about that later)
        global_rewards = th.sum(batch["reward"], dim=-1, keepdims=True)[:, :-1]
        local_rewards = batch["reward"][:, :-1]

        # We assume the rewards are valid (status=True) unless the decompose
        # function fails, and that we should use all of them (reward mask)
        reward_mask = None
        status = True

        # try decomposing global reward if necessary, and disregard the local rewards from the enivroment!
        # This is exactly where we assume the local rewards are unobservable directly / not computed by the env,
        # and where we compute them ourselves.
        if self.args.decompose_reward:
            status, reward_mask, local_rewards = decompose.decompose(self.args.reward_decomposer, batch)

        if self.args.local_observer:
            return status, reward_mask, local_rewards
        return status, reward_mask, global_rewards

    def compute_gradient(self, utilities, qvals):
        dq_du = th.autograd.grad(
            qvals,
            utilities,
            grad_outputs=th.ones(qvals.size()).to(self.args.device),
            create_graph=True,
        )[0]
        return dq_du

    def punish_negative_gradients(self, utilities, qvals):
        return th.sum(th.relu(-self.compute_gradient(utilities, qvals)))

    def compute_regularization(self, utilities, chosen_action_qvals, t_env):
        # Compute gradient based on p_enforce
        p_enforce = getattr(self.args, "p_enforce", "singletons")
        if p_enforce == "singletons":
            reg_loss = 0
            for i in range(chosen_action_qvals.shape[2]):
                q_vals = th.reshape(chosen_action_qvals[:, :, i], shape=(*chosen_action_qvals.shape[:2], 1))
                reg_loss += self.punish_negative_gradients(utilities, q_vals)
        elif p_enforce == "full":
            total_q = th.sum(chosen_action_qvals, dim=2)
            reg_loss = self.punish_negative_gradients(utilities, total_q)
        else:
            # TODO: add support for enforcing a general partition
            raise Exception(f"Unsupported partition for monotonicity: {p_enforce}")

        # We want regularization to be invariant to batch size, episode length, and num agents
        normalization = th.prod(th.tensor(chosen_action_qvals.shape), dim=0)
        reg_loss = reg_loss / normalization
        return reg_loss

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Build the rewards based on the scheme (local/global, decompose or not ...)
        status, reward_mask, rewards = self.build_rewards(batch)

        # Check if reward decomposition has failed
        if not status:
            # print("Decomposition failed for current batch, disregarding...")
            return
        # print("Successfully decomposed the reward function, training Q functions")

        # if given a reward mask, use in order to not use on all data
        if reward_mask is not None:
            mask = th.logical_and(mask, reward_mask)

        # Calculate estimated Q-Values
        mac_out = []
        hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        hidden_states = th.stack(hidden_states, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = th.stack(target_hidden_states[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Record utility values
        utilities = chosen_action_qvals

        # Mix
        if self.args.mixer == 'graphmix':
            chosen_action_qvals_peragent = chosen_action_qvals.clone()
            target_max_qvals_peragent = target_max_qvals.detach()

            chosen_action_qvals, local_rewards, alive_agents_mask = self.mixer(chosen_action_qvals,
                                                                               batch["state"][:, :-1],
                                                                               agent_obs=batch["obs"][:, :-1],
                                                                               team_rewards=rewards,
                                                                               hidden_states=hidden_states[:, :-1]
                                                                               )
            chosen_output_qvals = chosen_action_qvals.clone()

            target_max_qvals = self.target_mixer(target_max_qvals,
                                                 batch["state"][:, 1:],
                                                 agent_obs=batch["obs"][:, 1:],
                                                 hidden_states=target_hidden_states
                                                 )[0]

            ## Global loss
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            global_loss = (masked_td_error ** 2).sum() / mask.sum()

            ## Local losses
            # Calculate 1-step Q-Learning targets
            local_targets = local_rewards + self.args.gamma * (1 - terminated).repeat(1, 1, self.args.n_agents) \
                            * target_max_qvals_peragent

            # Td-error
            local_td_error = (chosen_action_qvals_peragent - local_targets)
            local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask

            # 0-out the targets that came from padded data
            local_masked_td_error = local_td_error * local_mask

            # Normal L2 loss, take mean over actual data
            local_loss = (local_masked_td_error ** 2).sum() / mask.sum()

            # total loss
            lambda_local = self.args.lambda_local
            loss = global_loss + lambda_local * local_loss

        else:
            if self.mixer is not None:
                # Since we want to optimize the bellman equation, and the target refers to the
                # next state, we do this 1: , :-1 trim to the state batch
                chosen_output_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], obs=batch["obs"][:, :-1])
                target_output_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], obs=batch["obs"][:, 1:])
            else:
                chosen_output_qvals = chosen_action_qvals
                target_output_qvals = target_max_qvals

            # Shape debugging purposes
            # print(f"Target Max qvals: {target_max_qvals.shape}")
            # print(f"Chosen Action qvals: {chosen_action_qvals.shape}")
            # print(f"Rewards: {rewards.shape}")
            # print(f"Mask: {mask.shape}")

            # Compute Loss
            if self.args.local_observer:
                loss_func = QLearner.compute_local_loss
            else:
                loss_func = QLearner.compute_global_loss
            loss = loss_func(self, rewards, terminated, mask, target_output_qvals, chosen_output_qvals)

            # Add regularization if necessary
            if getattr(self.args, "monotonicity_method", "weights") == "regularization":
                # If sample is set to true, we don't use the given utilities, rather we sample from the bounding box that
                # they imply. This creates a more uniform punishment for the gradient of Q by U
                if getattr(self.args, "sample_utilities", False):
                    n_agents = utilities.shape[-1]
                    copied_utilities = utilities.cpu().detach().numpy()
                    flattened_utilities = np.reshape(copied_utilities, (-1, n_agents))
                    sampled_utilities = th.tensor(np.random.uniform(
                        low=[np.min(flattened_utilities[:, i]) for i in range(n_agents)],
                        high=[np.max(flattened_utilities[:, i]) for i in range(n_agents)],
                        size=copied_utilities.shape
                    ), requires_grad=True, device=self.args.device).float()
                    sampled_q_vals = self.mixer(sampled_utilities, batch["state"][:, :-1], obs=batch["obs"][:, :-1])
                    reg_loss = self.compute_regularization(sampled_utilities, sampled_q_vals, t_env)
                else:
                    reg_loss = self.compute_regularization(utilities, chosen_output_qvals, t_env)

                coeff = self.args.monotonicity_coeff
                self.logger.log_stat("regularizing_loss", reg_loss.item(), t_env)

                # Comupte the total loss
                loss = coeff * reg_loss + (1 - coeff) * loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.args.local_observer:
            self.update_l_params(t_env)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.clone().cpu().detach().numpy(), t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_output_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                      t_env)
            self.log_stats_t = t_env

            # Visualize Q values if necessary (mostly for the payoff matrix enviroment)
            if getattr(self.args, "display_q_values", False):
                print(f"Q Values for {self.args.run_name}")
                mac_output = mac_out[:1, :1]
                q_tot = []
                for agent1_act, agent2_act in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    utilities = th.stack([mac_output[:, :, 0, agent1_act], mac_output[:, :, 1, agent2_act]], dim=2)
                    print(f"U1, U2: for a1={agent1_act}, a2={agent2_act}:\t{utilities}")
                    q_values = self.mixer(utilities, batch["state"][:1, :1], obs=None)
                    print(f"Q1, Q2: for a1={agent1_act}, a2={agent2_act}:\t{q_values}")
                    q_tot.append(th.sum(q_values))
                print(f"{q_tot[0]}\t{q_tot[1]}\n{q_tot[2]}\t{q_tot[3]}")

            # My own local exp logger
            self.args.exp_logger.save_learner_data(
                learner_name=self.args.name,
                learner_data={
                    "t_env": t_env,
                    "loss": loss.item(),
                    # "td_error": masked_td_error.abs().sum().item() / mask_elems,
                    "grad_norm": grad_norm.clone().cpu().detach().numpy(),
                    "q_taken_mean": (chosen_output_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                    # "target_mean": (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
                }
            )

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
