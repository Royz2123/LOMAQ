import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
import numpy as np

from modules.mixers.local_qmix import LocalQMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer

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
            elif args.mixer == "local_qmix":
                self.mixer = LocalQMixer(args=args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

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
        # first, get relevant indices from the graph
        total_loss = 0
        for depth_info in self.depth_ls:
            indices = self.args.graph_obj.get_nbrhoods(depth=depth_info["depth_l"])

            for reward_index in range(self.args.n_agents):
                # Computing the individual LNilT (L_N^l_i(theta))
                # print(depth_info, agent_index)

                nbrhood = indices[reward_index]
                local_rewards = rewards[:, :, nbrhood]
                local_rewards = th.sum(local_rewards, dim=-1).reshape(local_rewards.shape[0], local_rewards.shape[1], 1)
                local_terminated = terminated.expand_as(local_rewards)

                chosen_action_local_qvals = chosen_action_qvals[:, :, nbrhood]
                target_max_local_qvals = target_max_qvals[:, :, nbrhood]
                chosen_action_local_qvals = th.sum(chosen_action_local_qvals, dim=-1, keepdims=True)
                target_max_local_qvals = th.sum(target_max_local_qvals, dim=-1, keepdims=True)

                # print("Agent: ", agent_index, ", L depth: ", nbrhood, ",  Neighbors: ", nbrhood)
                # print("Rewards Shape: ", rewards.shape, local_rewards.shape)
                # print("Terminated Shape: ", terminated.shape, local_terminated.shape)
                # print("Chosen Action Q Shape: ", chosen_action_qvals.shape, chosen_action_local_qvals.shape)
                # print("Target Max Q Shape: ", target_max_qvals.shape, target_max_local_qvals.shape)

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
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

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

        # Mix
        if self.mixer is not None:
            # Since we want to optimize the bellman equation, and the target refers to the
            # next state, we do this 1: , :-1 trim to the state batch
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

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
        loss = loss_func(self, rewards, terminated, mask, target_max_qvals, chosen_action_qvals)

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
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                      t_env)
            self.log_stats_t = t_env

            self.args.exp_logger.save_learner_data(
                learner_name=self.args.name,
                learner_data={
                    "t_env": t_env,
                    "loss": loss.item(),
                    # "td_error": masked_td_error.abs().sum().item() / mask_elems,
                    "grad_norm": grad_norm.clone().detach().numpy(),
                    "q_taken_mean": (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
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
