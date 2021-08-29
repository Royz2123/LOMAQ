import cv2

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time

import wandb
import imageio


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.n_episodes = 0

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_last_upload_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.n_episodes += 1
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        last_test_episode = test_mode and (len(self.test_returns) == self.args.test_nepisode - 1)
        upload_vid_episode = (
                last_test_episode
                and hasattr(self.env, "viewer")
                and (self.t_env - self.log_last_upload_t) >= self.args.save_vid_interval
        )

        img_array = []
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            try:
                self.batch.update(pre_transition_data, ts=self.t)
            except Exception as e:
                print("PROBLEM IN UPDATE")
                print(e)
                print("\nDATA:\n")
                print(pre_transition_data)
                print("\n\n")
                raise e

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])

            # Here we added a modification for individually observed rewards
            total_reward = np.sum(np.array(reward))
            episode_return += total_reward

            # Save the step
            # if not test_mode:
            #     self.env.save_step(t_env=(self.t_env + self.t), step_reward=total_reward)

            if test_mode:
                # mode = "human" if self.args.human_mode else "rgb_array"
                self.env.render()

                if upload_vid_episode:
                    img = self.env.viewer.render(return_rgb_array=True)
                    img_array.append(img)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        # print(f"Episode Return {episode_return}")

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

            # Save the episode
            # self.env.save_episode(t_env=self.t_env, episode_reward=episode_return)

        cur_returns.append(episode_return)

        if last_test_episode:
            self._log(cur_returns, cur_stats, log_prefix)

        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

            self.args.exp_logger.save_env_data(
                learner_name=self.args.name,
                env_data={
                    "t_env": self.t_env,
                    "episode": self.n_episodes,
                    "epsilon": self.mac.action_selector.epsilon,
                    "episode_reward": episode_return,
                }
            )

        # Should we save a recording of this episode and try to upload?
        if upload_vid_episode:
            with imageio.get_writer("test.gif", mode="I") as writer:
                for idx, frame in enumerate(img_array):
                    writer.append_data(frame)
            self.logger.log_stat("test_vid_gif", "test.gif", self.t_env, video=True)
            self.log_last_upload_t = self.t_env


        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


