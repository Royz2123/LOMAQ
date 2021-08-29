from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn

from controllers.basic_controller import BasicMAC


# This multi-agent controller DOESN'T share parameters between agents
class HetroMac(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_agents(self, input_shape):
        self.agent = nn.ModuleList([
            agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)
        ])

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent[0].init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        hidden_states = []
        agent_outs = []
        for i, agent in enumerate(self.agent):
            curr_agent_outs, curr_hidden_states = agent(agent_inputs[:, i], self.hidden_states[:, i])
            hidden_states.append(curr_hidden_states.unsqueeze(0))
            agent_outs.append(curr_agent_outs)

        agent_outs = th.stack(agent_outs, dim=1)
        self.hidden_states = th.stack(hidden_states, dim=1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
