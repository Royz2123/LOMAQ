import torch as th
from torch.autograd import grad
import torch.nn as nn
import numpy as np

# Create some dummy data.
u = th.tensor(np.array([[1.0, 2.0]]), requires_grad=True)

# Do some computations.
q1 = th.sum(2 * u[:, 0] - 5 * u[:, 1])
q2 = th.sum(u)
q_tot = q1 + q2

dq_du = th.autograd.grad(q_tot, u, grad_outputs=th.ones(q_tot.size()))[0]
loss = th.sum(th.relu(-dq_du))
# J = []
# for i in range(D):
#     out = torch.zeros(1,D)
#     out[0][i] = 1
#     j = torch.autograd.grad(y, x, create_graph=True,grad_outputs=out)[0]
#     J.append(j[0])
# J = torch.stack(J)
# dy_dz = J[:,-1]

# dq_du = jacobian(q_tot, u)
print(f'dq/du: {dq_du}')
print(f'loss: {loss}')
