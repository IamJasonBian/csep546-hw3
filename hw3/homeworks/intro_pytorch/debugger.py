from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility
from torch import nn
from optimizers import SGDOptimizer

model = nn.Linear(20, 5)
torch.nn.init.constant_(model.weight, 1)
torch.nn.init.constant_(model.bias, 0.5)

optimizer = SGDOptimizer(model.parameters(), lr=1e-4)

# Dummy loss
loss = torch.sum(model(torch.ones((5, 20))))
loss.backward()

optimizer.step()

torch.testing.assert_allclose(
    model.weight, 0.9995 * torch.ones((5, 20)), rtol=1e-4, atol=1e-5
)
torch.testing.assert_allclose(
    model.bias, 0.4995 * torch.ones((5)), rtol=1e-4, atol=1e-5
)


