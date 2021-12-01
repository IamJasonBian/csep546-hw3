import torch

from utils import problem
import numpy as np


class SGDOptimizer(torch.optim.Optimizer):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, params, lr: float) -> None:
        """Constructor for Stochastic Gradient Descent (SGD) Optimizer.

        Provided code contains call to super class, which will initialize paramaters properly (see step function docs).
        This class will only update the parameters provided to it, based on their (already calculated) gradients.

        Args:
            params: Parameters to update each step. You don't need to do anything with them.
                They are properly initialize through the super call.
            lr (float): Learning Rate of the gradient descent.

        Note:
            - YOU ARE NOT ALLOWED to use torch.optim.SGD in this class
            - While you are not allowed to use the class above, it might be extremely beneficial to look at it's code when implementing step function.
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        super().__init__(params, {"lr": lr})
        self.lr = lr

    @problem.tag("hw3-A")
    def step(self, closure=None):  # noqa: E251
        """
        Performs a step of gradient descent. You should loop through each parameter, and update it's value based on its gradient, value and learning rate.

        Args:
            closure (optional): Ignore this. We will not use in this class, but it is required for subclassing Optimizer.
                Defaults to None.

        Hint:
            - Superclass stores parameters in self.param_groups (you will have to discover in what format).
        """

        weight = self.param_groups[0]['params'][0]
        bias = self.param_groups[0]['params'][1]

        weight.data -= self.lr*weight.grad.data
        bias.data -= self.lr*bias.grad.data

