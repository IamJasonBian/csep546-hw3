# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = (1/torch.sqrt(d))
        gen = Uniform(-1 * alpha, alpha)

        self.w0 = Parameter(gen.rsample((h,d)))
        self.b0 = Parameter(gen.rsample((h,)))
        self.b1 = Parameter(gen.rsample((k,)))



    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        l0 = torch.matmul(self.w0, x) + self.b0
        l1 = torch.matmul(self.w1, relu(l0)) + self.b1

        return relu(l1)


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = (1 / torch.sqrt(d))

        gen = Uniform(-1*alpha, alpha)

        self.w0 = Parameter(gen.rsample((h0,d)))
        self.w1 = Parameter(gen.rsample((h1, h0)))
        self.w2 = Parameter(gen.rsample((k, h1)))
        self.b0 = Parameter(gen.rsample((h0,)))
        self.b1 = Parameter(gen.rsample((h1,)))
        self.b2 = Parameter(gen.rsample((k, )))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        l0 = torch.matmul(self.w0, x) + self.b0
        l1 = torch.matmul(self.w1, relu(l0)) + self.b1
        l2 = torch.matmul(self.w2, relu(l1)) + self.b2

        return relu(l2)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    last_accuracy = 0.0
    losses = []
    while last_accuracy <= 0.99:
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, tgt.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses)/len(losses)


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    Optimizier = Adam(model.parameters())
    model = F1(64,784,10)
    Dataloader = DataLoader(list(zip(x,y)), batch_size =1)
    
    epoch= 0
    while epoch < 1:
        Optimizier.zero_grad()
        loss = train(model, Optimizer, Dataloader)
        print(loss)


if __name__ == "__main__":
    main()
