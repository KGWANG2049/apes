# CRITIC NETWORK
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.dirichlet import Dirichlet

# def LeakyReLU(x, x_max=1, hard_slope=1e-2):
# return (x <= x_max) * x + (x > x_max) * (x_max + hard_slope * (x - x_max))


class APESCriticNet(nn.Module):
    def __init__(self):
        super(APESCriticNet, self).__init__()
        self.mp = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.LeakyReLU = torch.nn.LeakyReLU()
        self.fc1 = nn.Linear(590, 200)  # 1728改为1728+start+goal
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 2)

    def forward(self, x, s, g, coefficients):
        # x = x.reshape((x.shape[0], -1, self.particle_size))
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = torch.flatten(x)
        s = torch.flatten(s)
        g = torch.flatten(g)
        coefficients = torch.flatten(coefficients)
        x = torch.cat((x, s, g, coefficients), 0)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)
        # x = x.cpu()
        # x = x.detach()
        # x = x.numpy().tolist

        return x


# GENERATOR NETWORK
class APESGeneratorNet(nn.Module):
    def __init__(self):
        super(APESGeneratorNet, self).__init__()
        self.mp = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.LeakyReLU = torch.nn.LeakyReLU()
        self.fc1 = nn.Linear(580, 200)  # 1728改为1728+start+goal
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x, s, g):
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = torch.flatten(x)
        s = torch.flatten(s)
        g = torch.flatten(g)
        x = torch.cat((x, s, g), 0)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)
        x = torch.exp(x)
        x = Dirichlet(x)
        return x

    # def sample(self, x, s, g):
    # coefficients_dist = self.forward(x, s, g)
    # coefficients_rs = coefficients_dist.rsample()
    # coefficients_entropy = -coefficients_dist.prob(coefficients_rs) * coefficients_dist.log_prob(coefficients_rs)

    # return coefficients_entropy
