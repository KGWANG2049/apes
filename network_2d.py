# CRITIC NETWORK
import torch
import math
import torch.nn as nn
import numpy as np


# def LeakyReLU(x, x_max=1, hard_slope=1e-2):
# return (x <= x_max) * x + (x > x_max) * (x_max + hard_slope * (x - x_max))

class APESCriticNet(nn.Module):
    def __init__(self, oc, start_v, goal_v, coefficients):
        super(APESCriticNet, self).__init__()
        self.oc = oc
        self.start_v = start_v
        self.goal_v = goal_v
        self.coefficients = coefficients
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.mp = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)

        self.fc1 = nn.Linear(1728, 512)  # 1728改1728+start+goal+50
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x, s, g, coefficients):
        # x = x.reshape((x.shape[0], -1, self.particle_size))
        self.conv1(self.oc)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = self.conv2(x)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = self.conv3(x)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = torch.flatten(x)
        x = torch.cat(
            ([x] if self.oc else [])
            + ([s] if self.start_v else [])
            + [g] if self.goal_v else []
            + [coefficients] if self.coefficients else [])
        x = self.fc1(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc2(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc3(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc4(x)
        x = x.cpu()
        x = x.detach()
        x = x.numpy().tolist

        return [x]


# GENERATOR NETWORK
class APESGeneratorNet(nn.Module):
    def __init__(self, oc, start_v, goal_v):
        super(APESGeneratorNet, self).__init__()
        self.oc = oc
        self.start_v = start_v
        self.goal_v = goal_v
        self.mp = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=(0, 0, 0))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)

        self.fc1 = nn.Linear(1728, 512)  # 1728改为1728+start+goal
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 50)

    def forward(self, x, s, g):
        self.conv1(self.oc)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = self.conv2(x)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = self.conv3(x)
        x = torch.nn.LeakyReLU(x)
        x = self.mp(x)

        x = torch.flatten(x)
        x = torch.cat(
            ([x] if self.oc else [])
            + ([s] if self.start_v else [])
            + [g] if self.goal_v else [])
        x = self.fc1(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc2(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc3(x)
        x = torch.nn.LeakyReLU(x)
        x = self.fc4(x)
        x = math.exp(x)

        return np.random.dirichlet([x], size=50)

    def sample(self, x, s, g):
        coefficients_dist = self.forward(x, s, g)
        coefficients_rs = coefficients_dist.rsample()
        coefficients_entropy = -coefficients_dist.prob(coefficients_rs) * coefficients_dist.log_prob(coefficients_rs)

        return coefficients_entropy
