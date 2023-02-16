
import torch
from copy import deepcopy
import torch.distributions as td
from mp2d.scripts.planning import Planning
from network_2d import APESGeneratorNet
from APES_2D_train import planning_requests
from torch.distributions.multivariate_normal import MultivariateNormal
from mp2d.scripts.manipulator import *
from mp2d.scripts.utilities import *
import networkx as nx
import numpy as np

dof = 2
SV = np.array(2)
GV = np.array(2)
W = np.array(50)
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
nodes_recording = {}
planning_requests_wo_obstacles = deepcopy(planning_requests)


# Find solutions in parallel using RRT


easy_path = "/home/wangkaige/Project/apes/easy_pl_req_250_nodes.json"
dataset = load_planning_req_dataset(easy_path)

# Define the weights for each multivariate Gaussian

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pl_req = planning_requests[461]
SV = pl_req.start
SV = torch.tensor(SV)
GV = pl_req.goal
GV = torch.tensor(GV)
req = pl_req
OC = pl.get_occupancy_map(req)
OC = torch.tensor(OC)
OC = OC.reshape([1, OC.shape[0], -1])
Gen_net = torch.load("/home/wangkaige/Project/apes/net.generator")
W = Gen_net(OC, SV, GV)
length_W = W.shape[1]
W = torch.tensor(W)
W = W.squeeze()
print(W)
# W = td.Categorical(torch.log(W).squeeze().exp())
# w = td.Categorical(log.w.squeeze().exp())
# print("cat", cat)
solution_list = []
path_w_idx = torch.load('/home/wangkaige/Project/apes/fixed_path_new')
print(path_w_idx)
mean = torch.load('/home/wangkaige/Project/apes/mean_new')
print(mean)

W_all = torch.zeros(path_w_idx[-1])
# print(W_all)
for i in range(length_W):
    # pl_env.visualize_path(req, path.solution_path)
    point_num = path_w_idx[i+1] - path_w_idx[i]
    W_line = torch.ones(point_num)*W[i]/point_num
    W_all[path_w_idx[i]: path_w_idx[i+1]] = W_line

W_sum = td.Categorical(torch.log(W_all).squeeze().exp())
# print("mean_sum:", mean)
#mean = torch.stack(mean).to(device)
#print(mean.shape)
cov = np.eye(2) * 0.1
cov = torch.tensor(cov)
# normal_dis = MultivariateNormal(mean, cov)  # Not sure about
dist = td.Independent(MultivariateNormal(mean, cov), 0)  # Not sure about 1
# print(dist)
gmm_dist = torch.distributions.MixtureSameFamily(W_sum, dist)
samples = gmm_dist.sample([3000])
_, ax = plt.subplots(1, 1)
ax.scatter(samples[:, 0], samples[:, 1], s=1)
px = [node[0] for node in mean]
py = [node[1] for node in mean]
# print(mean)
ax.scatter(px, py, color="red", s=10)
ax.scatter(req.start[0], req.start[1], color="green", s=60)
ax.scatter(req.goal[0], req.goal[1], color="blue", s=60)
ax.scatter(pl.visualize_obstacle_space(req)[0], pl.visualize_obstacle_space(req)[1], c="r", s=1)
plt.show()