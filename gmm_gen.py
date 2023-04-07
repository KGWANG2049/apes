import torch
import torch.distributions as td
from mp2d.scripts.utilities import *
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cpu")
def gmm_dist_generator(W):

    length_W = W.shape[0]
    W = W.squeeze().to(device)
    solution_list = []
    path_w_idx = torch.load('/home/wang_ujsjo/Praktikum/apes/fixed_path_new')
    # print(path_w_idx)
    mean = torch.load('/home/wang_ujsjo/Praktikum/apes/mean_new')
    # print(mean)
    W_all = torch.zeros(path_w_idx[-1])
    # print(W_all)
    for i in range(length_W):
        # pl_env.visualize_path(req, path.solution_path)
        point_num = path_w_idx[i + 1] - path_w_idx[i]
        W_line = torch.ones(point_num) * W[i] / point_num
        W_all[path_w_idx[i]: path_w_idx[i + 1]] = W_line
    W_sum = td.Categorical(torch.log(W_all).squeeze().exp())
    cov = np.eye(2) * 0.1
    cov = torch.tensor(cov)
    # normal_dis = MultivariateNormal(mean, cov)  # Not sure about
    dist = td.Independent(MultivariateNormal(mean, cov), 0)  # Not sure about 1
    # print(dist)
    gmm_dist = torch.distributions.MixtureSameFamily(W_sum, dist)
    # samples = gmm_dist.sample([2000])
    return gmm_dist



