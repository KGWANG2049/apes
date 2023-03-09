import torch
from matplotlib import pyplot as plt
from mp2d.scripts.planning import Planning
from Dist_gen import gmm_dist_generator
from mp2d.scripts.manipulator import manipulator
from mp2d.scripts.utilities import load_planning_req_dataset
from expercise import MyValidStateSampler
from Norm_RRTCocnnect import RandomValidStateSampler

dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
device = torch.device("cuda:0")
pl_req_file_name = "/home/wang_ujsjo/Praktikum/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)


def gmm_ran_compartor(start_num, goal_num):
    for i in range(start_num, goal_num):
        pl_req = planning_requests[i]
        oc = pl.get_occupancy_map(pl_req)
        oc = oc.reshape([1, oc.shape[0], -1])
        sv = pl_req.start
        gv = pl_req.goal
        oc = torch.tensor(oc).to(device)
        sv = torch.tensor(sv).to(device)
        gv = torch.tensor(gv).to(device)
        gen_model = torch.load("/home/wang_ujsjo/Praktikum/apes/net.generator")
        w = torch.tensor(gen_model(oc, sv, gv)).to(device)
        gmm_dist = gmm_dist_generator(w)
        sampler_gmm = MyValidStateSampler(gmm_dist, pl_req)
        sampler_ran = RandomValidStateSampler(pl_req)
        num_count_gmm = sampler_gmm.plan()
        num_count_gmm += num_count_gmm
        num_count_ran = sampler_ran
        num_count_ran += sampler_ran

        return num_count_ran, num_count_gmm


if __name__ == '__main__':
    ncr, ncg = gmm_ran_compartor(4000, 4300)
    _, ax = plt.subplots(1, 1)
    ax.scatter(0, ncr, color="bluegreen", s=60)
    ax.scatter(0, ncg, color="green", s=60)

    plt.show()
