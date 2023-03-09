import torch
from Dist_gen import gmm_dist_generator
from mp2d.scripts.planning import *
from mp2d.scripts.utilities import *
from experience import plan

mean = torch.load('/home/wang_ujsjo/Praktikum/apes/mean_new')
pl_req_file_name = "/home/wang_ujsjo/Praktikum/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)
links = [0.5, 0.5]
dof = 2
ma = manipulator(dof, links)
pl = Planning(ma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("...")


def get_sum_num_in_test_set():
    samples_sum = 0
    for i in range(500):
        req = planning_requests[i + 4001]
        sv = req.start
        sv = torch.tensor(sv).to(device)
        gv = req.goal
        gv = torch.tensor(gv).to(device)
        oc = pl.get_occupancy_map(req)
        oc = torch.tensor(oc)
        oc = oc.reshape([1, oc.shape[0], -1]).to(device)
        Gen_net = torch.load("/home/wang_ujsjo/Praktikum/apes/net.generator")
        w = Gen_net(oc, sv, gv).rsample().to(device)
        # print("WEIGHT", W)
        gmm_dist = gmm_dist_generator(w)
        samples = plan(req, gmm_dist)[0]
        samples_sum = samples + samples_sum
        return samples_sum


if __name__ == '__main__':
    get_sum_num_in_test_set()
