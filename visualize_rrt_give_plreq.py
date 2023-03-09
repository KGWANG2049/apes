import torch
from Dist_gen import gmm_dist_generator
from apes.Rand_RRTCocnnect import random_plan
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


def visualize_rrt_path_with_request(pl_req):
    sv = pl_req.start
    sv = torch.tensor(sv).to(device)
    gv = pl_req.goal
    gv = torch.tensor(gv).to(device)
    oc = pl.get_occupancy_map(pl_req)
    oc = torch.tensor(oc)
    oc = oc.reshape([1, oc.shape[0], -1]).to(device)
    Gen_net = torch.load("/home/wang_ujsjo/Praktikum/apes/net.generator")
    w = Gen_net(oc, sv, gv).rsample().to(device)
    # print("WEIGHT", W)
    gmm_dist = gmm_dist_generator(w)
    gmm_solution = plan(pl_req, gmm_dist)[1]
    gmm_iteration_num = plan(pl_req, gmm_dist)[0]
    ran_solution = random_plan(pl_req)
    print("gmm_plan_num:", gmm_iteration_num)
    print("random_plan_num", ran_solution)
    samples = gmm_dist.sample([2000])
    _, ax = plt.subplots(1, 1)
    ax.scatter(samples[:, 0], samples[:, 1], s=3)
    px = [node[0] for node in mean]
    py = [node[1] for node in mean]
    ax.scatter(px, py, color="red", s=10)
    ax.scatter(pl_req.start[0], pl_req.start[1], color="green", s=60)
    ax.scatter(pl_req.goal[0], pl_req.goal[1], color="blue", s=60)
    obstacles_space = pl.get_obstacle_space(pl_req)
    obst_space_x = [ns[0] for ns in obstacles_space]
    obst_space_y = [ns[1] for ns in obstacles_space]
    ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
    x = [ns[0] for ns in gmm_solution]
    y = [ns[1] for ns in gmm_solution]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    pl_req = planning_requests[4615]
    visualize_rrt_path_with_request(pl_req)
