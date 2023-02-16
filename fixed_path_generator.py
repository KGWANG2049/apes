import numpy as np
import torch

from mp2d.scripts.rrt_connect import RRTConnect
from mp2d.scripts.utilities import PlanningResult, load_planning_req_dataset
from mp2d.scripts.planning import Planning
from mp2d.scripts.manipulator import *

dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
fixed_path1 = 0
size = []
fixed_path_idx = [0]
mean = []
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)


def plan_with_rrt_connect(pl_req, ):
    pl_alg = RRTConnect(pl_env)
    solved = pl_alg.planning(pl_req)
    pr_rrt_connect = PlanningResult()
    if solved is False:
        pr_rrt_connect.has_solution = False
        pr_rrt_connect.solution_path = []
        return pr_rrt_connect

    solution_path = pl_alg.get_solution()
    is_path_valid = pl_env.check_path_validity(solution_path, pl_req.obstacles)
    if not is_path_valid:
        pr_rrt_connect.has_solution = False
        pr_rrt_connect.solution_path = []
        return pr_rrt_connect

    pr_rrt_connect.has_solution = True
    pr_rrt_connect.solution_path = solution_path
    return pr_rrt_connect


for i in range(50):
    pl_req_file_name = "/home/wangkaige/Project/apes/easy_pl_req_250_nodes.json"
    planning_requests = load_planning_req_dataset(pl_req_file_name)
    pl_req = planning_requests[i+300]
    path = plan_with_rrt_connect(pl_req)
    solution = np.array(path.solution_path)
    solution = torch.tensor(solution)
    print(solution.shape)
    size = solution.shape[0]
    print(size)
    fixed_path1 += size
    # solution = torch.tensor(solution)
    fixed_path_idx.append(fixed_path1)
    mean.append(solution)
mean = torch.cat(mean, dim=0)

print(mean.shape)


torch.save(fixed_path_idx, "fixed_path_new")
torch.save(mean, "mean_new")




