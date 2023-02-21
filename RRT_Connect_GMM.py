from mp2d.scripts.utilities import *
from mp2d.scripts.rrt_connect import RRTConnect
import matplotlib.pyplot as plt
import numpy as np

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), '/home/wangkaige/Project/mp2d/ompl-1.5.2/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
from time import sleep
from math import fabs
from path_gen import gmm_dist
import torch
from mp2d.scripts.manipulator import *
from mp2d.scripts.planning import *

pl_req_file_name = "/home/wangkaige/Project/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)
pl_req = planning_requests[666]
req = pl_req
dof = 2
links = [0.5, 0.5]
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
ma = manipulator(dof, links)
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)


class MyValidStateSampler(ob.ValidStateSampler):

    def __init__(self, si):
        super(MyValidStateSampler, self).__init__(si)
        self.name_ = "my sampler"

    def sample(self, state):
        p = gmm_dist.sample(pl_req)
        state[0] = p[0]
        state[1] = p[1]
        return True


# This function is needed, even when we can write a sampler like the one
# above, because we need to check path segments for validity

def isStateValid(state):
    obstacles = pl_req.obstacles
    is_valid = pl_env.manipulator.check_validity(state, obstacles)
    return is_valid

    # Let's pretend that the validity check is computationally relatively
    # expensive to emphasize the benefit of explicitly generating valid
    # samples
    # Valid states satisfy the following constraints:
    # -1<= x,y,z <=1
    # if .25 <= z <= .5, then |x|>.8 and |y|>.8


# return an instance of my sampler
def allocMyValidStateSampler(si):
    return MyValidStateSampler(si)


def plan():
    # construct the state space we are planning in
    space = ob.RealVectorStateSpace(2)

    # set the bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-4)
    bounds.setHigh(4)
    space.setBounds(bounds)

    # define a simple setup class
    ss = og.SimpleSetup(space)

    # set state validity checking for this space
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # create a start state
    start = ob.State(space)
    start[0] = pl_req.start[0]
    start[1] = pl_req.start[1]

    # create a goal state
    goal = ob.State(space)
    goal[0] = pl_req.goal[0]
    goal[1] = pl_req.goal[1]

    # set the start and goal states;
    ss.setStartAndGoalStates(start, goal)

    # set sampler (optional; the default is uniform sampling)
    si = ss.getSpaceInformation()

    # use my sampler
    si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocMyValidStateSampler))

    # create a planner for the defined space
    planner = og.RRTConnect(si)
    ss.setPlanner(planner)

    # attempt to solve the problem within ten seconds of planning time
    solved = ss.solve(20.0)
    solution_path = ss.getSolutionPath()

    ompl_solution = list(solution_path.getStates())
    solution = []
    for state in ompl_solution:
        np_state = np.zeros(2)
        np_state[0] = state[0]
        np_state[1] = state[1]
        solution.append(np_state)

    if solved:
        print("Found solution:")
        # print the path to screen

        print(ss.getSolutionPath())
        print(solution)
        x = [ns[0] for ns in solution]
        y = [ns[1] for ns in solution]
        plt.plot(x, y)
        plt.title('vaild_path')
        plt.show()




    else:
        print("No solution found")


if __name__ == '__main__':
    print("\nUsing my sampler:")
    plan()
