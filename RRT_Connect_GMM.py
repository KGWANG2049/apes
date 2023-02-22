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
from path_gen import *
import torch
from mp2d.scripts.manipulator import *
from mp2d.scripts.planning import *


class MyValidStateSampler(ob.ValidStateSampler):

    def __init__(self, si):
        super(MyValidStateSampler, self).__init__(si)
        self.name_ = "aaa"
        self.gmm = gmm_dist
        self.count = 0

    def sample(self, state):
        p = self.gmm.sample()
        p = p.cpu().numpy()
        state[0] = p[0]
        state[1] = p[1]
        self.count += 1
        print("self.count", self.count)
        return True

    def sampleUniform(self, state):
        p = self.gmm.sample()
        p = p.cpu().numpy()
        state[0] = p[0]
        state[1] = p[1]
        self.count += 1
        print("self.count", self.count)
        return True

    # def reset_counts(self):
    # self.count = 0
    def __call__(self, _):
        return self


# This function is needed, even when we can write a sampler like the one
# above, because we need to check path segments for validity

def isStateValid(state):
    obstacles = pl_req.obstacles
    is_valid = pl_env.manipulator.check_validity(state, obstacles)
    return is_valid


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
    sampler = MyValidStateSampler(si)

    # use my sampler
    si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(sampler))

    # create a planner for the defined space
    planner = og.RRTConnect(si)
    ss.setPlanner(planner)

    # attempt to solve the problem within 20 seconds of planning time
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
        _, ax = plt.subplots(1, 1)
        ax.scatter(samples[:, 0], samples[:, 1], s=1)
        px = [node[0] for node in mean]
        py = [node[1] for node in mean]
        ax.scatter(px, py, color="red", s=10)
        ax.scatter(req.start[0], req.start[1], color="green", s=60)
        ax.scatter(req.goal[0], req.goal[1], color="blue", s=60)
        obstacles_space = pl.get_obstacle_space(req)
        obst_space_x = [ns[0] for ns in obstacles_space]
        obst_space_y = [ns[1] for ns in obstacles_space]
        ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
        x = [ns[0] for ns in solution]
        y = [ns[1] for ns in solution]
        plt.plot(x, y)
        plt.show()
    else:
        print("No solution found")


if __name__ == '__main__':
    plan()
