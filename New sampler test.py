from Dist_gen import gmm_dist_generator
from mp2d.scripts.planning import *
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

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), '/home/wang_ujsjo/Praktikum/mp2d/ompl-1.5.2/py'
                                                                 '-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

links = [0.5, 0.5]
dof = 2
ma = manipulator(dof, links)
pl = Planning(ma)
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)


class MyValidStateSampler(ob.ValidStateSampler):
    # increase a input gmm_dist than example
    def __init__(self, si, gmm_dist):
        super(MyValidStateSampler, self).__init__(si)
        self.name_ = "aaa"
        self.gmm = gmm_dist
        self.count = 0
        self.max_count = 0

    def sample(self, state):
        p = self.gmm.sample()
        p = p.cpu().numpy()
        state[0] = p[0]
        state[1] = p[1]
        self.count += 1
        if self.count > self.max_count:
            self.max_count = self.count
        print(self.max_count)
        return True

    def get_count_max(self):
        return self.max_count

    def __call__(self, _):
        return self


# This function is needed, even when we can write a sampler like the one
# above, because we need to check path segments for validity

def isStateValid(state, pl_req):
    obstacles = pl_req.obstacles
    is_valid = pl_env.manipulator.check_validity(state, obstacles)
    return is_valid


# example use this function to instance class , find instance in example "# use my sampler (130),but i direct
# instance: sampler = MyValidStateSampler(si, gmm_dist) (99)"

"""def allocMyValidStateSampler(si):
    return MyValidStateSampler(si)"""


def plan(pl_req, gmm_dist):
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
    sampler = MyValidStateSampler(si, gmm_dist)
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
    max_count = sampler.get_count_max()
    print("maxcount:", max_count)
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
        """_, ax = plt.subplots(1, 1)
        ax.scatter(pl_req.start[0], pl_req.start[1], color="green", s=60)
        ax.scatter(pl_req.goal[0], pl_req.goal[1], color="blue", s=60)
        obstacles_space = pl.get_obstacle_space(pl_req)
        obst_space_x = [ns[0] for ns in obstacles_space]
        obst_space_y = [ns[1] for ns in obstacles_space]
        ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
        x = [ns[0] for ns in solution]
        y = [ns[1] for ns in solution]
        plt.plot(x, y)
        plt.show()"""
    else:
        print("No solution found")

