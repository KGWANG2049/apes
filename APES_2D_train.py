import os
import time
import planning
import utilities
import numpy as np
import torch.nn.functional
import torch.optim as optim
from torch.optim import Adam
from collections import deque
from mp2d.scripts.manipulator import manipulator
from network_2d import APESCriticNet, APESGeneratorNet

REPLAY_MIN = 1000
REPLAY_MAX = 10000
SAVE_INTERVAL = 500
REPLAY_SAMPLE_SIZE = 50
start = time.time()
recent_steps = []
TARGET_ENTROPY = [-5.0]
LOG_ALPHA_INIT = [-3.0]
LR = 1e-4
LOG_ALPHA_MIN = -10.
LOG_ALPHA_MAX = 20.

dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = planning.Planning(ma)
cwd = os.getcwd()
easy_path = cwd + "/home/wangkaige/PycharmProjects/apes/mp2d/data/easy_pl_req_250_nodes.json "
dataset = utilities.load_planning_req_dataset(easy_path)

pl_req_file_name = "/home/wangkaige/PycharmProjects/apes/mp2d/data/easy_pl_req_250_nodes.json "
planning_requests = utilities.load_planning_req_dataset(pl_req_file_name)
replay_buffer = deque(maxlen=REPLAY_MAX)

if __name__ == '__main__':
    SV = np.array(2)
    GV = np.array(2)
    W = np.array(50)
    pl_req = planning_requests[1]
    req = dataset[1]
    pl.visualize_request(req)
    OC = np.array(np.shape(pl.get_occupancy_map(req)))

    for i in range(0, REPLAY_MAX):
        pl_req = planning_requests[i]
        SV = pl_req.start
        SV = torch.tensor(SV)
        GV = pl_req.goal
        GV = torch.tensor(GV)
        req = dataset[i]
        pl.visualize_request(req)
        OC = pl.get_occupancy_map(req)
        pl.visualize_occupancy_map(req, OC)
        pl.generate_graph_halton(150)
        pr = pl.search(req)
        VALUE_ESTIMATE = pr.checked_counts
        W = APESGeneratorNet(OC, SV, GV)
        experience = (OC, SV, GV, W, VALUE_ESTIMATE)
        replay_buffer.append(experience)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_model = APESGeneratorNet(OC, SV, GV).float().to(device)
    critic_model = APESCriticNet(OC, SV, GV, W).float().to(device)
    gen_model_optimizer = optim.Adam(gen_model.parameters(), lr=LR)
    critic_model_optimizer: Adam = optim.Adam(critic_model.parameters(), lr=LR)
    log_alpha = torch.tensor(LOG_ALPHA_INIT, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=LR)

    while True:

        if len(replay_buffer) < REPLAY_MIN:
            print('Waiting for minimum buffer size ... {}/{}'.format(len(replay_buffer), REPLAY_MIN))
            continue
        # labels = sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients, sampled_values
        sampled_evaluations = replay_buffer.sample(REPLAY_SAMPLE_SIZE)
        # data_labels = {sampled_evaluations[i]: labels[i] for i in range(len(sampled_evaluations))}
        sampled_oc = torch.stack([t[0] for t in sampled_evaluations])
        sampled_start_v = torch.stack([t[1] for t in sampled_evaluations])
        sampled_goal_v = torch.stack([t[2] for t in sampled_evaluations])
        sampled_coefficients = torch.stack([t[3] for t in sampled_evaluations])
        sampled_values = torch.stack([t[4] for t in sampled_evaluations])

        # update alpha
        critic_loss = -torch.distributions.Normal(
            *critic_model(sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients)) \
            .log_prob(sampled_values).sum(dim=-1)
        critic_model_optimizer.zero_grad()
        gen_model_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
        critic_model_optimizer.step()

        # Update Generator model.
        coefficients_entropy = gen_model.sample(sampled_oc, sampled_start_v, sampled_goal_v)
        (value, sdt) = critic_model(sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients)
        critic_model_optimizer.zero_grad()
        gen_model_optimizer.zero_grad()
        dual_terms = (log_alpha.exp().detach() * coefficients_entropy).sum(dim=-1)
        gen_objective = value - dual_terms
        gen_objective.sum().backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        gen_model_optimizer.step()

        # update alpha
        alpha_optim.zero_grad()
        alpha_loss = log_alpha * ((coefficients_entropy - torch.tensor(
            TARGET_ENTROPY, device=device, dtype=torch.float32)).detach())
        alpha_loss.sum().backward()
        with torch.no_grad():
            log_alpha.grad *= (((-log_alpha.grad >= 0) | (log_alpha >= LOG_ALPHA_MIN)) &
                               ((-log_alpha.grad < 0) | (log_alpha <= LOG_ALPHA_MAX))).float()  # ppo
        alpha_optim.step()

        print("critic loss:", critic_loss)
        print("generator loss:", gen_objective)
        print("Alpha loss:", alpha_loss)

    #  2d input run or not , git, visual
