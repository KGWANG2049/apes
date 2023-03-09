
import torch.optim as optim
from torch.optim import Adam
from collections import deque
from tensorboardX import SummaryWriter
from torch.distributions import Dirichlet
from mp2d.scripts.planning import Planning
from mp2d.scripts.manipulator import manipulator
from apes.network_2d import APESCriticNet, APESGeneratorNet
from mp2d.scripts.utilities import load_planning_req_dataset
import torch.distributions as dist
import os
import random
import time
import numpy as np

import torch.nn.functional
BATCH_MAX = 16
REPLAY_SAMPLE_SIZE = 8
TRAIN_T = 100
SAVE_INTERVAL = 50
start = time.time()
recent_steps = []
TARGET_ENTROPY = [-20.0]
LOG_ALPHA_INIT = [0.0]
LR = 1e-3
LOG_ALPHA_MIN = -10.
LOG_ALPHA_MAX = 20.
cwd = os.getcwd()
easy_path = "/home/wang_ujsjo/Praktikum/apes/easy_pl_req_250_nodes.json"
dataset = load_planning_req_dataset(easy_path)
dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
pl_req_file_name = "/home/wang_ujsjo/Praktikum/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)
replay_buffer = deque(maxlen=BATCH_MAX)
writer = SummaryWriter()

if __name__ == '__main__':

    SV = np.array(2)
    GV = np.array(2)
    W = np.array(50)
    data = np.array([])
    pl_req = planning_requests[1]
    req = dataset[1]
    OC = np.array(np.shape(pl.get_occupancy_map(req)))
    device = torch.device("cuda:0")
    gen_model = APESGeneratorNet().double().to(device)
    gen_model.eval()
    critic_model = APESCriticNet().double().to(device)
    critic_model.eval()
    gen_model_optimizer = optim.Adam(gen_model.parameters(), lr=LR)
    critic_model_optimizer: Adam = optim.Adam(critic_model.parameters(), lr=LR)
    log_alpha = torch.tensor(LOG_ALPHA_INIT, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=LR)
    writer = SummaryWriter("LOSS_FUNCTION")

    for e in range(60):
        replay_buffer.clear()

        for i in range(0, BATCH_MAX):
            req = dataset[i]
            OC = pl.get_occupancy_map(req)
            OC = OC.reshape([1, OC.shape[0], -1])
            pl_req = planning_requests[i]
            SV = pl_req.start
            GV = pl_req.goal
            pl.generate_graph_halton(150)
            pr = pl.search(req)
            VALUE_ESTIMATE = pr.checked_counts
            VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE)
            # VALUE_ESTIMATE = plan().max_count
            # VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE)
            print("value111111", VALUE_ESTIMATE)
            OC = torch.tensor(OC).to(device)
            SV = torch.tensor(SV).to(device)
            GV = torch.tensor(GV).to(device)
            W = torch.tensor(gen_model(OC, SV, GV)).to(device)
            experience = ([OC, SV, GV, W, VALUE_ESTIMATE])
            replay_buffer.append(experience)
            print('Waiting for buffer size ... {}/{}'.format(len(replay_buffer), BATCH_MAX))

        for i in range(TRAIN_T):
            # labels = sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients, sampled_values
            sampled_evaluations = random.sample(replay_buffer, REPLAY_SAMPLE_SIZE)
            # data_labels = {sampled_evaluations[i]: labels[i] for i in range(len(sampled_evaluations))}
            sampled_oc = torch.stack([t[0] for t in sampled_evaluations])
            # print("OC:", sampled_oc, sampled_oc.shape)
            sampled_start_v = torch.stack([t[1] for t in sampled_evaluations])
            # print("start_v:", sampled_start_v, sampled_start_v.shape)
            sampled_goal_v = torch.stack([t[2] for t in sampled_evaluations])
            # print("goal_v:", sampled_goal_v, sampled_goal_v.shape)
            sampled_coefficients = torch.stack([t[3] for t in sampled_evaluations])
            # print("sampled_coefficient:", sampled_coefficients, sampled_coefficients.shape)
            sampled_values = torch.stack([t[4] for t in sampled_evaluations])
            # print("value:", sampled_values, sampled_values.shape)

            # update Cir
            critic_loss = 0

            for j in range(REPLAY_SAMPLE_SIZE):
                mean, std = critic_model(sampled_oc[j], sampled_start_v[j], sampled_goal_v[j], sampled_coefficients[j])
                std = torch.exp(std)
                # print("mean:", mean, "std:", std)
                priori_pro = dist.Normal(mean, std)
                # print("posterior:", priori_pro)
                posterior_prob = priori_pro.log_prob(sampled_values[j])
                # print("posterior_prob:", posterior_prob)
                # print("test", priori_pro.log_prob(a).exp())
                critic_loss = critic_loss + (-posterior_prob)
            # print("critic_loss", critic_loss)
            critic_model_optimizer.zero_grad()
            gen_model_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
            critic_model_optimizer.step()

            # Update generator
            gen_objective = 0
            for j in range(REPLAY_SAMPLE_SIZE):
                mean = critic_model(sampled_oc[j], sampled_start_v[j], sampled_goal_v[j], sampled_coefficients[j])[0]
                # print("mean_gen", mean)
                dir_dist = Dirichlet(sampled_coefficients[j])
                entropy = dir_dist.entropy()
                # print("entropy", entropy)
                dual_terms = (log_alpha.exp().detach() * entropy)
                # print("dual_term", dual_terms)
                gen_objective = gen_objective + (mean - dual_terms)
                # print("gen_objective", gen_objective)
            # print("gen_objectivesum", gen_objective)
            critic_model_optimizer.zero_grad()
            gen_model_optimizer.zero_grad()
            gen_objective.backward()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
            gen_model_optimizer.step()

            # update alpha
            alpha_loss = 0
            for j in range(REPLAY_SAMPLE_SIZE):
                dir_dist = Dirichlet(sampled_coefficients[j])
                entropy = dir_dist.entropy()

                alpha_loss_single = log_alpha.exp() * ((entropy - torch.tensor(
                    TARGET_ENTROPY, device=device, dtype=torch.float32)).detach())
                #  print("1", alpha_loss_single)
                alpha_loss = alpha_loss + alpha_loss_single
            # print("2", alpha_loss)
            # print("3", alpha_loss)
            alpha_optim.zero_grad()
            alpha_loss.backward()
            with torch.no_grad():
                log_alpha.grad *= (((-log_alpha.grad >= 0) | (log_alpha >= LOG_ALPHA_MIN)) &
                                   ((-log_alpha.grad < 0) | (log_alpha <= LOG_ALPHA_MAX))).float()  # ppo
            alpha_optim.step()

            # critic_loss = critic_loss.detach().numpy()
            critic_loss = critic_loss.item()
            gen_objective = gen_objective.item()
            alpha_loss = alpha_loss.item()
            writer.add_scalar("CRITIC LOSS", critic_loss, i)
            writer.add_scalar("GEN LOSS", gen_objective, i)
            writer.add_scalar("ALPHA LOSS", alpha_loss, i)

        torch.save(critic_model, 'net.critic')
        torch.save(gen_model, 'net.generator')

        writer.close()
        # tensorboard --logdir=/home/wangkaige/Project/apes/LOSS_FUNCTION


