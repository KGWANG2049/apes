import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torch.optim as optim
from torch.distributions import Dirichlet
from torch.optim import Adam
from collections import deque
import torch.distributions as dist
from tensorboardX import SummaryWriter
from mp2d.scripts.planning import Planning
from apes.Dist_gen import gmm_dist_generator
from mp2d.scripts.manipulator import manipulator
from apes.network_2d import APESCriticNet, APESGeneratorNet
from mp2d.scripts.utilities import load_planning_req_dataset
from experience import plan
from Rand_RRTCocnnect import random_plan
from datetime import datetime


BUFFER_MAX = 4
REPLAY_SAMPLE_SIZE = 2
EPOCH = 10
LR_cri = 1e-4
LR_gen = 1e-4
TARGET_ENTROPY = [-4.0]
LOG_ALPHA_INIT = [-1.0]
LOG_ALPHA_MIN = -10.
LOG_ALPHA_MAX = 20.
RAN_VALUE_LIST = []
VALUE_ESTIMATE_LIST = []
dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
pl_req_file_name = "/home/wang_ujsjo/Praktikum/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)
replay_buffer = deque(maxlen=BUFFER_MAX)
writer = SummaryWriter()

if __name__ == '__main__':

    SV = np.array(2)
    GV = np.array(2)
    W = np.array(50)
    data = np.array([])
    OC = np.array(np.shape(pl.get_occupancy_map(planning_requests[1])))
    device = torch.device("cuda:0")
    gen_model = APESGeneratorNet().double().to(device)
    gen_model.eval()
    critic_model = APESCriticNet().double().to(device)
    critic_model.eval()
    gen_model_optimizer = optim.Adam(gen_model.parameters(), lr=LR_gen)
    critic_model_optimizer: Adam = optim.Adam(critic_model.parameters(), lr=LR_cri)
    log_alpha = torch.tensor(LOG_ALPHA_INIT, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=LR_gen)
    writer = SummaryWriter("Loss_Function".format(datetime.now()))

    for p in range(EPOCH):
        replay_buffer.clear()
        VALUE_RAN_SUM = 0
        VALUE_ESTIMATE_SUM = 0
        for i in range(0, BUFFER_MAX):
            # ran_idx = torch.randint(low=0, high=4000, size=(1,))
            pl_req = planning_requests[i]
            OC = pl.get_occupancy_map(pl_req)
            OC = OC.reshape([1, OC.shape[0], -1])
            SV = pl_req.start
            GV = pl_req.goal
            pl.generate_graph_halton(150)
            pr = pl.search(pl_req)
            # VALUE_ESTIMATE = pr.checked_counts
            # VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE)
            OC = torch.tensor(OC).to(device)
            SV = torch.tensor(SV).to(device)
            GV = torch.tensor(GV).to(device)
            W = gen_model(OC, SV, GV).rsample().to(device)
            # print("W", W, W.shape)
            gmm_dist = gmm_dist_generator(W)
            VALUE_ESTIMATE = plan(pl_req, gmm_dist)[0]
            VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE).to(device)
            # print("GMM maxcount", VALUE_ESTIMATE)
            experience = ([OC, SV, GV, W, VALUE_ESTIMATE])
            replay_buffer.append(experience)
            VALUE_RAN = random_plan(pl_req)
            VALUE_RAN_SUM = VALUE_RAN + VALUE_RAN_SUM
            VALUE_ESTIMATE_SUM = VALUE_ESTIMATE.cpu() + VALUE_ESTIMATE_SUM
            # del MVS, RVS
            print('Current Batch', p + 1, "/", EPOCH,
                  'Waiting for buffer size ... {}/{}'.format(len(replay_buffer), BUFFER_MAX))
            print("value ran", VALUE_RAN_SUM, "value gmm", VALUE_ESTIMATE_SUM)
        RAN_VALUE_LIST.append(VALUE_RAN_SUM)
        VALUE_ESTIMATE_LIST.append(VALUE_ESTIMATE_SUM)

        # labels = sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients, sampled_values
        sampled_evaluations = random.sample(replay_buffer, BUFFER_MAX)
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

        for i in range(20):
            critic_loss = 0
            for j in range(REPLAY_SAMPLE_SIZE):
                sam_idx = j+REPLAY_SAMPLE_SIZE*i
                mean, std = critic_model(sampled_oc[sam_idx], sampled_start_v[sam_idx], sampled_goal_v[sam_idx], sampled_coefficients[sam_idx])
                std = torch.exp(std)
                # print("mean:", mean, "std:", std)
                priori_pro = dist.Normal(mean, std)
                # print("posterior:", priori_pro)
                posterior_prob = priori_pro.log_prob(sampled_values[j])
                # print("posterior_prob:", posterior_prob)
                # print("test", priori_pro.log_prob(a).exp())
                critic_loss = (critic_loss + (-posterior_prob))/REPLAY_SAMPLE_SIZE

            critic_model_optimizer.zero_grad()

            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
            critic_model_optimizer.step()

            # Update generator
            gen_objective = 0
            for k in range(REPLAY_SAMPLE_SIZE):
                sam_idx = k + REPLAY_SAMPLE_SIZE * i
                print("1212121111112121", sam_idx)
                mean = critic_model(sampled_oc[sam_idx], sampled_start_v[sam_idx], sampled_goal_v[sam_idx], sampled_coefficients[sam_idx])[0]
                # print("mean", mean)
                # dir_dist = Dirichlet(sampled_coefficients[k])
                entropy = gen_model(sampled_oc[sam_idx], sampled_start_v[sam_idx], sampled_goal_v[sam_idx]).entropy()
                # print("entropy", entropy)
                dual_terms = (log_alpha.exp().detach() * entropy)
                # print("dual_term", dual_terms)
                gen_objective = (gen_objective + mean - dual_terms)/REPLAY_SAMPLE_SIZE
                # gen_objective = gen_objective + mean

            # print("gen_objectivesum", gen_objective)
            # critic_model_optimizer.zero_grad()
            # gen_model_optimizer.zero_grad()

            gen_model_optimizer.zero_grad()
            gen_objective.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
            gen_model_optimizer.step()
            print("critic_loss", critic_loss)
            print("gen_objective", gen_objective)

            # update alpha
            alpha_loss = 0
            for m in range(REPLAY_SAMPLE_SIZE):
                sam_idx = m + REPLAY_SAMPLE_SIZE * i
                # dir_dist = Dirichlet(sampled_coefficients[j])
                entropy = gen_model(sampled_oc[sam_idx], sampled_start_v[sam_idx], sampled_goal_v[sam_idx]).entropy()
                print("entropy", entropy)
                alpha_loss_single = log_alpha.exp() * ((entropy - torch.tensor(
                    TARGET_ENTROPY, device=device, dtype=torch.float32)).detach())
                #  print("1", alpha_loss_single)
                alpha_loss = (alpha_loss + alpha_loss_single)/REPLAY_SAMPLE_SIZE
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

            # loss function visualable

            writer.add_scalar("CRITIC LOSS", critic_loss)
            writer.add_scalar("GEN LOSS", gen_objective)
            # writer.add_scalar("ALPHA LOSS", alpha_loss, p)
        writer.close()

    # Compare result visualization
    plt.plot(RAN_VALUE_LIST, color="red")
    plt.plot(VALUE_ESTIMATE_LIST, color="green")  # Use Tensor.cpu() to copy the tensor to host memory first.
    plt.xlabel(' Train Epoch')
    plt.ylabel('Iteration Numbers')
    plt.legend(['normal RRTConnect', 'RRTConnect with APES'])
    plt.show()
    # here can add compare with compartor , variant can add VALUE_ESTIMATE as number of iterations, but still need
    # add rrt_random in for i in range(0, BUFFER_MAX): to get other num_iterations
    torch.save(critic_model, 'net.critic')
    torch.save(gen_model, 'net.generator')

    # tensorboard --logdir=/home/wang_ujsjo/Praktikum/apes/Loss_Function or

    # tensorboard --logdir /home/wang_ujsjo/Praktikum/apes/Loss_Function

# modified can be saved 08.03,23
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.DoubleTensor [200, 50]], which is output 0 of AsStridedBackward0, is at version 2; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

# Process finished with exit code 1