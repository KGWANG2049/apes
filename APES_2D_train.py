import time
import argparse
import planning
import torch.nn.functional
import torch.optim as optim
from torch.optim import Adam
from collections import deque
from network_2d import APESCriticNet, APESGeneratorNet

parser = argparse.ArgumentParser(description='APES Training Args')
# parser.add_argument('oc', (解析后的名称)dest='occupancy grid', default=True, action='store_false')
#  parser.add_argument('oc', default=True, action='store_false')
parser.add_argument('start_v', default=True, action='store_false')
parser.add_argument('goal_v', default=True, action='store_false')
parser.add_argument('coefficients', default=True, action='store_false')
# parser.add_argument('value_estimate', default=True, action='store_false')
args = parser.parse_args()

OC = planning.occ
SV = args.start_v
GV = args.goal_v
W = args.coefficients
VALUE_ESTIMATE = planning.PlanningResult
step = 0
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

replay_buffer = deque(maxlen=REPLAY_MAX)
experience = (OC, SV, GV, W, VALUE_ESTIMATE)
replay_buffer.append(experience)

if __name__ == '__main__':
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

#  2d input run or not , git
