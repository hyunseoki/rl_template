import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np


def discount_cumulation(rewards, gamma):
    """
    Compute discounted cumulative rewards.
    Args:
        rewards: list or 1D array of rewards
        gamma: discount factor
    Returns:
        discounted cumulative rewards as a 1D array
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]


class Policy(torch.nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 32)
        self.fc2 = nn.Linear(32, a_dim)
        # self.reset_parameters()


    def reset_parameters(self):
        # Weight initialization
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0.0)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1) ## (batch, a_dim)
        return x


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def learn(policy, optimizer, state_mem, action_mem, reward_mem, device, gamma, grad_log):
    return_mem = discount_cumulation(reward_mem, gamma)

    loss = 0
    for t in range(len(action_mem)):
        state = torch.tensor(state_mem[t], dtype=torch.float).to(device)
        action = torch.tensor(action_mem[t]).to(device)
        return_t = torch.tensor(return_mem[t], dtype=torch.float).to(device)

        # π(a_t|s_t)
        action_probs = policy(state)
        log_prob = torch.log(action_probs[action])
        # REINFORCE update: loss = -log(π(a_t|s_t)) * G_t
        loss += -log_prob * return_t

    optimizer.zero_grad()
    loss.backward()
    grad_log.append(grad_norm(policy)) # gradient norm 기록
    optimizer.step()

def main():
    gamma = 0.99
    n_episode = 1000
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    model = Policy(s_dim, a_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    state_mem = list()
    action_mem = list()
    reward_mem = list()

    epi_length = list()
    grad_log = list()
    for e in range(n_episode):
        state, info =env.reset()
        while True:
            action_probs = model(torch.tensor(state, dtype=torch.float).to(device))
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            state_mem.append(state)
            action_mem.append(action)
            reward_mem.append(reward)

            state = next_state
            if terminated or truncated:
                epi_length.append(len(action_mem))
                break


        for _ in range(5):  # 여러 번 학습
            learn(model, optimizer, state_mem, action_mem, reward_mem, device, gamma, grad_log)
        state_mem.clear()
        action_mem.clear()
        reward_mem.clear()

        print(e, 'episode length:', epi_length[-1])
        if np.min(epi_length[-5:]) >= env.spec.max_episode_steps:
            print("Solved!")
            break

    torch.save(model.state_dict(), './oh/policy.pth')
    env.close()

    plt.subplot(1,2,1)
    plt.plot(range(1, len(epi_length)+1), epi_length)
    smooth=np.convolve(epi_length, 10*[0.1], 'valid')
    plt.plot(range(1, len(smooth)+1), smooth)   
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length Over Time")

    plt.subplot(1,2,2)
    plt.plot(grad_log)
    plt.xlabel("Update step")
    plt.ylabel("Gradient norm")
    plt.title("Gradient norm over updates")
    plt.savefig('./oh/reinforce.png')
    plt.show()


if __name__ == "__main__":
    main()