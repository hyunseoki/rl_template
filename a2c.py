import gymnasium as gym
import torch
import numpy as np
import tqdm
from GAE import advantage_GAE, discount_cumulation
from matplotlib import pyplot as plt


import scipy
import numpy as np


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)
        # Orthogonal initialization
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.orthogonal_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
    def get_action(self, state, std):
        mean = self.forward(state)
        noise = torch.normal(mean=torch.zeros_like(mean), std=std)
        action = mean + noise
        return action
    
    def log_prob(self, action, mean, std):
        # 가우시안 분포의 확률 계산: PDF = (1/√(2πσ²)) * exp(-0.5*((x-μ)/σ)²)
        # 로그 확률 계산: log(PDF) = log(1/√(2πσ²)) + log(exp(-0.5*((x-μ)/σ)²))
        # 첫 번째 항 : log(1/√(2πσ²)) = -0.5 * log(2π) - log(σ)
        # 두 번째 항 : log(exp(-0.5*((x-μ)/σ)²)) = -0.5 * ((x-μ)/σ)²

        first_term = -0.5 * torch.log(torch.tensor(2 * np.pi)) - torch.log(torch.tensor(std))
        second_term = -0.5 * ((action - mean) / std) ** 2
        log_prob = first_term + second_term
        
        return log_prob.sum(dim=-1)  # 액션 차원에 대해 합산하여 로그 확률 반환 


class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        # Orthogonal initialization
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.orthogonal_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Buffer():
    def __init__(self, size, state_dim, action_dim, gamma=0.99, lamda=0.95):
        self.state_mem = np.zeros((size, state_dim), dtype=np.float32)
        self.action_mem = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_mem = np.zeros(size, dtype=np.float32)
        self.value_mem = np.zeros(size, dtype=np.float32)
        self.pointer = 0
        self.gamma = gamma
        self.lamda = lamda


    def store(self, state, action, reward, value):
        self.state_mem[self.pointer] = state
        self.action_mem[self.pointer] = action
        self.reward_mem[self.pointer] = reward
        self.value_mem[self.pointer] = value
        self.pointer += 1


    def finish_trajectory(self):
        self.advantage_mem = advantage_GAE(
            rewards=self.reward_mem[:self.pointer], 
            values=self.value_mem[:self.pointer], 
            gamma=self.gamma, 
            lamda=self.lamda
        )
        self.return_mem = discount_cumulation(
            self.reward_mem[:self.pointer], self.gamma
        )


    def get(self):
        size = self.pointer
        self.pointer = 0
        return self.state_mem[:size], \
            self.action_mem[:size],\
            self.advantage_mem, \
            self.return_mem[:size]


class Args:
    def __init__(self):
        self.device = 'cpu'
        self.task_name = 'InvertedPendulum-v5'
        self.gamma = 0.99
        self.lamda = 0.95

        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.buffer_size = 1000
        self.n_epoch = 2000
        self.learning_iter = 10

        self.std = 0.6065
        self.std_decay = 0.999
        self.entropy_loss_term = False


def main():
    seed_everything(42)
    args = Args()
    task_name = args.task_name
    env = gym.make(task_name, render_mode='rgb_array')
    s, info = env.reset()

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    buffer = Buffer(env.spec.max_episode_steps, s_dim, a_dim, args.gamma, args.lamda)

    actor=ActorNetwork(s_dim, a_dim)
    actor = actor.to(args.device)
    critic=CriticNetwork(s_dim)
    critic = critic.to(args.device)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    epi_length = list()
    scores = list()

    # print(f'Observation space : {env.observation_space}')
    # print(f'Action space : {env.action_space}')

    for i in tqdm.tqdm(range(args.n_epoch)):
        score = 0
        std = args.std * (args.std_decay ** i)
        s, info = env.reset()

        while True:
            action = actor.get_action(torch.tensor(s, dtype=torch.float32, device=args.device), std)
            s_next, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            value = critic(torch.tensor(s, dtype=torch.float32, device=args.device)).item()

            buffer.store(s, action.detach().cpu().numpy(), reward, value)
            score += reward
            s = s_next

            if terminated or truncated:
                buffer.finish_trajectory()
                epi_length.append(buffer.pointer)
                scores.append(score)
                break

        state_mem, action_mem, advantage_mem, ret_mem = buffer.get()
        state_mem = torch.tensor(state_mem, dtype=torch.float32, device=args.device)
        action_mem = torch.tensor(action_mem, dtype=torch.float32, device=args.device)
        advantage_mem = torch.tensor(advantage_mem.copy(), dtype=torch.float32, device=args.device)
        # 학습 루프에서 advantage 정규화 추가
        # advantage_mem = (advantage_mem - advantage_mem.mean()) / (advantage_mem.std() + 1e-8)
        ret_mem = torch.tensor(ret_mem.copy(), dtype=torch.float32, device=args.device)    

        for k in range(args.learning_iter):
            # Update Actor
            optimizer_actor.zero_grad()
            mean = actor(state_mem.to(args.device))
            log_prob = actor.log_prob(action_mem.to(args.device), mean, std)
            actor_loss = -(log_prob * advantage_mem.to(args.device)).mean()
            if args.entropy_loss_term:
                entropy_loss = torch.log(torch.tensor(std, device=args.device)) + 0.5 * torch.log(torch.tensor(2*np.pi*np.e, device=args.device))
                actor_loss -= entropy_loss * 0.01
            actor_loss.backward()
            optimizer_actor.step()

            # Update Critic
            optimizer_critic.zero_grad()
            value = critic(state_mem.to(args.device)).squeeze()
            critic_loss = torch.nn.functional.mse_loss(value, ret_mem.to(args.device))
            critic_loss.backward()
            optimizer_critic.step()

        if (i+1)%20==0: print(i+1,'에피소드 평균 점수:',np.mean(scores[-20:]))
        if np.min(scores[-5:])>=env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 조기종료
            break

    torch.save(actor.state_dict(), './a2c_continuous/actor.pth')
    env.close()

    plt.figure(figsize=(16,5))
    plt.plot(range(1,len(scores)+1),scores)
    smooth=np.convolve(scores,10*[0.1],mode='valid')
    plt.plot(range(1,len(smooth)+1),smooth)
    plt.title('A2C scores for '+task_name)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid()
    plt.savefig('./a2c_continuous/scores.png')
    plt.show()


if __name__ == '__main__':
    main()

