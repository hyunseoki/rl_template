## DNN based q-learning

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
# tf.debugging.set_log_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
import numpy as np
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt


gamma = 0.99
eps = 1.0
eps_decay = 0.9999
min_eps = 0.05
n_episode = 2000

greedy_select = lambda x : np.random.choice(np.argwhere(x==np.max(x)).flatten())


def build_network():
    mlp = Sequential()
    mlp.add(Dense(32, input_dim=s_dim, activation='relu'))
    mlp.add(Dense(32, activation='relu'))
    mlp.add(Dense(a_dim, activation='linear'))
    mlp.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return mlp

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0] ## s_dim = 4
a_dim = env.action_space.n ## a_dim = 2

model = build_network()

epi_length = list()

for i in range(n_episode):
    state, info = env.reset()
    length = 0
    while True:
        
        # Q_θ(s_t, ·), ##(2,)
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # a_t ~ π_ε(a | s_t)
        if np.random.rand() < eps:
            a = np.random.randint(0, a_dim)
        else:
            a = greedy_select(q_values)
        
        # s_{t+1}, r_t ~ P(· | s_t, a_t)
        next_state, r, terminated, truncated, info = env.step(a)

        # Q_θ(s_{t+1}, ·)
        next_q_values = model.predict(next_state.reshape(1, -1), verbose=0)[0]
        target_q_values = q_values
        if terminated:
            target_q_values[a] = r
        else:
            ## r + γ max_a' Q_θ(s', a')
            target_q_values[a] = r + gamma * np.max(next_q_values)
        
        model.fit(state.reshape([1, s_dim]), target_q_values.reshape([1, a_dim]), batch_size=1, epochs=1, verbose=0)
        state = next_state
        length += 1
        eps = max(min_eps, eps * eps_decay)
        

        if terminated or truncated:
            epi_length.append(length)
            break

    if np.min(epi_length[-5:]) >= env.spec.max_episode_steps:
        print(f"Solved in episode {i}")
        break
    if (i+1) % 10 == 0:
        print(f"Episode {i+1}, 에피소드 길이 : {np.mean(epi_length[-10:])}, eps : {eps:.3f}")

model.save('./f6-2.keras')
env.close()

plt.plot(range(1, len(epi_length)+1), epi_length)
smooth = np.convolve(epi_length, 10*[0.1], model='valid')
plt.plot(range(1, len(smooth)+1), smooth, 'r')
plt.title('Converged of Q-learning on CartPole-v1')
plt.ylabel('Episode length')
plt.xlabel('Episode')
plt.grid()
plt.show()
