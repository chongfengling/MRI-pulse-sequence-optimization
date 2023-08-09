import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from utilities import *
import argparse
import random

# action space: {t1:_, t3:_, d1:_, d2:_, G1:_, G2:_} 6 arrays.


class Env:
    # Environment, generates state and reward based on action
    def __init__(self):
        pass

    def make(self, env_name, args, plot=False):
        # create the environment
        if env_name == "Two-Constant-Gradient":
            # x space (spatial space)
            self.FOV_x = args.FOV_x  # field of view in x space
            self.N = (
                args.N  # sampling points in x space (and k space, time space during ADC)
            )
            self.delta_x = self.FOV_x / self.N  # sampling interval in x space
            self.x_axis = np.linspace(
                -self.FOV_x / 2, self.FOV_x / 2 - self.delta_x, self.N
            )  # symmetric x space
            # k space (frequency space)
            self.delta_k = 1 / self.FOV_x  # sampling interval in k space
            self.FOV_k = self.delta_k * self.N  # field of view in k space
            self.k_axis = np.linspace(
                -self.FOV_k / 2, self.FOV_k / 2 - self.delta_k, self.N
            )  # symmetric k space
            self.gamma = 2.68e8  # rad/s/T
            self.gamma_bar = 0.5 * self.gamma / np.pi  # s^-1T^-1
            # t space (time space) based on G1 and G2
            # create object over x space
            self.density = np.zeros(len(self.x_axis))
            self.density[
                int(len(self.x_axis) / 4 + len(self.x_axis) / 8) : int(
                    len(self.x_axis) / 4 * 3 - len(self.x_axis) / 8
                )
            ] = 1
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(self.x_axis, self.density, '-', label='object')
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('density')
                plt.grid()
                plt.show()
            # prepare for simulation
            # create spins after the rf pulse (lying on the y-axis)
            # assume the spins are lying on each sampling point over y-axis
            self.m0 = 1.0
            self.w_0 = 0
            self.vec_spins = np.zeros((3, self.N))
            self.vec_spins[1, :] = 1

        else:
            raise RuntimeError("Env name not found.")

    def reset(self):
        # reset the environment and return the initial state
        return np.random.rand(len(self.x_axis))

    def action_space_sample(self):
        # return a random action from the action space
        # 6 variables: t1:_, t3:_, d1:_, d2:_, G1symbol:_, G2symbol:_, Gvalues:_
        self.action_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = np.random.uniform(
            low=self.action_low, high=self.action_high, size=7
        )

    def step(self, action, plot=False):
        # action is an array of 7 variables (t1, t2, d1, d2, G1symbol, G2symbol, Gvalues)
        # take an action and return the next state, reward, a boolean indicating if the episode is done and additional info

        # calculate the gradient values
        (t1, t2, d1, d2, G1symbol, G2symbol, Gvalue) = action
        Gvalue = Gvalue * 1e-6
        gamma_bar_G = self.gamma_bar * Gvalue * 1e-3
        delta_t = self.delta_k / gamma_bar_G
        Ts = self.FOV_k / gamma_bar_G  # ADC duration FIXED
        t_max = (
            1.5 * Ts - delta_t
        )  # maximum time (ms) (rephasing process is 2 times longer than dephasingprocess)
        t_axis = np.linspace(0, t_max, int(self.N * 1.5))
        G_values_array = np.zeros(len(t_axis))
        # two gradient can be overlapped
        # print(f't1: {t1}, t2: {t2}, d1: {d1}, d2: {d2}, G1symbol: {G1symbol}, G2symbol: {G2symbol}, Gvalue: {Gvalue}')
        G_values_array[
            int(t1 * len(t_axis)) : int(t1 * len(t_axis) + d1 * len(t_axis))
        ] += (G1symbol * Gvalue)
        G_values_array[
            int(t2 * len(t_axis)) : int(t2 * len(t_axis) + d2 * len(t_axis))
        ] += (G2symbol * Gvalue)
        k_traj = np.cumsum(G_values_array) * 1e-3
        if plot:
            _, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
            ax1.plot(t_axis, G_values_array)
            ax1.set_ylabel('Gx (mT/m)')

            ax2.plot(t_axis, k_traj)
            ax2.set_ylabel('k (1/m)')

            ax2.set_xticks(
                [0, t_axis[int(self.N / 2) - 1], t_axis[int(self.N * 1.5) - 1]]
            )
            ax2.set_xticklabels(['$t_1$', r'$t_2 (t_3)$', r'$t_4$'])

            plt.show()

        # do relaxation
        # define larmor frequency w_G of spins during relaxation
        # shape = (number of time steps, number of sampling points)
        w_G = np.outer(G_values_array, self.x_axis) * self.gamma * 1e-3 + self.w_0

        res = multiple_Relaxation(
            self.vec_spins,
            m0=self.m0,
            w=0,
            w0=w_G,
            t1=1e10,
            t2=1e10,
            t=1.5 * Ts,
            steps=int(self.N * 1.5),
            axis='z',
        )

        store = []
        for i in range(2):
            tmp = res[
                i, :, :
            ].squeeze()  # shape: (number of steps, number of sampling points)

            store.append(tmp @ self.density)  # multiply by true density

        Mx_1, My_1 = store[0][: int(self.N / 2)], store[1][: int(self.N / 2)]
        Mx_2, My_2 = store[0][int(self.N / 2) :], store[1][int(self.N / 2) :]

        # plot the full signal
        signal_Mx = np.concatenate((Mx_1, Mx_2), axis=0)
        signal_My = np.concatenate((My_1, My_2), axis=0)
        adc_signal = Mx_2 * 1 + 1j * My_2
        re_density = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(adc_signal)))
        abs_re_density = np.abs(re_density)
        mse = (np.linalg.norm(abs_re_density - self.density) ** 2) / len(self.density)
        plt.plot(self.x_axis, abs_re_density, label='reconstruction')
        plt.plot(self.x_axis, self.density, label='original')
        plt.legend()
        # plt.show()
        # print(f'error (MSE) {mse}')
        info = None

        return abs_re_density, mse, False, info

    # def action_space.sample(self):
    #     # return a random action
    #     pass

    def render(self):
        # display the current state of the environment
        pass


class ActorNetwork(nn.Module):
    # Actor Network, generates action based on state
    # observation is the state
    def __init__(self, state_space, action_space):
        super(ActorNetwork, self).__init__()

        # Fully-connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(state_space, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        # Output layer
        # rescale?
        # self.output_layer = nn.Linear(32, action_space)
        self.output_layer = nn.Sequential(nn.Linear(32, action_space), nn.Sigmoid())

    def forward(self, state):
        tmp = self.fc_layers(state)
        out = self.output_layer(tmp)
        # out = 0.9 * out + 0.01
        return out


class CriticNetwork(nn.Module):
    # Critic Network, generates Q value based on (current?) state and action
    def __init__(self, state_space, action_space):
        super(CriticNetwork, self).__init__()

        # state input stream
        self.state_stream = nn.Sequential(
            nn.Linear(state_space, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU()
        )

        # action input stream
        self.action_stream = nn.Sequential(
            nn.Linear(action_space, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU()
        )

        # combined layer
        self.combined_layer = nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        state_tmp = self.state_stream(state)
        action_tmp = self.action_stream(action)
        combined = torch.cat((state_tmp, action_tmp), dim=1)
        out = self.combined_layer(combined)
        return out


class DPPG:
    def __init__(self, state_space, action_space, env, args):
        self.state_space = state_space
        self.action_space = action_space

        self.env = env  #! necessary?

        self.seed = args.seed
        self.lr_a, self.lr_c = args.lr_a, args.lr_c

        # create Actor Network and its target network
        self.actor = ActorNetwork(state_space, action_space)
        self.actor_target = ActorNetwork(state_space, action_space)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_a
        )  #!
        # make sure the target network has the same weights as the original network

        hard_update(self.actor_target, self.actor)
        # for target_param, param in zip(
        #     self.actor.parameters(), self.actor_target.parameters()
        # ):
        #     target_param.data.copy_(param.data)

        # create Critic Network and its target network
        self.critic = CriticNetwork(state_space, action_space)
        self.critic_target = CriticNetwork(state_space, action_space)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_c
        )  #!
        # make sure the target network has the same weights as the original network

        # ? check
        hard_update(self.critic_target, self.critic)

        # for target_param, param in zip(
        #     self.critic.parameters(), self.critic_target.parameters()
        # ):
        #     target_param.data.copy_(param.data)

        # initialize replay buffer
        self.memory_capacity = args.memory_capacity
        self.batch_size = args.batch_size
        # store one current states, one action, one reward, one next state
        self.memory = np.zeros(
            (self.memory_capacity, self.state_space * 2 + self.action_space + 1),
            dtype=np.float32,
        )
        self.mpointer = 0  # memory pointer

        # define hyper-parameters
        self.tau = args.tau
        self.gamma = args.gamma
        #! self.depsilon = 1.0 / 50000

        self.exploration_var = args.exploration_var
        self.left_clip, self.right_clip = args.left_clip, args.right_clip

        # self.epsilon = 1.0
        self.s_t = None  # most recent state
        self.a_t = None  # most recent action
        self.is_training = True

        self.criterion = nn.MSELoss()

    def select_action(self, state, exploration_noise=True):
        # return an action based on the current state with or without exploration noise
        # print(to_numpy(self.actor(to_tensor(state))).shape)
        action = to_numpy(self.actor(to_tensor(state)))

        if exploration_noise:
            # add noise controlled by hyper-parameter var
            action = np.clip(
                np.random.normal(action, self.exploration_var),
                self.left_clip,
                self.right_clip,
            )

        self.s_t = action
        return action

    def store_transition(self, state, action, reward, state_):
        # store the transition in the replay buffer

        transition = np.hstack(
            (
                state.astype(np.float32),
                action.astype(np.float32),
                [reward],
                state_.astype(np.float32),
            )
        )

        current_index = (
            self.mpointer % self.memory_capacity
        )  # replace if memory is full
        self.memory[current_index, :] = transition
        self.mpointer += 1

    def update_network(self):
        # update two networks based on the transitions stored in the replay buffer

        # sample a batch of transitions
        indices = random.sample(range(self.memory_capacity), self.batch_size)

        batch = self.memory[indices, :]
        batch_state = to_tensor(batch[:, : self.state_space])
        batch_action = to_tensor(
            batch[:, self.state_space : self.state_space + self.action_space]
        )
        # ? scale the reward to small value?
        batch_reward = batch[:, -self.state_space - 1 : -self.state_space]
        batch_state_ = to_tensor(batch[:, -self.state_space :])

        # prepare for the target q batch
        with torch.no_grad():
            q_target_batch = self.critic_target(
                batch_state_, self.actor_target(batch_state_)
            )

        y = batch_reward + self.gamma * to_numpy(q_target_batch)
        q_batch = self.critic(batch_state, batch_action)
        # calculate the value loss
        value_loss = self.criterion(q_batch, to_tensor(y))
        # update the critic network based on the value loss
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update the actor network
        self.actor.zero_grad()
        policy_loss = -self.critic(batch_state, self.actor(batch_state)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # (soft) update two target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


def parse_arguments():
    parser = argparse.ArgumentParser(description='DPPG_Two-Constant-Gradient')
    parser.add_argument('--env', default='Two-Constant-Gradient', type=str, help='env')
    parser.add_argument('--FOV_x', default=32, type=int, help='FOV_x')
    parser.add_argument(
        '--N',
        default=32,
        type=int,
        help='sampling points in x space (and k space, time space during ADC)',
    )
    parser.add_argument('--seed', default=215, type=int, help='seed')
    parser.add_argument(
        '--state_space',
        default=32,
        type=int,
        help='state_space including density of objects',
    )
    parser.add_argument(
        '--action_space',
        default=7,
        type=int,
        help='action_space including t1, t3, d1, d2, G1symbol, G2symbol, Gvalue',
    )
    parser.add_argument(
        '--num_episode',
        default=8,
        type=int,
        help='number of episodes. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep', default=200, type=int, help='number of steps per episode'
    )

    parser.add_argument(
        '--a_hidden1', default=512, type=int, help='hidden layer 1 in actor network'
    )
    parser.add_argument(
        '--a_hidden2', default=128, type=int, help='hidden layer 2 in actor network'
    )
    parser.add_argument(
        '--a_hidden3', default=32, type=int, help='hidden layer 3 in actor network'
    )

    parser.add_argument(
        '--c_s_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_s_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_a_hidden1',
        default=16,
        type=int,
        help='hidden layer 1 in critic network for action input stream',
    )
    parser.add_argument(
        '--c_a_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for action input stream',
    )

    parser.add_argument(
        '--c_combined_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for combined network',
    )
    parser.add_argument(
        '--c_combined_hidden2',
        default=256,
        type=int,
        help='hidden layer 2 in critic network for combined network',
    )

    parser.add_argument(
        '--lr_a', default=0.001, type=float, help='learning rate of actor network'
    )
    parser.add_argument(
        '--lr_c', default=0.002, type=float, help='learning rate of critic network'
    )

    parser.add_argument(
        '--memory_capacity', default=50, type=int, help='memory capacity'
    )
    parser.add_argument(
        '--batch_size', default=32, type=int, help='minibatch size from memory'
    )

    parser.add_argument(
        '--exploration_var',
        default=0.1,
        type=float,
        help='exploration variance in random process',
    )
    parser.add_argument('--left_clip', default=0.01, type=float, help='left clip')
    parser.add_argument('--right_clip', default=0.99, type=float, help='right clip')

    parser.add_argument(
        '--gamma', default=0.9, type=float, help='reward discount factor'
    )
    parser.add_argument('--tau', default=0.01, type=float, help='soft update factor')

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    env = Env()
    agent = DPPG(
        state_space=args.state_space, action_space=args.action_space, env=env, args=args
    )
    env.make(env_name=args.env, args=args, plot=False)

    def train(
        agent, env, num_episode=args.num_episode, num_steps_per_ep=args.num_steps_per_ep
    ):
        reward_record = []
        for i in range(num_episode):
            # reset the environment
            state = env.reset()
            # record time and current reward in this episode
            t1 = time.time()
            episode_reward = 0

            for j in range(num_steps_per_ep):
                # return an action based on the current state
                action = agent.select_action(state, exploration_noise=True)
                # print(action)
                # interact with the environment
                # ? value of reward should be scaled or not
                state_, reward, done, info = env.step(action)
                # store the transition
                agent.store_transition(state, action, reward, state_)

                # update the network if the replay memory is full
                if agent.mpointer > agent.memory_capacity:
                    # break
                    # print(
                    #     f'update, mpointer = {agent.mpointer}, memory_capacity = {agent.memory_capacity}'
                    # )
                    agent.update_network()

                # output records
                state = state_
                episode_reward += reward
                if j == num_steps_per_ep - 1:
                    # if True:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, num_episode, episode_reward, time.time() - t1
                        )
                    )
                    reward_record.append(episode_reward)
        print(reward_record)
        plt.plot(state)

    train(agent=agent, env=env)


main()
