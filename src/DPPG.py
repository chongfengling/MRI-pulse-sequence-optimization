import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from utilities import *
import argparse
import random
from datetime import datetime
import os

# action space: {t1:_, t3:_, d1:_, d2:_, G1:_, G2:_} 6 arrays.


class Env:
    # Environment, generates state and reward based on action
    def __init__(self, args, plot=False):
        # create the universal environment
        # x space (spatial space)
        self.FOV_x = args.FOV_x  # field of view in x space (mm)
        self.N = (
            args.N  # sampling points in x space (and k space, time space during ADC)
        )
        self.delta_x = self.FOV_x / self.N  # sampling interval in x space, (mm)
        self.x_axis = np.linspace(
            -self.FOV_x / 2, self.FOV_x / 2 - self.delta_x, self.N
        )  # symmetric x space
        # k space (frequency space)
        self.delta_k = 1 / self.FOV_x  # sampling interval in k space (1/mm)
        self.FOV_k = self.delta_k * self.N  # field of view in k space (1/mm)
        self.k_axis = np.linspace(
            -self.FOV_k / 2, self.FOV_k / 2 - self.delta_k, self.N
        )  # symmetric k space
        self.gamma = 2.68e8  # rad/(s*T)
        self.gamma_bar = 0.5 * self.gamma / np.pi  # s^-1T^-1
        # t space (time space) based on G1 and G2
        # create object over x space. one env, one object
        self.density = np.zeros(len(self.x_axis))
        self.density[int(len(self.x_axis) / 8) : int(len(self.x_axis) / 4)] = 2
        self.density_complex = self.density.astype(complex)
        # prepare for simulation
        # create spins after the rf pulse (lying on the y-axis)
        # assume the spins are lying on each sampling point over y-axis
        self.m0 = 1.0
        self.w_0 = 0
        self.vec_spins = np.zeros((3, self.N))
        self.vec_spins[1, :] = 1
        self.env_name = args.env
        self.T2 = args.T2

    def make(self, args):
        # specify the action space
        if self.env_name == "Two-Constant-Gradient":
            # 6 variables: t1:_, t3:_, d1:_, d2:_, G1symbol:_, G2symbol:_, Gvalues:_
            self.action_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            self.action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.action_space = np.random.uniform(
                low=self.action_low, high=self.action_high, size=7
            )
        elif self.env_name == "Two-Constant-Gradient-With_Slope":
            self.action_low = np.array([0.0, 0.0, 0.0, 0.0])
            self.action_high = np.array([1.0, 1.0, 1.0, 1.0])
            self.action_space = np.random.uniform(
                low=self.action_low, high=self.action_high, size=4
            )
            self.max_slew_rate = args.max_slew_rate
            self.slope_penalty_factor = args.slope_penalty_factor
        else:
            raise ValueError("Invalid environment name.")

    def reset(self):
        # reset and return a new state (reconstructed density with two components: real and imaginary)
        return np.random.rand(len(self.x_axis) * 2)

    def step(self, action, plot=False):
        # (d1, d2, d3, d4, d5, d6, d7, d8, GValue_01) = action
        (alpha, beta, GValue_01) = action
        # max gradient is 40 mT/m = 4e-5 T/mm
        GValue = GValue_01 * 1e-5 * 40
        G1 = GValue * 1
        G2 = GValue * (-1)
        # calculate the delta_t and total time based on max gradient value
        gamma_bar_G = self.gamma_bar * GValue * 1e-3
        delta_t = self.delta_k / gamma_bar_G  # Fixed

        Ts = self.FOV_k / gamma_bar_G  # ADC duration FIXED
        t_max = (
            1.5 * Ts - delta_t
        )  # maximum time (ms) (rephasing process is 2 times longer than dephasingprocess)

        self.t_axis = np.linspace(0, t_max, int(self.N * 1.5))
        # define gradient values over time
        G_values_array = np.zeros(len(self.t_axis))

        N_G1 = nearest_even(alpha * self.N * 0.5)
        N_G2 = nearest_even(beta * self.N)
        N_G1_up = int(0.5 * (self.N * 0.5 - N_G1))
        N_G2_up = int(0.5 * (self.N - N_G2))
        assert N_G1 + N_G2 + N_G1_up * 2 + N_G2_up * 2 == self.N * 1.5

        G_values_array[:N_G1_up] = np.linspace(0, G1, N_G1_up)
        G_values_array[N_G1_up : N_G1_up + N_G1] = G1
        G_values_array[N_G1_up + N_G1 : N_G1_up + N_G1 + N_G1_up] = np.linspace(
            G1, 0, N_G1_up
        )

        G_values_array[
            N_G1_up + N_G1 + N_G1_up : N_G1_up + N_G1 + N_G1_up + N_G2_up
        ] = np.linspace(0, G2, N_G2_up)
        G_values_array[
            N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up : N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2
        ] = G2
        G_values_array[
            N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2 : N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2
            + N_G2_up
        ] = np.linspace(G2, 0, N_G2_up)

        # store the G_values_array
        self.G_values_array = G_values_array
        # calculate k space trajectory
        self.k_traj = np.cumsum(G_values_array) * 1e-3

        # do relaxation
        # define larmor frequency w_G of spins during relaxation
        # shape = (number of time steps, number of sampling points)
        w_G = np.outer(self.G_values_array, self.x_axis) * self.gamma * 1e-3 + self.w_0
        # T1 relaxation is not considered as it does not affect the signal on the xy plane
        res = multiple_Relaxation(
            self.vec_spins,
            m0=self.m0,
            w=0,
            w0=w_G,
            t1=1e10,
            t2=self.T2,
            t_axis=self.t_axis,
            steps=int(self.N * 1.5),
            axis='z',
        )
        store = []
        for i in range(2):
            tmp = res[
                i, :, :
            ].squeeze()  # shape: (number of steps, number of sampling points)
            store.append(tmp @ self.density)  # multiply by true density
        # signal during ADC
        Mx_2, My_2 = store[0][int(self.N / 2) :], store[1][int(self.N / 2) :]
        adc_signal = Mx_2 * 1j + 1 * My_2
        re_density = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(adc_signal)))
        re_density = np.real(re_density) * 1j + np.imag(re_density)
        abs_re_density = np.abs(re_density)
        r_i_re_density = np.concatenate(
            (np.real(re_density), np.imag(re_density)), axis=0
        )

        # reward has two components: the first is the MSE of two complex arrays, the second is the slew rate
        # reward of mse < 0
        mse = mse_of_two_complex_nparrays(re_density, self.density_complex)
        info = -mse
        reward = -mse
        # reward of slew rate in the range of [-1 or 1] * factor

        # calculate the slew rate
        sr_1, sr_2 = (GValue / (delta_t * N_G1_up), GValue / (delta_t * N_G2_up))
        done = 0.0

        if (
            max(sr_1, sr_2) > self.max_slew_rate * 10
        ):  # unacceptable slew rate, fail for this episode
            # reward_slew_rate = 1
            # print(f'max slew rate exceeded: {max(sr_1, sr_2)}, GValue: {GValue}, delta_t: {delta_t}')
            reward += -1
            done = 1.0
        elif max(sr_1, sr_2) > self.max_slew_rate * 4:
            reward += -0.5
        else:
            # reward_slew_rate = 5
            # done = 0.0
            reward += 0.5
        #! ?
        experienced_mean = -2.18
        experienced_std = 0.62
        reward = (reward - experienced_mean) / experienced_std

        if plot:
            plt.plot(self.x_axis, abs_re_density, label='reconstruction')
            plt.plot(self.x_axis, self.density, label='original')
            plt.legend()
            plt.show()
        # at this time use abs_re_density as the state
        return r_i_re_density, reward, done, info

    def render(self):
        # display the state (reconstructed density) and the object (true density) and gradients and k space trajectory

        pass


class ActorNetwork(nn.Module):
    # Actor Network, generates action based on state
    # observation is the state
    def __init__(self, state_space, action_space, args):
        super(ActorNetwork, self).__init__()

        # Fully-connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(state_space, args.a_hidden1),
            nn.ReLU(),
            nn.Linear(args.a_hidden1, args.a_hidden2),
            nn.ReLU(),
            nn.Linear(args.a_hidden2, args.a_hidden3),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(args.a_hidden3, action_space), nn.Sigmoid()
        )

    def forward(self, state):
        tmp = self.fc_layers(state)
        out = self.output_layer(tmp)

        return out


class CriticNetwork(nn.Module):
    # Critic Network, generates Q value based on (current?) state and action
    def __init__(self, state_space, action_space, args):
        super(CriticNetwork, self).__init__()

        # state input stream
        self.state_stream = nn.Sequential(
            nn.Linear(state_space, args.c_s_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_s_hidden1, args.c_s_hidden2),
            nn.ReLU(),
        )

        # action input stream
        self.action_stream = nn.Sequential(
            nn.Linear(action_space, args.c_a_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_a_hidden1, args.c_a_hidden2),
            nn.ReLU(),
        )

        # combined layer
        self.combined_layer = nn.Sequential(
            nn.Linear(args.c_s_hidden2 + args.c_a_hidden2, args.c_combined_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_combined_hidden1, args.c_combined_hidden2),
            nn.ReLU(),
            nn.Linear(args.c_combined_hidden2, 1),
        )

    def forward(self, state, action):
        state_tmp = self.state_stream(state)
        action_tmp = self.action_stream(action)
        combined = torch.cat((state_tmp, action_tmp), dim=1)
        out = self.combined_layer(combined)
        return out


class DPPG:
    def __init__(self, env, args):
        self.state_space = args.state_space
        self.action_space = args.action_space

        self.env = env  #! necessary?

        self.seed = args.seed
        self.lr_a, self.lr_c = args.lr_a, args.lr_c

        # create Actor Network and its target network
        self.actor = ActorNetwork(self.state_space, self.action_space, args)
        self.actor_target = ActorNetwork(self.state_space, self.action_space, args)
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
        self.critic = CriticNetwork(self.state_space, self.action_space, args)
        self.critic_target = CriticNetwork(self.state_space, self.action_space, args)
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
        # store one current states, one action, one reward, one next state, one done
        self.memory = np.zeros(
            (self.memory_capacity, self.state_space * 2 + self.action_space + 2),
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
            action[:8] = np.exp(action[:8]) / np.sum(np.exp(action[:8]), axis=0)

        self.s_t = action
        return action

    def store_transition(self, state, action, reward, state_, done):
        # store the transition in the replay buffer

        transition = np.hstack(
            (
                state.astype(np.float32),
                action.astype(np.float32),
                [reward],
                state_.astype(np.float32),
                [done],
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
        batch_reward = batch[:, -self.state_space - 2 : -self.state_space - 1]
        batch_state_ = to_tensor(batch[:, -self.state_space - 1 : -1])
        batch_done = batch[:, -1:]

        # prepare for the target q batch
        q_target_batch = self.critic_target(
            batch_state_, self.actor_target(batch_state_)
        )

        # if game is over (done = 0.0), no value for the next state's Q-value
        # y is the target Q-value
        y = batch_reward + self.gamma * to_numpy(q_target_batch) * batch_done
        # calculate the current Q-value
        q_batch = self.critic(batch_state, batch_action)
        # calculate the value loss
        value_loss = self.criterion(q_batch, to_tensor(y))
        # update the critic network based on the value loss
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update the actor network
        policy_loss = -self.critic(batch_state, self.actor(batch_state)).mean()
        self.actor.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # (soft) update two target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def save_model(self, path):
        # save the model
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic.state_dict(), path + '_critic.pth')
        torch.save(self.actor_target.state_dict(), path + '_actor_target.pth')
        torch.save(
            self.critic_target.state_dict(),
            path + '_critic_target.pth',
        )

    def load_model(self, path):
        # load the model
        self.actor.load_state_dict(torch.load(path + '_actor.pth'))
        self.critic.load_state_dict(torch.load(path + '_critic.pth'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='DPPG_Two-Constant-Gradient')
    parser.add_argument(
        '--env', default='Two-Constant-Gradient-With_Slope', type=str, help='env'
    )
    parser.add_argument('--FOV_x', default=32, type=int, help='FOV_x')
    parser.add_argument(
        '--N',
        default=32,
        type=int,
        help='sampling points in x space (and k space, time space during ADC)',
    )
    parser.add_argument(
        '--T2', default=3e1, type=float, help='T2 relaxation time in ms'
    )
    parser.add_argument(
        '--max_slew_rate',
        default=2e-4,
        type=float,
        help='max slew rate of gradient in (T / (m * ms)), defined as peak amplitude of gradient divided by rise time',
    )
    parser.add_argument(
        '--slope_penalty_factor', default=0, type=float, help='slope penalty factor'
    )
    parser.add_argument('--seed', default=215, type=int, help='seed')
    parser.add_argument(
        '--state_space',
        default=32 * 2,
        type=int,
        help='state_space including density of objects',
    )
    parser.add_argument(
        '--action_space',
        default=3,
        type=int,
        help='action_space including t1, t3, d1, d2, G1symbol, G2symbol, Gvalue',
    )
    parser.add_argument(
        '--num_episode',
        default=16,
        type=int,
        help='number of episodes. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep', default=4096, type=int, help='number of steps per episode'
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
        '--lr_a', default=0.0001, type=float, help='learning rate of actor network'
    )
    parser.add_argument(
        '--lr_c', default=0.001, type=float, help='learning rate of critic network'
    )
    parser.add_argument('--warmup', default=256, type=int, help='warmup, no training')
    parser.add_argument(
        '--memory_capacity', default=500000, type=int, help='memory capacity'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='minibatch size from memory'
    )

    parser.add_argument(
        '--exploration_var',
        default=0.8,
        type=float,
        help='exploration variance in random process',
    )
    parser.add_argument('--left_clip', default=0.01, type=float, help='left clip')
    parser.add_argument('--right_clip', default=0.99, type=float, help='right clip')

    parser.add_argument(
        '--gamma', default=0.9, type=float, help='reward discount factor'
    )  # 0.8
    parser.add_argument(
        '--tau', default=0.01, type=float, help='soft update factor'
    )  # 0.1

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    env = Env(args=args, plot=False)
    agent = DPPG(env=env, args=args)
    env.make(args=args)

    def train(
        agent, env, num_episode=args.num_episode, num_steps_per_ep=args.num_steps_per_ep
    ):
        reward_record = []
        current_datetime = datetime.now()
        # Convert the month, day, hour, and minute to a string, excluding the year
        datetime_string = f"{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}-{current_datetime.minute:02}"

        if not os.path.exists(f'src/Training/{datetime_string}'):
            os.makedirs(f'src/Training/{datetime_string}')

        path = f'src/Training/{datetime_string}/e{args.num_episode}_s{args.num_steps_per_ep}'

        for i in range(num_episode):
            # reset the environment
            state = env.reset()
            # record time and current reward in this episode
            t1 = time.time()
            episode_reward = 0
            reward_set = []

            for j in range(num_steps_per_ep):
                # return an action based on the current state
                action = agent.select_action(state, exploration_noise=True)
                # print(action)
                # interact with the environment
                # ? value of reward should be scaled or not. Yes but not here
                state_, reward, done, info = env.step(action)
                # store the transition
                agent.store_transition(state, action, reward, state_, done)

                # update the network if the replay memory is full
                if agent.mpointer > args.warmup:
                    agent.update_network()

                # output records
                state = state_
                episode_reward += reward
                reward_set.append(reward)
                if j == num_steps_per_ep - 1 or done:
                    # if True:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i+1, num_episode, episode_reward, time.time() - t1
                        )
                    )
                    reward_record.append(episode_reward)

                if not j % 128:
                    # if True:
                    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, figsize=(10, 6))

                    ax1.plot(env.t_axis, env.G_values_array)
                    ax1.set_ylabel('Gx (T/mm)')
                    # ax1.set_ylim([[-1, 1] * 1e-5 * 40])
                    ax1.set_ylim([-1e-5 * 40, 1e-5 * 40])
                    ax2.plot(env.t_axis, env.k_traj)
                    ax2.set_ylabel('k (1/mm)')
                    # ax1.legend()
                    ax2.set_xticks(
                        [
                            0,
                            env.t_axis[int(env.N / 2) - 1],
                            env.t_axis[int(env.N * 1.5) - 1],
                        ]
                    )
                    ax2.set_xticklabels(['$t_1$', r'$t_2 (t_3)$', r'$t_4$'])
                    ax3.plot(env.x_axis, state[:int(len(state_)/2)], '-o', label='real')
                    ax3.plot(env.x_axis, state[int(len(state_)/2):], '-*', label='imag')
                    ax3.plot(env.x_axis, env.density, 'b-', label='object_real')
                    ax3.plot(env.x_axis, np.zeros(len(env.x_axis)), 'k-', label='object_imag')
                    ax3.legend()
                    ax4.plot(env.x_axis, env.density, '-', label='object')
                    ax4.plot(env.x_axis, np.abs(state[:int(len(state_)/2)] + 1j * state[int(len(state_)/2):]), '-o', label='reconstruction')
                    ax1.set_title(f'i_{i}_j_{j}_mse = {info}, reward = {reward}')
                    ax4.legend()
                    plt.savefig(path + f'i_{i}_j_{j}.png', dpi=300) 
                    # plt.show()
                    plt.close()
                    # plt.savefig(path + f'i_{i}_j_{j}.png', dpi=300) 

                if done:
                    break
            print(f'mean reward: {np.mean(reward_set)}, std: {np.std(reward_set)}')

        agent.save_model(path=path)

    train(agent=agent, env=env)


main()
